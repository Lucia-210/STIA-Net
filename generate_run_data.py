import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


EPS = 1e-8
SEED = 42
CHANNEL_SPECS = (
    ("single", "T11.bin"),
    ("magnitude", "T12_real.bin", "T12_imag.bin"),
    ("magnitude", "T13_real.bin", "T13_imag.bin"),
    ("single", "T22.bin"),
    ("magnitude", "T23_real.bin", "T23_imag.bin"),
    ("single", "T33.bin"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run.py training data from multitemporal T3 directories.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing time directories and label mat.")
    parser.add_argument("--label-file", required=True, help="MAT file containing label matrix.")
    parser.add_argument("--label-key", default="GroundTruth", help="Variable name of the label matrix in the MAT file.")
    parser.add_argument("--output-root", required=True, help="Output directory containing TrainData/ and ValidationData/.")
    parser.add_argument("--train-rate", type=float, required=True, help="Training sampling rate over all pixels.")
    parser.add_argument("--val-rate", type=float, required=True, help="Validation sampling rate over all pixels.")
    parser.add_argument("--block-size", type=int, required=True, help="Patch size.")
    parser.add_argument("--time-count", type=int, required=True, help="Number of time directories to use.")
    parser.add_argument("--time-name-length", type=int, default=0, help="Restrict time directory name length; 0 disables filtering.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of samples processed per patch batch.")
    return parser.parse_args()


def read_config_shape(config_path: Path) -> tuple[int, int]:
    lines = [line.strip() for line in config_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row_idx = lines.index("Nrow")
    col_idx = lines.index("Ncol")
    return int(lines[row_idx + 1]), int(lines[col_idx + 1])


def list_time_dirs(root: Path, time_count: int, time_name_length: int) -> list[Path]:
    time_dirs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if time_name_length and len(path.name) != time_name_length:
            continue
        if not path.name.isdigit():
            continue
        if (path / "T3" / "config.txt").exists():
            time_dirs.append(path)
    time_dirs.sort(key=lambda p: p.name)
    return time_dirs[:time_count]


def read_label_mat(label_path: Path, label_key: str, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        mat = sio.loadmat(label_path)
        label = np.array(mat[label_key])
    except NotImplementedError:
        with h5py.File(label_path, "r") as f:
            label = np.array(f[label_key])

    if label.shape == expected_shape[::-1]:
        label = label.T
    if label.shape != expected_shape:
        raise RuntimeError(f"Label shape {label.shape} does not match expected {expected_shape}")
    return label.astype(np.uint8, copy=False)


def remap_ground_truth(gt: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    valid_labels = sorted(int(v) for v in np.unique(gt) if v > 0)
    mapping = {src: idx + 1 for idx, src in enumerate(valid_labels)}
    lookup = np.zeros(int(max(valid_labels)) + 1, dtype=np.uint8)
    for src, dst in mapping.items():
        lookup[src] = dst
    return lookup[gt], mapping


def sample_positions(mapped_gt: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = mapped_gt.shape
    total = rows * cols
    picked = rng.choice(total, size=int(total * rate), replace=False)
    labels = mapped_gt.reshape(-1)[picked]
    valid = labels > 0
    picked = picked[valid]
    labels = labels[valid]
    rr = picked // cols
    cc = picked % cols
    coords = np.stack([rr, cc], axis=1).astype(np.int32)
    return coords, labels.astype(np.uint8)


def write_one_hot(ds: h5py.Dataset, labels: np.ndarray, class_count: int) -> None:
    one_hot = np.zeros((class_count, labels.shape[0]), dtype=np.float32)
    one_hot[labels.astype(np.int64) - 1, np.arange(labels.shape[0])] = 1.0
    ds[...] = one_hot


def create_output_file(path: Path, block_size: int, time_count: int, sample_count: int, class_count: int) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    train_data = f.create_dataset(
        "TrainData",
        shape=(6, block_size, block_size, time_count, sample_count),
        dtype="float32",
        chunks=(1, block_size, block_size, 1, min(sample_count, 128)),
    )
    train_label = f.create_dataset(
        "TrainLabelPro",
        shape=(class_count, sample_count),
        dtype="float32",
        chunks=(class_count, min(sample_count, 1024)),
    )
    coords = f.create_dataset("SampleCoords", shape=(2, sample_count), dtype="int32")
    return f, train_data, train_label, coords


def build_memmaps(t3_dir: Path, shape: tuple[int, int]) -> dict[str, np.memmap]:
    rows, cols = shape
    memmaps = {}
    for spec in CHANNEL_SPECS:
        for name in spec[1:]:
            if name not in memmaps:
                memmaps[name] = np.memmap(t3_dir / name, dtype="<f4", mode="r", shape=(rows, cols))
    return memmaps


def transform_chunk(spec: tuple[str, ...], memmaps: dict[str, np.memmap], row_slice: slice) -> np.ndarray:
    if spec[0] == "single":
        data = memmaps[spec[1]][row_slice]
        return np.log(np.abs(data).astype(np.float32) + EPS)
    if spec[0] == "magnitude":
        data_a = memmaps[spec[1]][row_slice]
        data_b = memmaps[spec[2]][row_slice]
        return np.log(np.hypot(data_a, data_b).astype(np.float32) + EPS)
    raise ValueError(f"Unsupported channel spec: {spec}")


def compute_channel_stats(memmaps: dict[str, np.memmap], rows: int, row_chunk: int = 1024) -> list[tuple[float, float]]:
    stats = []
    for spec in CHANNEL_SPECS:
        min_v = None
        max_v = None
        for start in range(0, rows, row_chunk):
            stop = min(start + row_chunk, rows)
            chunk = transform_chunk(spec, memmaps, slice(start, stop))
            chunk_min = float(chunk.min())
            chunk_max = float(chunk.max())
            min_v = chunk_min if min_v is None else min(min_v, chunk_min)
            max_v = chunk_max if max_v is None else max(max_v, chunk_max)
        stats.append((min_v, max_v))
    return stats


def normalize_values(values: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    if max_v > min_v:
        return ((values - min_v) / (max_v - min_v)).astype(np.float32, copy=False)
    out = np.zeros_like(values, dtype=np.float32)
    return out


def extract_channel_patches(
    spec: tuple[str, ...],
    memmaps: dict[str, np.memmap],
    rr: np.ndarray,
    cc: np.ndarray,
    min_v: float,
    max_v: float,
) -> np.ndarray:
    if spec[0] == "single":
        values = np.log(np.abs(memmaps[spec[1]][rr, cc]).astype(np.float32) + EPS)
    else:
        values = np.log(np.hypot(memmaps[spec[1]][rr, cc], memmaps[spec[2]][rr, cc]).astype(np.float32) + EPS)
    return normalize_values(values, min_v, max_v)


def extract_into_dataset(
    ds: h5py.Dataset,
    memmaps: dict[str, np.memmap],
    stats: list[tuple[float, float]],
    coords: np.ndarray,
    time_idx: int,
    block_size: int,
    batch_size: int,
    shape: tuple[int, int],
) -> None:
    rows, cols = shape
    half = block_size // 2
    offset = np.arange(block_size, dtype=np.int32)

    for start in range(0, coords.shape[0], batch_size):
        end = min(start + batch_size, coords.shape[0])
        batch = coords[start:end]
        r0 = np.clip(batch[:, 0] - half, 0, rows - block_size)
        c0 = np.clip(batch[:, 1] - half, 0, cols - block_size)
        rr = r0[:, None, None] + offset[None, :, None]
        cc = c0[:, None, None] + offset[None, None, :]

        for channel_idx, spec in enumerate(CHANNEL_SPECS):
            min_v, max_v = stats[channel_idx]
            ds[channel_idx, :, :, time_idx, start:end] = np.transpose(
                extract_channel_patches(spec, memmaps, rr, cc, min_v, max_v),
                (1, 2, 0),
            )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    time_dirs = list_time_dirs(data_root, args.time_count, args.time_name_length)
    if len(time_dirs) < args.time_count:
        raise RuntimeError(f"Only found {len(time_dirs)} time directories under {data_root}")

    rows, cols = read_config_shape(time_dirs[0] / "T3" / "config.txt")
    ground_truth = read_label_mat(data_root / args.label_file, args.label_key, (rows, cols))
    mapped_gt, mapping = remap_ground_truth(ground_truth)
    class_count = len(mapping)

    rng = np.random.default_rng(SEED)
    train_coords, train_labels = sample_positions(mapped_gt, args.train_rate, rng)
    val_coords, val_labels = sample_positions(mapped_gt, args.val_rate, rng)

    train_path = output_root / "TrainData" / "TrainData.mat"
    val_path = output_root / "ValidationData" / "TrainData.mat"
    train_file, train_ds, train_label_ds, train_coord_ds = create_output_file(
        train_path, args.block_size, args.time_count, train_coords.shape[0], class_count
    )
    val_file, val_ds, val_label_ds, val_coord_ds = create_output_file(
        val_path, args.block_size, args.time_count, val_coords.shape[0], class_count
    )

    try:
        write_one_hot(train_label_ds, train_labels, class_count)
        write_one_hot(val_label_ds, val_labels, class_count)
        train_coord_ds[...] = train_coords.T
        val_coord_ds[...] = val_coords.T

        for time_idx, time_dir in enumerate(time_dirs):
            print(f"Loading time {time_idx + 1}/{args.time_count}: {time_dir.name}", flush=True)
            memmaps = build_memmaps(time_dir / "T3", (rows, cols))
            stats = compute_channel_stats(memmaps, rows)
            print(f"Writing train patches for {time_dir.name}", flush=True)
            extract_into_dataset(
                train_ds, memmaps, stats, train_coords, time_idx, args.block_size, args.batch_size, (rows, cols)
            )
            print(f"Writing validation patches for {time_dir.name}", flush=True)
            extract_into_dataset(
                val_ds, memmaps, stats, val_coords, time_idx, args.block_size, args.batch_size, (rows, cols)
            )

        meta = {
            "data_root": str(data_root),
            "time_dirs": [time_dir.name for time_dir in time_dirs],
            "block_size": args.block_size,
            "train_rate": args.train_rate,
            "val_rate": args.val_rate,
            "seed": SEED,
            "class_count": class_count,
            "label_mapping": mapping,
            "train_samples": int(train_coords.shape[0]),
            "validation_samples": int(val_coords.shape[0]),
        }
        (output_root / "TrainData" / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_root / "ValidationData" / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    finally:
        train_file.close()
        val_file.close()


if __name__ == "__main__":
    main()
