import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def read_paths(paths):
    values = []
    for path in paths:
        values.extend(
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return list(dict.fromkeys(values))


def farthest_indices(points, count):
    count = min(int(count), points.shape[0])
    selected = np.empty((count,), dtype=np.int64)
    selected[0] = 0
    min_distance = ((points - points[0]) ** 2.0).sum(axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(min_distance))
        distance = (
            (points - points[selected[index]]) ** 2.0
        ).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
    return selected


def build_one(item):
    (
        clean_path,
        output_path,
        region_count,
        points_per_region,
        candidate_count,
        overwrite,
        seed,
    ) = item
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return "skipped", str(output_path)

    clean = np.load(clean_path).astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    if 0 < candidate_count < clean.shape[0]:
        candidates = np.sort(
            rng.choice(
                clean.shape[0],
                size=candidate_count,
                replace=False,
            )
        )
    else:
        candidates = np.arange(clean.shape[0], dtype=np.int64)
    local_centers = farthest_indices(clean[candidates], region_count)
    center_indices = candidates[local_centers]
    _, neighbor_indices = cKDTree(clean).query(
        clean[center_indices],
        k=min(points_per_region, clean.shape[0]),
    )
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        center_indices=center_indices.astype(np.int32),
        neighbor_indices=neighbor_indices,
    )
    return "written", str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument(
        "--datalist",
        nargs="+",
        default=["datalist/train.txt", "datalist/validate.txt"],
    )
    parser.add_argument("--output-name", default="shape_regions.npz")
    parser.add_argument("--region-count", type=int, default=256)
    parser.add_argument("--points-per-region", type=int, default=64)
    parser.add_argument("--fps-candidates", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rel_paths = read_paths(args.datalist)
    jobs = []
    for index, rel_path in enumerate(rel_paths):
        clean_path = Path(args.clean_root) / rel_path / "clean.npy"
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        output_path = clean_path.parent / args.output_name
        jobs.append(
            (
                str(clean_path),
                str(output_path),
                args.region_count,
                args.points_per_region,
                args.fps_candidates,
                args.overwrite,
                args.seed + index,
            )
        )

    counts = {"written": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_one, job) for job in jobs]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            unit="shape",
            desc="Caching shape regions",
        ):
            status, _ = future.result()
            counts[status] += 1
    print(counts)


if __name__ == "__main__":
    main()
