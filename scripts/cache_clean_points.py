import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups


def compute_edge_geom_labels(points, num_classes=12, knn=32):
    k = max(8, min(int(knn), points.shape[0]))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    curv = np.zeros((points.shape[0],), dtype=np.float64)
    normals = np.zeros((points.shape[0], 3), dtype=np.float64)

    for i, inds in enumerate(idx):
        pts = points[inds]
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        total = float(vals.sum())
        if total > 1e-12:
            curv[i] = vals[0] / total
        normal = vecs[:, 0]
        normals[i] = normal / max(np.linalg.norm(normal), 1e-12)

    normal_angle = np.zeros_like(curv)
    for i, inds in enumerate(idx):
        dots = np.abs(normals[inds] @ normals[i])
        dots = np.clip(dots, 0.0, 1.0)
        normal_angle[i] = np.percentile(np.arccos(dots), 90)

    score = curv + 0.25 * normal_angle
    order = np.argsort(score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.shape[0])
    labels = np.floor(ranks * int(num_classes) / max(points.shape[0], 1)).astype(np.int32)
    return np.clip(labels, 0, int(num_classes) - 1).astype(np.int32, copy=False)


def read_datalists(paths: Iterable[Path]) -> List[str]:
    seen = set()
    rel_paths: List[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rel = line.strip().replace("\\", "/")
                if not rel or rel.startswith("#") or rel in seen:
                    continue
                seen.add(rel)
                rel_paths.append(rel)
    return rel_paths


def atomic_save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def cache_one(args: Tuple[str, str, str, str, str, str, int, int, int, int, bool, int]):
    (
        rel_path,
        input_dataset_dir,
        output_dir,
        mesh_name,
        output_name,
        edge_geom_label_name,
        num_samples,
        num_vertex_samples,
        num_edge_geom_classes,
        edge_geom_knn,
        overwrite,
        seed,
    ) = args
    np.random.seed(seed)
    mesh_path = Path(input_dataset_dir) / rel_path / mesh_name
    output_path = Path(output_dir) / rel_path / output_name
    label_path = Path(output_dir) / rel_path / edge_geom_label_name

    if output_path.exists() and label_path.exists() and not overwrite:
        return "skip", rel_path, None
    if output_path.exists() and not label_path.exists() and not overwrite:
        clean_points = np.load(output_path).astype(np.float32, copy=False)
        labels = compute_edge_geom_labels(
            clean_points,
            num_classes=num_edge_geom_classes,
            knn=edge_geom_knn,
        )
        atomic_save_npy(label_path, labels)
        return "label", rel_path, labels.shape
    if not mesh_path.exists():
        return "missing", rel_path, str(mesh_path)

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    clean_points, _, _, _ = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=num_samples,
        num_vertex_samples=num_vertex_samples,
    )
    clean_points = clean_points.astype(np.float32, copy=False)
    labels = compute_edge_geom_labels(
        clean_points,
        num_classes=num_edge_geom_classes,
        knn=edge_geom_knn,
    )

    atomic_save_npy(output_path, clean_points)
    atomic_save_npy(label_path, labels)
    return "ok", rel_path, clean_points.shape


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache clean point clouds from ShapeNet OBJ meshes."
    )
    parser.add_argument(
        "--input_dataset_dir",
        default="dataset_clean",
        help="Root containing shapenet/<synset>/<model>/models/model_normalized.obj.",
    )
    parser.add_argument(
        "--output_dir",
        default="cache_clean_points",
        help="Root where shapenet/<synset>/<model>/clean.npy will be written.",
    )
    parser.add_argument(
        "--datalist",
        nargs="+",
        default=["datalist/train.txt", "datalist/validate.txt"],
        help="One or more datalist files with relative model directories.",
    )
    parser.add_argument("--mesh_name", default="models/model_normalized.obj")
    parser.add_argument("--output_name", default="clean.npy")
    parser.add_argument("--edge_geom_label_name", default="edge_geom_label.npy")
    parser.add_argument("--num_edge_geom_classes", type=int, default=12)
    parser.add_argument("--edge_geom_knn", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=32768)
    parser.add_argument("--num_vertex_samples", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    datalist_paths = [Path(p) for p in args.datalist]
    rel_paths = read_datalists(datalist_paths)
    tasks = [
        (
            rel_path,
            args.input_dataset_dir,
            args.output_dir,
            args.mesh_name,
            args.output_name,
            args.edge_geom_label_name,
            args.num_samples,
            args.num_vertex_samples,
            args.num_edge_geom_classes,
            args.edge_geom_knn,
            args.overwrite,
            args.seed + i,
        )
        for i, rel_path in enumerate(rel_paths)
    ]

    counts = {"ok": 0, "label": 0, "skip": 0, "missing": 0, "error": 0}
    if args.workers <= 1:
        iterator = map(cache_one, tasks)
        for status, rel_path, detail in tqdm(iterator, total=len(tasks)):
            counts[status] = counts.get(status, 0) + 1
            if status in {"missing", "error"}:
                print(f"{status}: {rel_path} ({detail})")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(cache_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    status, rel_path, detail = future.result()
                except Exception as exc:
                    status, rel_path, detail = "error", "<unknown>", repr(exc)
                counts[status] = counts.get(status, 0) + 1
                if status in {"missing", "error"}:
                    print(f"{status}: {rel_path} ({detail})")

    print(
        "cache clean points done: "
        f"ok={counts.get('ok', 0)}, "
        f"label={counts.get('label', 0)}, "
        f"skip={counts.get('skip', 0)}, "
        f"missing={counts.get('missing', 0)}, "
        f"error={counts.get('error', 0)}"
    )


if __name__ == "__main__":
    main()
