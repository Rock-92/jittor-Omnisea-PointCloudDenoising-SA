import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import trimesh
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups


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


def build_surface_sample_metadata(vertices, faces, hidden_states, num_samples):
    perm = hidden_states["perm"].astype(np.int32, copy=False)
    face_samples = hidden_states["original_face_index"].astype(np.int32, copy=False)
    random_lengths = hidden_states["random_lengths"].astype(np.float32, copy=False)

    face_index = np.empty((num_samples,), dtype=np.int32)
    barycentric = np.empty((num_samples, 3), dtype=np.float32)

    vertex_face = np.full((vertices.shape[0],), -1, dtype=np.int32)
    vertex_slot = np.zeros((vertices.shape[0],), dtype=np.int32)
    unset = np.ones((vertices.shape[0],), dtype=np.bool_)
    for face_id, face in enumerate(faces):
        for local_slot, vertex_id in enumerate(face):
            if unset[vertex_id]:
                vertex_face[vertex_id] = face_id
                vertex_slot[vertex_id] = local_slot
                unset[vertex_id] = False

    num_vertex_samples = len(perm)
    face_index[:num_vertex_samples] = vertex_face[perm]
    missing = face_index[:num_vertex_samples] < 0
    if missing.any():
        face_index[:num_vertex_samples][missing] = 0
    barycentric[:num_vertex_samples] = 0.0
    barycentric[np.arange(num_vertex_samples), vertex_slot[perm]] = 1.0

    face_index[num_vertex_samples:] = face_samples
    uv = random_lengths.reshape(-1, 2)
    barycentric[num_vertex_samples:, 0] = 1.0 - uv[:, 0] - uv[:, 1]
    barycentric[num_vertex_samples:, 1] = uv[:, 0]
    barycentric[num_vertex_samples:, 2] = uv[:, 1]
    return face_index, barycentric


def cache_one(args: Tuple[str, str, str, str, str, str, str, int, int, bool]):
    (
        rel_path,
        input_dataset_dir,
        output_dir,
        mesh_name,
        output_name,
        mesh_output_name,
        surface_sample_name,
        num_samples,
        num_vertex_samples,
        overwrite,
    ) = args
    mesh_path = Path(input_dataset_dir) / rel_path / mesh_name
    output_path = Path(output_dir) / rel_path / output_name
    mesh_output_path = output_path.parent / mesh_output_name
    surface_sample_path = output_path.parent / surface_sample_name

    if (
        output_path.exists()
        and mesh_output_path.exists()
        and surface_sample_path.exists()
        and not overwrite
    ):
        return "skip", rel_path, None
    if not mesh_path.exists():
        return "missing", rel_path, str(mesh_path)

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    clean_points, _, _, hidden_states = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=num_samples,
        num_vertex_samples=num_vertex_samples,
    )
    clean_points = clean_points.astype(np.float32, copy=False)
    surface_face_index, surface_barycentric = build_surface_sample_metadata(
        vertices=vertices,
        faces=faces,
        hidden_states=hidden_states,
        num_samples=clean_points.shape[0],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp_path, clean_points)
    os.replace(tmp_path, output_path)

    tmp_mesh_path = mesh_output_path.with_name(
        f"{mesh_output_path.name}.tmp.{os.getpid()}.npz"
    )
    np.savez_compressed(
        tmp_mesh_path,
        vertices=vertices.astype(np.float32, copy=False),
        faces=faces.astype(np.int32, copy=False),
    )
    os.replace(tmp_mesh_path, mesh_output_path)

    tmp_surface_path = surface_sample_path.with_name(
        f"{surface_sample_path.name}.tmp.{os.getpid()}.npz"
    )
    np.savez_compressed(
        tmp_surface_path,
        face_index=surface_face_index,
        barycentric=surface_barycentric,
    )
    os.replace(tmp_surface_path, surface_sample_path)
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
    parser.add_argument("--mesh_output_name", default="mesh.npz")
    parser.add_argument("--surface_sample_name", default="surface_sample.npz")
    parser.add_argument("--num_samples", type=int, default=32768)
    parser.add_argument("--num_vertex_samples", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datalist_paths = [Path(p) for p in args.datalist]
    rel_paths = read_datalists(datalist_paths)
    tasks = [
        (
            rel_path,
            args.input_dataset_dir,
            args.output_dir,
            args.mesh_name,
            args.output_name,
            args.mesh_output_name,
            args.surface_sample_name,
            args.num_samples,
            args.num_vertex_samples,
            args.overwrite,
        )
        for rel_path in rel_paths
    ]

    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
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
        f"skip={counts.get('skip', 0)}, "
        f"missing={counts.get('missing', 0)}, "
        f"error={counts.get('error', 0)}"
    )


if __name__ == "__main__":
    main()
