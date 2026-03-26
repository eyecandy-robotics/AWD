#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute convex hull meshes for all meshes in a folder."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing source meshes (searched recursively).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where convex hull meshes will be written.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".stl", ".obj", ".ply", ".dae"],
        help="Mesh extensions to process (default: .stl .obj .ply .dae).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def normalize_extensions(raw_extensions: list[str]) -> set[str]:
    normalized = set()
    for ext in raw_extensions:
        extension = ext.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.add(extension)
    return normalized


def load_as_single_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return None
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    extensions = normalize_extensions(args.extensions)
    if not extensions:
        raise ValueError("No valid mesh extensions provided.")

    try:
        import trimesh  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'trimesh'. Install it with: pip install trimesh"
        ) from exc

    source_files = [
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]

    if not source_files:
        print("No mesh files found. Nothing to do.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    for src in sorted(source_files):
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {dst}")
            skipped += 1
            continue

        try:
            mesh = load_as_single_mesh(src)
            if mesh is None or mesh.vertices is None or len(mesh.vertices) < 4:
                print(f"[SKIP] Not enough geometry: {src}")
                skipped += 1
                continue

            hull = mesh.convex_hull
            if hull.vertices is None or len(hull.vertices) < 4:
                print(f"[SKIP] Invalid convex hull: {src}")
                skipped += 1
                continue

            hull.export(dst)
            print(f"[OK] {src} -> {dst}")
            processed += 1
        except Exception as exc:
            print(f"[FAIL] {src}: {exc}")
            failed += 1

    print("\nSummary")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
