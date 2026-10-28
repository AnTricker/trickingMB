
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
MODEL_SUFFIXES = ("_m", "_l", "_x")


def find_raw_video(json_stem, raw_dir):
    """
    Example:
      aerial_l.json -> aerial.mp4
      aerial_x.json -> aerial.mp4
    """
    video_stem = json_stem

    for suffix in MODEL_SUFFIXES:
        if json_stem.endswith(suffix):
            video_stem = json_stem[:-len(suffix)]
            break

    for ext in VIDEO_EXTENSIONS:
        candidate = raw_dir / f"{video_stem}{ext}"
        if candidate.is_file():
            return candidate

    # Also support uppercase extensions
    matches = [
        p for p in raw_dir.iterdir()
        if p.is_file()
        and p.stem == video_stem
        and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple raw videos match {json_stem}: {matches}"
        )

    raise FileNotFoundError(
        f"Raw video not found for {json_stem}. "
        f"Expected filename: {video_stem}.mp4"
    )


def run_script(script, video, json_path, out_dir, ref_3d=None):
    """Execute an existing MotionBERT inference script."""

    cmd = [
        sys.executable,
        str(ROOT / script),
        "--vid_path", str(video),
        "--json_path", str(json_path),
        "--out_path", str(out_dir),
    ]

    if ref_3d is not None:
        cmd += [
            "--ref_3d_motion_path",
            str(ref_3d),
        ]

    print(f"\nRunning: {script}", flush=True)

    subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=True,
    )


def process_video(
    json_path,
    raw_dir,
    x3d_dir,
    mesh_dir,
    skip_mesh=False,
    force=False,
):
    stem = json_path.stem
    video = find_raw_video(stem, raw_dir)

    x3d_video = x3d_dir / f"{stem}X3D.mp4"
    x3d_numpy = x3d_dir / f"{stem}X3D.npy"
    mesh_video = mesh_dir / f"{stem}Mesh.mp4"

    print(f"JSON : {json_path.name}")
    print(f"Video: {video.name}")

    # Temporary intermediate outputs
    with tempfile.TemporaryDirectory(
        prefix=f"motionbert_{stem}_"
    ) as tmp:
        tmp_dir = Path(tmp)

        # ---------------------------------
        # Step 1: 3D Skeleton Reconstruction
        # ---------------------------------
        needs_x3d = (
            force
            or not x3d_video.is_file()
            or not x3d_numpy.is_file()
        )

        if needs_x3d:
            temp_x3d_dir = tmp_dir / "X3D"

            run_script(
                script="infer_wild.py",
                video=video,
                json_path=json_path,
                out_dir=temp_x3d_dir,
            )

            source_mp4 = temp_x3d_dir / "X3D.mp4"
            source_npy = temp_x3d_dir / "X3D.npy"

            if not source_mp4.is_file():
                raise FileNotFoundError(
                    f"Missing generated file: {source_mp4}"
                )

            if not source_npy.is_file():
                raise FileNotFoundError(
                    f"Missing generated file: {source_npy}"
                )

            shutil.copy2(source_mp4, x3d_video)
            shutil.copy2(source_npy, x3d_numpy)

            print(f"[OK] {x3d_video.name}")
            print(f"[OK] {x3d_numpy.name}")

        else:
            print("[SKIP] Existing X3D outputs")

        # ---------------------------
        # Step 2: SMPL Mesh Recovery
        # ---------------------------
        if skip_mesh:
            print("[SKIP] Mesh generation disabled")
            return

        if mesh_video.is_file() and not force:
            print("[SKIP] Existing Mesh output")
            return

        temp_mesh_dir = tmp_dir / "mesh"

        run_script(
            script="infer_wild_mesh.py",
            video=video,
            json_path=json_path,
            out_dir=temp_mesh_dir,
            ref_3d=x3d_numpy,
        )

        source_mesh = temp_mesh_dir / "mesh.mp4"

        if not source_mesh.is_file():
            raise FileNotFoundError(
                f"Missing generated file: {source_mesh}"
            )

        shutil.copy2(source_mesh, mesh_video)

        print(f"[OK] {mesh_video.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch MotionBERT 3D and Mesh Inference"
    )

    parser.add_argument(
        "--input",
        default="input",
        help="Input root containing raw/ and alphapose/",
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Output root",
    )

    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Generate X3D only",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs",
    )

    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    raw_dir = input_root / "raw"
    alphapose_dir = input_root / "alphapose"

    if not raw_dir.is_dir():
        parser.error(f"Missing directory: {raw_dir}")

    if not alphapose_dir.is_dir():
        parser.error(f"Missing directory: {alphapose_dir}")

    json_files = sorted(
        alphapose_dir.glob("*.json")
    )

    if not json_files:
        parser.error(
            f"No AlphaPose JSON files found in {alphapose_dir}"
        )

    x3d_dir = output_root / "X3D"
    mesh_dir = output_root / "mesh"

    x3d_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_mesh:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MotionBERT Batch Inference")
    print(f"Videos : {len(json_files)}")
    print(f"Input  : {input_root}")
    print(f"Output : {output_root}")
    print(f"Mesh   : {not args.skip_mesh}")
    print("=" * 60)

    succeeded = []
    failed = []

    for i, json_path in enumerate(json_files, 1):
        print("\n" + "=" * 60)
        print(f"[{i}/{len(json_files)}] {json_path.stem}")
        print("=" * 60)

        try:
            process_video(
                json_path=json_path,
                raw_dir=raw_dir,
                x3d_dir=x3d_dir,
                mesh_dir=mesh_dir,
                skip_mesh=args.skip_mesh,
                force=args.force,
            )

            succeeded.append(json_path.name)
            print(f"[SUCCESS] {json_path.stem}")

        except Exception as exc:
            failed.append((json_path.name, str(exc)))

            print(f"[FAILED] {json_path.name}")
            print(f"Reason: {exc}")

            # Continue with the next JSON
            continue

    print("\n" + "=" * 60)
    print("Batch Summary")
    print("=" * 60)
    print(f"Total     : {len(json_files)}")
    print(f"Succeeded : {len(succeeded)}")
    print(f"Failed    : {len(failed)}")

    if failed:
        print("\nFailed files:")
        for name, reason in failed:
            print(f"  {name}: {reason}")

        raise SystemExit(1)

    print("\nAll inference tasks completed.")


if __name__ == "__main__":
    main()
