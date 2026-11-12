"""Run GVHMR recursively and collect its combined-view videos."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


GVHMR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = GVHMR_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "input",
                        help="Input directory (recursively searches for .mp4 files).")
    parser.add_argument("--output-root", type=Path, default=GVHMR_ROOT / "outputs",
                        help="Root directory for all raw GVHMR outputs.")
    parser.add_argument("--summary-root", type=Path,
                        default=PROJECT_ROOT / "output" / "Output_InCam_Global_Horizon",
                        help="Root directory for copies of combined-view videos.")
    parser.add_argument("-s", "--static-cam", action="store_true",
                        help="Use a static camera; skip camera motion estimation.")
    return parser.parse_args()


def discover_videos(input_root):
    """Reject output collisions before running any jobs or copying files."""
    videos = sorted(
        (path for path in input_root.rglob("*")
         if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.relative_to(input_root).as_posix(),
    )
    seen = {}
    for video in videos:
        relative = video.relative_to(input_root)
        # Also reject case-only differences for portable Windows output paths.
        key = (relative.parent / relative.stem).as_posix().casefold()
        if key in seen:
            raise ValueError(f"Conflicting output paths: {seen[key]} and {video}")
        seen[key] = video
    return videos


def copy_summary(source, destination):
    """Create a copy exclusively; remove an incomplete copy on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        target = destination.open("xb")
    except FileExistsError:
        if destination.is_file():
            return False
        raise
    try:
        with target, source.open("rb") as original:
            shutil.copyfileobj(original, target)
        shutil.copystat(source, destination)
    except BaseException:
        destination.unlink()
        raise
    return True


def main():
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    summary_root = args.summary_root.expanduser().resolve()
    if not input_root.is_dir():
        print(f"[ERROR] Input directory not found: {input_root}", file=sys.stderr)
        return 1
    # Generated .mp4 files must never become inputs on subsequent runs.
    for root in (output_root, summary_root):
        if root == input_root or input_root in root.parents:
            print(f"[ERROR] Output root must be outside input: {root}", file=sys.stderr)
            return 1
    try:
        videos = discover_videos(input_root)
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1

    counts = dict(inference=0, reused=0, copied=0, skipped=0, failed=0)
    print(f"Found {len(videos)} input videos in {input_root}", flush=True)
    for index, video in enumerate(videos, start=1):
        relative = video.relative_to(input_root)
        category_root = output_root / relative.parent
        result_dir = category_root / video.stem
        result_video = result_dir / f"{video.stem}_3_incam_global_horiz.mp4"
        summary_video = summary_root / relative.parent / result_video.name
        print(f"[{index}/{len(videos)}] {relative}", flush=True)
        try:
            if result_video.is_file():
                counts["reused"] += 1
                print("  [REUSE] Combined video exists; skipping GVHMR.", flush=True)
            else:
                has_prediction = (result_dir / "hmr4d_results.pt").is_file()
                if has_prediction:
                    counts["reused"] += 1
                    print("  [RESUME] Reuse prediction; complete rendering/merge.", flush=True)
                else:
                    counts["inference"] += 1
                    print("  [INFER] Run GVHMR.", flush=True)
                category_root.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "tools/demo/demo.py",
                    "--video", video.as_posix(),
                    "--output_root", category_root.as_posix(),
                ]
                if args.static_cam:
                    command.append("-s")
                subprocess.run(command, cwd=GVHMR_ROOT, check=True)

            if not result_video.is_file():
                raise FileNotFoundError(f"Combined video was not produced: {result_video}")
            if copy_summary(result_video, summary_video):
                counts["copied"] += 1
                print(f"  [COPY] {summary_video}", flush=True)
            else:
                counts["skipped"] += 1
                print(f"  [SKIP COPY] {summary_video}", flush=True)
        except (OSError, subprocess.CalledProcessError) as error:
            counts["failed"] += 1
            print(f"  [ERROR] {relative}: {error}", file=sys.stderr, flush=True)

    # Inference/reuse count selected actions, including failed attempts;
    # copied/skipped count successful summary handling. Counts can overlap.
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()),
          flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
