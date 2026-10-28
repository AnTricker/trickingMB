
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from mmpose.apis import MMPoseInferencer


# RTMPose Body8-Halpe26
MODELS = {
    "m": "rtmpose-m_8xb512-700e_body8-halpe26-256x192",
    "l": "rtmpose-l_8xb512-700e_body8-halpe26-384x288",
    "x": "rtmpose-x_8xb256-700e_body8-halpe26-384x288",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def get_bbox_area(person):
    """Calculate bounding box area for person selection."""
    bbox = person.get("bbox")

    if not bbox:
        return 0.0

    if isinstance(bbox[0], list):
        bbox = bbox[0]

    x1, y1, x2, y2 = bbox

    return max(0, x2 - x1) * max(0, y2 - y1)


def convert_to_alphapose(frames, output_path):
    """Convert MMPose Halpe-26 JSON to AlphaPose format."""

    if isinstance(frames, dict) and "instance_info" in frames:
        frames = frames["instance_info"]

    if not isinstance(frames, list) or not frames:
        raise ValueError("Invalid or empty RTMPose JSON.")

    frames = sorted(frames, key=lambda x: int(x["frame_id"]))

    frame_ids = [int(frame["frame_id"]) for frame in frames]

    if frame_ids != list(range(len(frames))):
        raise ValueError(
            "Frame IDs must be continuous and start from 0."
        )

    results = []

    for frame in frames:
        frame_id = int(frame["frame_id"])
        people = frame["instances"]

        if not people:
            raise ValueError(
                f"Frame {frame_id}: no person detected."
            )

        # Select the person with the largest bounding box
        person = max(people, key=get_bbox_area)

        xy = person["keypoints"]
        scores = person["keypoint_scores"]

        if len(xy) != 26 or len(scores) != 26:
            raise ValueError(
                f"Frame {frame_id}: expected 26 keypoints, "
                f"got {len(xy)}."
            )

        keypoints = [
            float(value)
            for (x, y), score in zip(xy, scores)
            for value in (x, y, score)
        ]

        results.append({
            "image_id": f"{frame_id}.jpg",
            "idx": 0,
            "keypoints": keypoints
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    return len(results)


def collect_videos(input_path):
    """
    Accept either one video or a directory of videos.
    Directory mode scans the immediate folder only.
    """

    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {input_path.suffix}"
            )

        return [input_path]

    if input_path.is_dir():
        videos = sorted(
            (
                path for path in input_path.iterdir()
                if path.is_file()
                and path.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=lambda path: path.name.lower()
        )

        if not videos:
            raise ValueError(
                f"No supported video files found in: {input_path}"
            )

        return videos

    raise FileNotFoundError(
        f"Input file or directory not found: {input_path}"
    )


def process_video(
    input_path,
    model_size,
    output_root,
    inferencer
):
    """Run RTMPose and convert one video to AlphaPose JSON."""

    filename = f"{input_path.stem}_{model_size}"

    prediction_dir = output_root / "predictions"
    visualization_dir = output_root / "visualizations"
    alphapose_dir = output_root / "alphapose"

    for directory in (
        prediction_dir,
        visualization_dir,
        alphapose_dir
    ):
        directory.mkdir(parents=True, exist_ok=True)

    prediction_output = prediction_dir / f"{filename}.json"
    alphapose_output = alphapose_dir / f"{filename}.json"

    # Temporary outputs are isolated for each input video
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        tmp_prediction_dir = tmp_root / "predictions"
        tmp_visualization_dir = tmp_root / "visualizations"

        print("[1/3] Running RTMPose inference...")

        for _ in inferencer(
            str(input_path),
            pred_out_dir=str(tmp_prediction_dir),
            vis_out_dir=str(tmp_visualization_dir),
            show=False
        ):
            pass

        prediction_files = list(
            tmp_prediction_dir.rglob("*.json")
        )

        if len(prediction_files) != 1:
            raise RuntimeError(
                "Expected one prediction JSON, "
                f"found {len(prediction_files)}."
            )

        # Read before saving to avoid incomplete conversion
        with open(
            prediction_files[0],
            encoding="utf-8"
        ) as f:
            frames = json.load(f)

        print("[2/3] Converting to AlphaPose format...")

        num_frames = convert_to_alphapose(
            frames,
            alphapose_output
        )

        print("[3/3] Saving visualization...")

        visualization_files = [
            path
            for path in tmp_visualization_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ]

        if len(visualization_files) != 1:
            raise RuntimeError(
                "Expected one visualization video, "
                f"found {len(visualization_files)}."
            )

        visualization_source = visualization_files[0]

        visualization_output = (
            visualization_dir
            / f"{filename}{visualization_source.suffix}"
        )

        shutil.copy2(
            prediction_files[0],
            prediction_output
        )

        shutil.copy2(
            visualization_source,
            visualization_output
        )

    print(f"Prediction   : {prediction_output}")
    print(f"Visualization: {visualization_output}")
    print(f"AlphaPose    : {alphapose_output}")
    print(f"Frames       : {num_frames}")

    return num_frames


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RTMPose inference + AlphaPose conversion "
            "(single video or directory)"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video file or directory"
    )

    parser.add_argument(
        "--model",
        choices=["m", "l", "x"],
        default="m",
        help="RTMPose size: m, l, x (default: m)"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Output directory (default: output)"
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device (default: cuda:0)"
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    try:
        videos = collect_videos(input_path)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print("=" * 60)
    print(f"Input     : {input_path}")
    print(f"Output    : {output_root}")
    print(f"Model     : RTMPose-{args.model}")
    print(f"Device    : {args.device}")
    print(f"Videos    : {len(videos)}")
    print("=" * 60)

    print("\nLoading RTMPose model...")

    # Load model only once for the entire batch
    inferencer = MMPoseInferencer(
        pose2d=MODELS[args.model],
        device=args.device
    )

    succeeded = []
    failed = []

    for index, video in enumerate(videos, start=1):
        print("\n" + "=" * 60)
        print(
            f"[{index}/{len(videos)}] "
            f"Processing: {video.name}"
        )
        print("=" * 60)

        try:
            num_frames = process_video(
                input_path=video,
                model_size=args.model,
                output_root=output_root,
                inferencer=inferencer
            )

            succeeded.append((video.name, num_frames))

            print(f"[OK] {video.name}")

        except Exception as exc:
            failed.append((video.name, str(exc)))

            print(f"[FAILED] {video.name}")
            print(f"Reason: {exc}")

            # Continue processing remaining videos
            continue

    print("\n" + "=" * 60)
    print("RTMPose Batch Summary")
    print("=" * 60)

    print(f"Total     : {len(videos)}")
    print(f"Succeeded : {len(succeeded)}")
    print(f"Failed    : {len(failed)}")

    if failed:
        print("\nFailed videos:")

        for name, reason in failed:
            print(f"  - {name}: {reason}")

    print("\nBatch processing completed.")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
