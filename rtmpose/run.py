
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

    if not isinstance(frames, list):
        raise ValueError("Invalid RTMPose JSON format.")

    results = []

    for frame in sorted(frames, key=lambda x: int(x["frame_id"])):
        frame_id = int(frame["frame_id"])
        people = frame["instances"]

        if not people:
            raise ValueError(
                f"Frame {frame_id}: no person detected."
            )

        # Select person with largest bounding box
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

    # MotionBERT expects continuous frame indices
    frame_ids = [int(f["frame_id"]) for f in frames]

    if sorted(frame_ids) != list(range(len(frames))):
        raise ValueError(
            "Frame IDs are not continuous from 0."
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="RTMPose inference + AlphaPose conversion"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--model",
        choices=["m", "l", "x"],
        default="m",
        help="RTMPose model size (default: m)"
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

    if not input_path.is_file():
        parser.error(f"Input video not found: {input_path}")

    output_root = Path(args.output).resolve()

    # Example: rodeoCorkscrew_l
    filename = f"{input_path.stem}_{args.model}"

    prediction_dir = output_root / "predictions"
    visualization_dir = output_root / "visualizations"
    alphapose_dir = output_root / "alphapose"

    for directory in [
        prediction_dir,
        visualization_dir,
        alphapose_dir
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    prediction_output = prediction_dir / f"{filename}.json"
    alphapose_output = alphapose_dir / f"{filename}.json"

    print(f"Input : {input_path}")
    print(f"Model : RTMPose-{args.model}")
    print(f"Device: {args.device}")

    # Temporary directory prevents collisions between models
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        tmp_prediction_dir = tmp_root / "predictions"
        tmp_visualization_dir = tmp_root / "visualizations"

        inferencer = MMPoseInferencer(
            pose2d=MODELS[args.model],
            device=args.device
        )

        print("\n[1/3] Running RTMPose inference...")

        for _ in inferencer(
            str(input_path),
            pred_out_dir=str(tmp_prediction_dir),
            vis_out_dir=str(tmp_visualization_dir),
            show=False
        ):
            pass

        # Find generated prediction JSON
        prediction_files = list(tmp_prediction_dir.glob("*.json"))

        if len(prediction_files) != 1:
            raise RuntimeError(
                f"Expected one prediction JSON, "
                f"found {len(prediction_files)}."
            )

        shutil.copy2(
            prediction_files[0],
            prediction_output
        )

        print(f"Prediction: {prediction_output}")

        print("\n[2/3] Saving visualization...")

        visualization_files = [
            p for p in tmp_visualization_dir.iterdir()
            if p.is_file()
        ] if tmp_visualization_dir.exists() else []

        if len(visualization_files) != 1:
            raise RuntimeError(
                f"Expected one visualization file, "
                f"found {len(visualization_files)}."
            )

        visualization_source = visualization_files[0]
        visualization_output = (
            visualization_dir /
            f"{filename}{visualization_source.suffix}"
        )

        shutil.copy2(
            visualization_source,
            visualization_output
        )

        print(f"Visualization: {visualization_output}")

    print("\n[3/3] Converting to AlphaPose format...")

    with open(prediction_output, encoding="utf-8") as f:
        frames = json.load(f)

    num_frames = convert_to_alphapose(
        frames,
        alphapose_output
    )

    print(f"AlphaPose: {alphapose_output}")

    print("\nRTMPose inference completed!")
    print(f"Converted {num_frames} frames.")
    print(f"Model: RTMPose-{args.model}")


if __name__ == "__main__":
    main()
