
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.input, encoding="utf-8") as f:
    frames = json.load(f)

results = []

for frame in sorted(frames, key=lambda x: x["frame_id"]):
    people = frame["instances"]

    if not people:
        raise ValueError(f"Frame {frame['frame_id']}: no person detected")

    # 選擇最大 Bounding Box 的人物
    def area(p):
        x1, y1, x2, y2 = p["bbox"][0]
        return (x2 - x1) * (y2 - y1)

    person = max(people, key=area)

    xy = person["keypoints"]
    scores = person["keypoint_scores"]

    assert len(xy) == len(scores) == 26

    keypoints = [
        value
        for (x, y), score in zip(xy, scores)
        for value in (x, y, score)
    ]

    results.append({
        "image_id": f"{frame['frame_id']}.jpg",
        "idx": 0,
        "keypoints": keypoints
    })

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(results, f)

print(f"Converted {len(results)} frames")
