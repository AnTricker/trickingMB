#!/usr/bin/env python3
"""Prepare RTMPose JSON + MotionBERT NPY for GT-free evaluation.

Only produces *_prepared.npz, *_metadata.json and batch_summary.json.
Each video matches a JSON/NPY with the same filename prefix followed by
an optional suffix (e.g. aerial_x.json, aerial-lX3D.npy).
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}


def match_name(video_stem, result_stem):
    base = re.sub(r'\s*\(\d+\)$', '', video_stem).casefold()
    result = result_stem.casefold()
    return result == base or any(result.startswith(base + sep) for sep in ('_', '-', '.', ' '))


def find_result(video, candidates, kind):
    matches = [p for p in candidates if match_name(video.stem, p.stem)]
    if not matches:
        raise FileNotFoundError(f'Missing {kind} for {video.stem}')
    if len(matches) > 1:
        raise ValueError(f'Multiple {kind} matches for {video.stem}: {[p.name for p in matches]}')
    return matches[0]


def prepare(video, json_path, npy_path, out_root):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Container counts may include samples excluded by the MP4 edit list.
        # Match the frames actually delivered to pose inference by the decoder.
        frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
    finally:
        cap.release()
    if fps <= 0 or frames <= 0:
        raise ValueError('Invalid video FPS or frame count')

    records = json.loads(json_path.read_text(encoding='utf-8-sig'))
    xy = np.full((frames, 26, 2), np.nan, dtype=np.float32)
    conf = np.full((frames, 26), np.nan, dtype=np.float32)
    for item in records:
        m = re.search(r'(\d+)(?=\.[^.]+$)', str(item['image_id']))
        if not m:
            raise ValueError(f'Invalid image_id: {item["image_id"]}')
        i = int(m.group(1))
        if not 0 <= i < frames:
            raise ValueError(f'JSON frame {i} outside video range')
        if np.isfinite(xy[i]).all():
            raise ValueError(f'Duplicate frame {i} in JSON')
        joints = np.asarray(item['keypoints'], dtype=np.float32).reshape(26, 3)
        xy[i], conf[i] = joints[:, :2], joints[:, 2]
    missing = np.flatnonzero(~np.isfinite(xy).all(axis=(1, 2)))
    if len(missing):
        raise ValueError(f'Missing/invalid 2D frames: {missing[:10].tolist()} ({len(missing)} total)')

    pose = np.asarray(np.load(npy_path, allow_pickle=False), dtype=np.float32)
    if pose.shape != (frames, 17, 3) or not np.isfinite(pose).all():
        raise ValueError(f'Invalid 3D array: {pose.shape}, expected ({frames},17,3) with finite values')

    centered = pose - pose[:, 0:1, :]
    # A single scale for the whole clip: median of the average left/right leg lengths.
    leg_pairs = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    lengths = np.stack([np.linalg.norm(centered[:, a] - centered[:, b], axis=1)
                        for a, b in leg_pairs], axis=1)
    scale = float(np.median(lengths.reshape(frames, 2, 3).sum(axis=2).mean(axis=1)))
    if not np.isfinite(scale) or scale <= 1e-9:
        raise ValueError('Invalid body scale')

    out = out_root / video.stem
    out.mkdir(parents=True, exist_ok=True)
    name = video.stem
    np.savez_compressed(
        out / f'{name}_prepared.npz',
        pose_2d=xy, confidence_2d=conf,
        pose_3d_raw=pose, pose_3d_pelvis_centered=centered,
        pose_3d_normalized=centered / scale,
        frame_idx=np.arange(frames, dtype=np.int32),
        timestamps=np.arange(frames, dtype=np.float64) / fps,
        fps=np.float64(fps), body_scale=np.float64(scale),
    )
    metadata = {
        'video_path': str(video), 'json_path': str(json_path), 'pose3d_path': str(npy_path),
        'fps': fps, 'frame_count': frames, 'width': width, 'height': height,
        'reported_frame_count': reported_frames, 'decoded_frame_count': frames,
        'body_scale_in_raw_units': scale, 'joint_3d': 'H36M-17 assumed',
        'normalization': 'pelvis-centered / fixed clip-level body scale',
    }
    (out / f'{name}_metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return frames, out


def main():
    p = argparse.ArgumentParser(description='Batch preparation for MotionBERT evaluation')
    p.add_argument('--video-dir', type=Path, required=True)
    p.add_argument('--json-dir', type=Path, required=True)
    p.add_argument('--pose3d-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()

    for folder in (args.video_dir, args.json_dir, args.pose3d_dir):
        if not folder.is_dir():
            p.error(f'Folder does not exist: {folder}')

    videos = sorted((f for f in args.video_dir.rglob('*') if f.suffix.lower() in VIDEO_EXTS),
                    key=lambda f: str(f).casefold())
    jsons = [f for f in args.json_dir.rglob('*') if f.suffix.lower() == '.json']
    npys = [f for f in args.pose3d_dir.rglob('*') if f.suffix.lower() == '.npy']
    summary = []
    for video in videos:
        try:
            js = find_result(video, jsons, 'JSON')
            npy = find_result(video, npys, 'NPY')
            n, out = prepare(video, js, npy, args.output_dir)
            print(f'[OK] {video.stem}: {n} frames -> {out}', flush=True)
            summary.append({'video': video.stem, 'status': 'ok', 'frames': n})
        except Exception as e:
            print(f'[ERROR] {video.name}: {e} (skipped)', flush=True)
            summary.append({'video': video.stem, 'status': 'error', 'reason': str(e)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'batch_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Batch complete: {sum(row["status"] == "ok" for row in summary)}/{len(videos)} OK', flush=True)


if __name__ == '__main__':
    main()
