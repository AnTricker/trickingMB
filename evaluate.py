#!/usr/bin/env python3
"""Batch GT-free consistency evaluation of prepared RTMPose/MotionBERT clips.

Example (Windows):
  python evaluate_tricking.py --input-dir ./analysis
  python evaluate_tricking.py --input-dir ./analysis --output-dir ./evaluation --confidence-threshold 0.5

Inputs: each immediate subfolder in --input-dir contains exactly one
*_prepared.npz and the corresponding *_metadata.json.
Outputs: grouped by metric into TemporalJerk/, OrientationJump/, BoneLengthCV/
under --output-dir (default: --input-dir). Shared batch summary and failure
frame tables are written at the output root; NO MP4 output.
Dependencies: numpy, matplotlib. No need for MMPose/OpenCV.
Errors are reported as [ERROR] and skipped; other clips continue processing.
"""
import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRIC_DIRS = ('TemporalJerk', 'OrientationJump', 'BoneLengthCV')

# Standard MotionBERT H36M-17 order; must verify upstream keypoint mapping.
# 0 pelvis; 1 R hip, 2 R knee, 3 R ankle; 4 L hip, 5 L knee, 6 L ankle;
# 7 spine, 8 thorax, 9 neck, 10 head;
# 11 L shoulder, 12 L elbow, 13 L wrist; 14 R shoulder, 15 R elbow, 16 R wrist.
BONES = [
    ('R_hip', 0, 1), ('R_thigh', 1, 2), ('R_shin', 2, 3),
    ('L_hip', 0, 4), ('L_thigh', 4, 5), ('L_shin', 5, 6),
    ('spine', 0, 7), ('torso', 7, 8), ('neck', 8, 9), ('head', 9, 10),
    ('L_clavicle', 8, 11), ('L_upperarm', 11, 12), ('L_forearm', 12, 13),
    ('R_clavicle', 8, 14), ('R_upperarm', 14, 15), ('R_forearm', 15, 16),
]


def read_json(p):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(p, obj):
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(p, rows, fields):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def stat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(np.percentile(x, 95))) if x.size else (None, None)


def unit(v, eps=1e-7):
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length <= eps:
        return None
    return v / length


def orientation_matrix(p):
    """Body-fixed frame from shoulders and hips; None for degenerate axes.

    x: averaged right-minus-left hip/shoulder lateral axis;
    y: upper torso direction orthogonal to x; z: x cross y.
    This is a joint-derived orientation PROXY, not SMPL/global camera yaw.
    """
    shoulder_r, shoulder_l = p[14], p[11]
    hip_r, hip_l = p[1], p[4]
    lateral = 0.5 * ((shoulder_r - shoulder_l) + (hip_r - hip_l))
    x = unit(lateral)
    if x is None:
        return None
    torso = 0.5 * (shoulder_r + shoulder_l) - 0.5 * (hip_r + hip_l)
    y = unit(torso - np.dot(torso, x) * x)
    if y is None:
        return None
    z = unit(np.cross(x, y))
    if z is None:
        return None
    return np.column_stack([x, y, z])


def metrics_from_poses(raw, normalized, fps, frame_ids, timestamps, conf, threshold, n_chunk):
    n = len(raw)
    # Difference defined by study: J_{t,j} = ||P_{t+2}-3P_{t+1}+3P_t-P_{t-1}|| / dt^3.
    # Stored at frame t (t=1..T-3); undefined at other frames.
    jerk_per_joint = np.full((n, 17), np.nan, dtype=float)
    diff3 = normalized[3:] - 3 * normalized[2:-1] + 3 * normalized[1:-2] - normalized[:-3]
    jerk_per_joint[1:n-2] = np.linalg.norm(diff3, axis=2) * fps**3
    jerk_frame = np.full(n, np.nan, dtype=float)
    jerk_frame[1:n-2] = np.mean(jerk_per_joint[1:n-2], axis=1)

    # Body axis changes in degrees BETWEEN frames; index t is t-1 -> t.
    rotation = [orientation_matrix(frame) for frame in raw]
    orientation = np.full(n, np.nan)
    for t in range(1, n):
        if rotation[t] is None or rotation[t-1] is None:
            continue
        cos_theta = np.clip((np.trace(rotation[t-1].T @ rotation[t]) - 1.0) / 2.0, -1., 1.)
        orientation[t] = np.degrees(np.arccos(cos_theta))

    # Bone CV is invariant to constant scale and translation, hence use raw.
    lengths = np.column_stack([np.linalg.norm(raw[:, a] - raw[:, b], axis=1) for _, a, b in BONES])
    bone_mean = lengths.mean(axis=0)
    if np.any(bone_mean <= 1e-9):
        raise ValueError('At least one bone has near-zero mean length; cannot calculate reliable CV')
    bone_cv = 100 * lengths.std(axis=0, ddof=0) / bone_mean
    bone_rows = [{'bone': name, 'joint_a': a, 'joint_b': b,
                  'mean_length_raw_units': float(bone_mean[i]), 'cv_percent': float(bone_cv[i])}
                 for i, (name, a, b) in enumerate(BONES)]

    jmean, jp95 = stat(jerk_frame)
    omean, op95 = stat(orientation)
    finite_jerk = np.isfinite(jerk_frame)
    finite_ori = np.isfinite(orientation)
    conf_avg = np.nanmean(conf, axis=1) if conf is not None else np.full(n, np.nan)
    conf_low = np.mean(conf < threshold, axis=1) if conf is not None else np.full(n, np.nan)
    rows = []
    bad_rows = []
    for t in range(n):
        j = float(jerk_frame[t]) if finite_jerk[t] else ''
        o = float(orientation[t]) if finite_ori[t] else ''
        jflag = bool(finite_jerk[t] and jp95 is not None and jerk_frame[t] >= jp95)
        oflag = bool(finite_ori[t] and op95 is not None and orientation[t] >= op95)
        boundary = bool(n_chunk > 0 and (t % n_chunk in (0, 1)) and t >= n_chunk)
        row = {'frame_idx': int(frame_ids[t]), 'timestamp_s': float(timestamps[t]),
               'jerk_mean': j, 'jerk_peak_joint': int(np.argmax(jerk_per_joint[t])) if finite_jerk[t] else '',
               'orientation_deg_per_frame': o,
               'mean_confidence_2d': float(conf_avg[t]) if np.isfinite(conf_avg[t]) else '',
               'fraction_low_confidence_2d': float(conf_low[t]) if np.isfinite(conf_low[t]) else '',
               'jerk_p95_flag': int(jflag), 'orientation_p95_flag': int(oflag),
               'chunk_boundary_candidate': int(boundary)}
        rows.append(row)
        if jflag or oflag:
            bad_rows.append(row)
    result = {
        'n_frames': n, 'fps': float(fps), 'dt_seconds': float(1/fps),
        'jerk_mean': jmean, 'jerk_p95': jp95,
        'orientation_mean_deg_per_frame': omean,
        'orientation_p95_deg_per_frame': op95,
        'orientation_valid_transitions': int(finite_ori.sum()),
        'orientation_invalid_transitions': int(n - 1 - finite_ori[1:].sum()),
        'bone_cv_mean_percent': float(bone_cv.mean()),
        'bone_cv_max_percent': float(bone_cv.max()),
        'bone_cv_max_bone': BONES[int(np.argmax(bone_cv))][0],
        'flagged_frames': len(bad_rows), 'body_orientation_proxy': 'shoulder/hip frame (H36M-17 assumed)',
        'jerk_units': 'clip-normalized body-length units / s^3 (NOT physical m/s^3)',
        'bone_cv_definition': '100 * std(population) / mean of each bone over full clip',
        'flags_definition': '>= within-clip P95; exploratory flags, NOT proven reconstruction errors',
    }
    return result, rows, bad_rows, bone_rows, jerk_per_joint


def figure_line(path, ts, values, mean, p95, ylabel, title):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts, values, linewidth=1.4, label='Frame value')
    if mean is not None:
        ax.axhline(mean, linestyle='--', linewidth=1, label=f'Mean: {mean:.3g}')
    if p95 is not None:
        ax.axhline(p95, linestyle=':', linewidth=1.5, label=f'P95: {p95:.3g}')
    ax.set(title=title, xlabel='Time (s)', ylabel=ylabel)
    ax.grid(alpha=.2)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_bones(path, bones, title):
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [b['bone'] for b in bones]
    vals = [b['cv_percent'] for b in bones]
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)), names, rotation=55, ha='right')
    ax.set(ylabel='Bone length CV (%)', title=title)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_folder(folder, out_root, args):
    pose_files = sorted(folder.glob('*_prepared.npz'))
    if len(pose_files) != 1:
        raise ValueError(f'Expected exactly 1 *_prepared.npz, found {len(pose_files)}')
    pose_file = pose_files[0]
    prefix = pose_file.name[:-len('_prepared.npz')]
    meta_file = folder / f'{prefix}_metadata.json'
    if not meta_file.is_file():
        raise FileNotFoundError(f'Missing {meta_file.name}')
    metadata = read_json(meta_file)
    with np.load(pose_file, allow_pickle=False) as data:
        required = ('pose_3d_raw', 'pose_3d_normalized', 'frame_idx', 'timestamps')
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f'Missing NPZ keys: {missing}')
        raw = np.asarray(data['pose_3d_raw'], dtype=float)
        norm = np.asarray(data['pose_3d_normalized'], dtype=float)
        frames = np.asarray(data['frame_idx'])
        ts = np.asarray(data['timestamps'], dtype=float)
        conf = np.asarray(data['confidence_2d'], dtype=float) if 'confidence_2d' in data else None
        fps_npz = float(data['fps']) if 'fps' in data else None
    if raw.ndim != 3 or raw.shape[1:] != (17, 3) or norm.shape != raw.shape:
        raise ValueError(f'Expected matching (T,17,3), got raw={raw.shape}, normalized={norm.shape}')
    n = raw.shape[0]
    if n < 4:
        raise ValueError('Need at least 4 frames for 3rd-order Jerk')
    if frames.shape != (n,) or ts.shape != (n,):
        raise ValueError('Frame indices or timestamps length mismatch')
    if conf is not None and conf.shape != (n,26):
        raise ValueError(f'Expected confidence_2d shape ({n},26), found {conf.shape}')
    if not np.isfinite(raw).all() or not np.isfinite(norm).all() or not np.isfinite(ts).all():
        raise ValueError('NaN or Inf in 3D poses/timestamps')
    if conf is not None and not np.isfinite(conf).all():
        raise ValueError('NaN or Inf in confidence_2d')
    if not np.all(np.diff(frames) == 1):
        raise ValueError('Frame indices not contiguous')
    fps = float(metadata['fps'])
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError('Invalid FPS')
    if fps_npz is not None and not np.isclose(fps, fps_npz, rtol=1e-4):
        raise ValueError('Metadata FPS does not match NPZ FPS')
    diffs = np.diff(ts)
    if np.any(diffs <= 0) or not np.allclose(diffs, 1/fps, rtol=.005, atol=1e-5):
        raise ValueError('Timestamps not uniformly sampled at FPS; jerk implementation assumes constant dt')
    if int(metadata['frame_count']) != n:
        raise ValueError('Metadata frame_count does not match NPZ')

    summary, rows, failures, bones, per_joint = metrics_from_poses(
        raw, norm, fps, frames, ts, conf, args.confidence_threshold, args.chunk_size)
    summary.update({'clip': folder.name, 'input_prefix': prefix,
                    'source_npz': str(pose_file), 'joint_order_note': metadata.get('joint_3d', 'unknown')})

    # Output folders are organized by ANALYSIS METHOD, not movement name.
    jerk_dir = out_root / 'TemporalJerk'
    orient_dir = out_root / 'OrientationJump'
    bone_dir = out_root / 'BoneLengthCV'
    for d in (jerk_dir, orient_dir, bone_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Per-frame tables separated into the relevant metric directory.
    common = ('frame_idx', 'timestamp_s', 'mean_confidence_2d',
              'fraction_low_confidence_2d', 'chunk_boundary_candidate')
    jerk_fields = list(common) + ['jerk_mean', 'jerk_peak_joint', 'jerk_p95_flag']
    orient_fields = list(common) + ['orientation_deg_per_frame', 'orientation_p95_flag']
    write_csv(jerk_dir / f'{prefix}_jerk.csv', rows, jerk_fields)
    write_csv(orient_dir / f'{prefix}_orientation.csv', rows, orient_fields)
    write_csv(bone_dir / f'{prefix}_bone_cv.csv', bones, list(bones[0]))
    joint_rows = [dict(frame_idx=int(frames[i]), timestamp_s=float(ts[i]),
                       **{f'joint_{j}': (float(per_joint[i,j]) if np.isfinite(per_joint[i,j]) else '')
                          for j in range(17)}) for i in range(n)]
    write_csv(jerk_dir / f'{prefix}_joint_jerk.csv', joint_rows, list(joint_rows[0]))

    jerks = [float(r['jerk_mean']) if r['jerk_mean'] != '' else np.nan for r in rows]
    angles = [float(r['orientation_deg_per_frame']) if r['orientation_deg_per_frame'] != '' else np.nan for r in rows]
    figure_line(jerk_dir / f'{prefix}_jerk.png', ts, jerks,
                summary['jerk_mean'], summary['jerk_p95'],
                'Normalized jerk (body scale / s³)', f'{folder.name}: Temporal Jerk')
    figure_line(orient_dir / f'{prefix}_orientation.png', ts, angles,
                summary['orientation_mean_deg_per_frame'], summary['orientation_p95_deg_per_frame'],
                'Orientation change (degree/frame)', f'{folder.name}: Orientation Jump')
    figure_bones(bone_dir / f'{prefix}_bone_cv.png', bones, f'{folder.name}: Bone Length CV')
    # Failure frames are combined across clips in a single root-level CSV.
    return summary, [dict(clip=folder.name, **row) for row in failures]


def main(argv=None):
    p = argparse.ArgumentParser(description='Evaluate all prepared Tricking clips (three GT-free metrics)')
    p.add_argument('--input-dir', type=Path, default=Path('./analysis'), help='Contains clip subfolders')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Output root; default same as input root; output grouped by analysis method')
    p.add_argument('--confidence-threshold', type=float, default=0.5,
                   help='Descriptive low-2D-score threshold; scores are NOT calibrated probabilities')
    p.add_argument('--chunk-size', type=int, default=243,
                   help='Candidate chunk boundaries (0 disables); not evidence of actual chunking')
    p.add_argument('--traceback', action='store_true', help='Print detailed tracebacks for failed clips')
    args = p.parse_args(argv)
    if not args.input_dir.is_dir():
        p.error(f'Input directory does not exist: {args.input_dir}')
    if args.chunk_size < 0:
        p.error('--chunk-size must be >= 0')
    out_root = args.output_dir if args.output_dir is not None else args.input_dir
    out_root.mkdir(parents=True, exist_ok=True)
    # Generated metric directories are never input clips, including on reruns
    # that change --output-dir after an earlier in-place evaluation.
    exclude = set(METRIC_DIRS)
    folders = sorted((d for d in args.input_dir.iterdir()
                      if d.is_dir() and d.name not in exclude), key=lambda d: d.name.lower())
    if not folders:
        print(f'[WARN] No subfolders in {args.input_dir}', flush=True)
        return 1
    summary_rows, all_failure_rows, errors = [], [], []
    for folder in folders:
        try:
            summary, failures = evaluate_folder(folder, out_root, args)
            summary_rows.append(summary)
            all_failure_rows.extend(failures)
            print(f'[OK] {folder.name}: {summary["n_frames"]} frames -> {out_root} ({len(METRIC_DIRS)} metrics)', flush=True)
        except Exception as e:
            error = {'clip': folder.name, 'error_type': type(e).__name__, 'message': str(e)}
            errors.append(error)
            print(f'[ERROR] {folder.name}: {type(e).__name__}: {e} (skipped)', flush=True)
            if args.traceback:
                traceback.print_exc()
    all_fields = ['clip', 'input_prefix', 'n_frames', 'fps', 'jerk_mean', 'jerk_p95',
                  'orientation_mean_deg_per_frame', 'orientation_p95_deg_per_frame',
                  'orientation_valid_transitions', 'orientation_invalid_transitions',
                  'bone_cv_mean_percent', 'bone_cv_max_percent', 'bone_cv_max_bone', 'flagged_frames']
    write_csv(out_root / 'metrics_summary.csv', summary_rows, all_fields)
    failure_fields = (['clip', 'frame_idx', 'timestamp_s', 'jerk_mean', 'jerk_peak_joint',
                       'orientation_deg_per_frame', 'mean_confidence_2d',
                       'fraction_low_confidence_2d', 'jerk_p95_flag',
                       'orientation_p95_flag', 'chunk_boundary_candidate'])
    write_csv(out_root / 'failure_frames.csv', all_failure_rows, failure_fields)
    dump_json(out_root / 'evaluation_details.json', summary_rows)
    dump_json(out_root / 'evaluation_errors.json', errors)
    print(f'Batch complete: {len(summary_rows)}/{len(folders)} OK', flush=True)
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
