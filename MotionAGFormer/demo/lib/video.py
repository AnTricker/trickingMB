"""Shared video opening policy for inference and visualization."""
import cv2


def open_video(path):
    # FFmpeg exposes MP4 display-matrix rotation to OpenCV. Let the decoder
    # apply it exactly once, before detection and coordinate normalization.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Cannot open video: {path}')
    enabled = cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not enabled and cap.get(cv2.CAP_PROP_ORIENTATION_META) % 360:
        cap.release()
        raise RuntimeError('OpenCV cannot apply video rotation; use an FFmpeg-enabled OpenCV build.')
    return cap
