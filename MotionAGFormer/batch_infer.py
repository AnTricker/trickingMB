"""Run the existing MotionAGFormer demo once for each video in a folder."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Folder containing videos (non-recursive)')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        parser.error(f'Folder does not exist: {folder}')

    root = Path(__file__).resolve().parent
    videos = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}),
        key=lambda p: p.name.lower(),
    )
    if not videos:
        parser.error(f'No videos found in: {folder}')

    for i, video in enumerate(videos, 1):
        print(f'[{i}/{len(videos)}] {video.name}', flush=True)
        # vis.py prepends ./demo/video/ to its --video argument.
        video_arg = Path(os.path.relpath(video, root / 'demo' / 'video')).as_posix()
        subprocess.run(
            [sys.executable, 'demo/vis.py', '--video', video_arg, '--gpu', args.gpu],
            cwd=root,
            check=True,
        )


if __name__ == '__main__':
    main()
