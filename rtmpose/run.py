from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d='body26',
    device='cuda:0'
)

for _ in inferencer(
    'input/1.mp4',
    out_dir='output',
    show=False
):
    pass

print("RTMPose inference completed")
