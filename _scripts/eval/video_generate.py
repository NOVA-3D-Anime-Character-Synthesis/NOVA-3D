from _util.util_v1 import * ; import _util.util_v1 as uutil
import cv2
import os


ap = uutil.argparse.ArgumentParser()
ap.add_argument('--folder_path')
args = ap.parse_args()
# 设置图片帧所在的文件夹路径
folder_path = args.folder_path
video_name = folder_path.split('/')[-1]
# 设置输出视频文件的路径和名称
output_video_path = f'{folder_path}/{video_name}.mp4'

# 获取文件夹中的所有图片帧文件
frames = [f for f in os.listdir(folder_path) if f.endswith('.png')]
frames.sort()
# 读取第一张图片获取尺寸信息
frame = cv2.imread(os.path.join(folder_path, frames[0]))
height, width, layers = frame.shape
fps = 60
# 使用VideoWriter创建视频对象
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 遍历所有图片帧并将其写入视频
for frame_name in frames:
    frame = cv2.imread(os.path.join(folder_path, frame_name))
    video.write(frame)

# 释放资源
cv2.destroyAllWindows()
video.release()

print(f'Video created successfully at {output_video_path}')
