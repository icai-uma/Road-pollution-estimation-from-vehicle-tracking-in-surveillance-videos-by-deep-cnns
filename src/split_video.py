import cv2
import os
from Config import Config

size_of_output_videos = 3600 # In frames

# Video.
vidcap = cv2.VideoCapture(Config.CAMERA_VIDEO_PATH)
fourCC = cv2.VideoWriter_fourcc(*'XVID')

success, vid_image = vidcap.read()
frame_index = 0
video_index = 0

if not os.path.isdir(f"../output/{Config.PREFFIX}"):
    os.makedirs(f"../output/{Config.PREFFIX}")

new_video = None
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
while success:
    if new_video is None:
        new_video = cv2.VideoWriter(f"../output/{Config.PREFFIX}/{Config.PREFFIX}_{video_index}.avi", fourCC, int(fps), (vid_image.shape[1], vid_image.shape[0]))

    print(f"Processed frames : {frame_index}/?")
    new_video.write(vid_image)

    frame_index += 1

    if frame_index % size_of_output_videos == 0:
        new_video.release()
        new_video = None
        video_index +=1    
    
    success, vid_image = vidcap.read()

new_video.release()
    
