import cv2
import os

# Camera.
camera_video_path = "../input/calle_alcala.flv"
# Output folder
output_folder = "../output/calle_alcala_images/"
#output images structure
img_structures = "img_{:0>10}.png"

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

img_output_path_structure = os.path.join(output_folder,img_structures)

vidcap = cv2.VideoCapture(camera_video_path)
success, camera_image = vidcap.read()
index = 0

while success:
    print(index)
    cv2.imwrite(img_output_path_structure.format(index), camera_image)
    index+=1
    success, camera_image = vidcap.read()
