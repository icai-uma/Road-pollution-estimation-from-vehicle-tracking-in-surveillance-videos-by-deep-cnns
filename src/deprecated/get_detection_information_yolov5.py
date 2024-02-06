import cv2
import os
import sys
import math
import time
import numpy as np
import general_utils

import yolov5_Region_Selector
from Config import Config
import test_time_augmentation

object_detector = yolov5_Region_Selector.Yolov5_Region_Selector()                   # Object detector wrapper.


for VIDEO_INDEX in range(Config.VIDEO_INDEX+1):
    CAMERA_VIDEO_PATH = f"../input/{Config.PREFFIX}/{Config.PREFFIX}_{VIDEO_INDEX}.avi"
    DETECTION_SAVE_FOLDER = f"../input/{Config.PREFFIX}/{Config.PREFFIX}_{VIDEO_INDEX}/{Config.MODEL_NAME}"
    print(CAMERA_VIDEO_PATH)
    print(DETECTION_SAVE_FOLDER)
    # Video.
    vidcap = cv2.VideoCapture(CAMERA_VIDEO_PATH)
    fourCC = cv2.VideoWriter_fourcc(*'XVID')

    success, camera_image = vidcap.read()

    frame_index = 0
    print("Video Shape:")
    print(camera_image.shape)
    while success:
        print(f"Processed frames : {frame_index}/?")
        #swapped_object_regions_for_frame = np.array(object_detector.get_objects_rois([camera_image], crop_position = Config.CROP_POSITION, crop_size = Config.CROP_SIZE)[0])    
                                                                                                            # We get the objects in the frame regions as a list of tuples 
                                                                                                            # with two points (upper left corner and low right corner): [((h_1,w_1),(h_2,w_2))]
        t0 = time.time()
        transformations_images, transformations = test_time_augmentation.get_transformations_from_img(camera_image, Config.TEST_TIME_TRANSFORMATIONS)   # We get the transformed images.

        for trfm_img, trfm in zip(transformations_images, transformations):
            output_dict = object_detector.get_objects_rois([trfm_img], crop_position = Config.CROP_POSITION, crop_size = Config.CROP_SIZE, split=False)[0]
            if not os.path.isdir(DETECTION_SAVE_FOLDER):
                os.makedirs(DETECTION_SAVE_FOLDER)
            if trfm:
                output_path = f"{DETECTION_SAVE_FOLDER}/{trfm}_frame_{frame_index}.json"
            else:
                output_path = f"{DETECTION_SAVE_FOLDER}/frame_{frame_index}.json"

            general_utils.save_to_json(output_dict, output_path)
        print(time.time()-t0)

        # We get the next image from the input.
        success, camera_image = vidcap.read()

        frame_index+=1
