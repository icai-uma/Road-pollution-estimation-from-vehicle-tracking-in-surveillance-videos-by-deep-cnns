import Region_Selector
import os
import sys
import math
import numpy as np
import tensorflow as tf

# Root directory of yolov3-keras-tf2 project
ROOT_DIR = os.path.abspath("../../yolov3-keras-tf2/")
sys.path.append(ROOT_DIR)  # To find local version of the library

import Main.detector as yolov3_detector

region_selector_input_size = (416, 416, 3)      # We will split the images into images with this shape to work better with smaller objects.

"""
In order to mazimize detection efectivity, we will split the image into smaller images with the shape the regio selector expects.

"""
def split_image(image, shape):

    split_list = []
    split_position = []

    img_shape = image.shape

    # print(img_shape)
    # print(shape)

    images_per_height = math.ceil(img_shape[0]/shape[0])
    images_per_width = math.ceil(img_shape[1]/shape[1])

    # print(images_per_height)
    # print(images_per_width)

    # So we will return images_per_height*images_per_width images from left to right from high to low.

    for img_h in range(images_per_height):
        for img_w in range(images_per_width):
            low_h_index = img_h*shape[0]
            high_h_index = (img_h+1)*shape[0]
            low_w_index = img_w*shape[1]
            high_w_index = (img_w+1)*shape[1]

            high_w_index = min(high_w_index, img_shape[1])
            high_h_index = min(high_h_index, img_shape[0])

            img_split = image[low_h_index:high_h_index, low_w_index:high_w_index, :]

            if img_split.shape[0] < shape[0] or img_split.shape[1] < shape[1]:
                temp_img_split = np.zeros(shape=shape)
                temp_img_split[:,:,:] = 128
                temp_img_split[:img_split.shape[0], :img_split.shape[1], :] = img_split
                img_split = temp_img_split
            print(img_split.shape)
            print(img_split)
            # print([(low_h_index, low_w_index),(high_h_index, high_w_index)])
            split_list.append(img_split)
            split_position.append([(low_h_index, low_w_index),(high_h_index, high_w_index)])

    return split_list, split_position

"""
     Class to perform object detection using YOLO v3 with darknet weights and COCO clases.
"""

class Yolov3_Region_Selector(Region_Selector.Region_Selector):
    detector = None     # Yolo version 3 detector.
    
    def __init__(self):
        class_names_file = os.path.abspath(os.path.join(ROOT_DIR,"Models/ms_coco_classnames.txt"))
        print(class_names_file)
        self.detector = yolov3_detector.Detector(
            (416, 416, 3),
            model_configuration= os.path.join(ROOT_DIR, "Config/yolo3.cfg"),
            classes_file=class_names_file,
            score_threshold=0.5,
            iou_threshold=0.5,
            max_boxes=100
        )
        
        self.detector.initialize_network(trained_weights=os.path.join(ROOT_DIR,"Models/yolov3.weights"))
        
    """
    Method to return a list of regions for each frame.
        frames : list
            List of images to get regions from.
        get_regions_from_central_frame : boolean
            The central image is segmented and are all images cropped from that prediction?
        output_path : str
            Path to save segmentation results. If None, the path defined during __init__() will be used.
        initial_index_to_save : int
            initial index to use when saving the segmentated images.
        ---
        returns : list
            A list with a list of cropped images for each obtained roi (one cropped image for each roi). The shape would be: 
            [[frame_1_crop_1,frame_2_crop_1,frame_3_crop_1],[frame_1_crop_2,frame_2_crop_2,frame_3_crop_2]]
        returns :
            A list with the position of each cropped image within the whole image.
    """
    def get_regions(self, frames, get_regions_from_central_frame = True, output_path = None, initial_index_to_save = 0):
        
        if not output_path is None:
            self.output_path = output_path
            
        total_numer_of_frames_frame_objects = len(frames)
        #print(total_numer_of_frames_frame_objects)
        if get_regions_from_central_frame and total_numer_of_frames_frame_objects % 2 == 1:
            central_frame_index = total_numer_of_frames_frame_objects//2
            central_frame = frames[central_frame_index]
            #print(central_frame)
            #print(central_frame.shape)

            central_frame_rois, images_with_classification_drawn = self.get_objects_rois([central_frame])                   # We get the central frame roi by using Mask RCNN.
            central_frame_rois = central_frame_rois[0]
            if not images_with_classification_drawn is None:
                images_with_classification_drawn = images_with_classification_drawn[0]
            
            regions = []
            for ((h_1,h_2),(w_1,w_2)) in central_frame_rois:                        # For each object detected, 
                frame_regions = []                                                  # we use the index to crop the same position in all the frames.
                for frame in frames:                            
                    frame_regions.append(frame[h_1:h_2,w_1:w_2])
                    #print(frame.shape)
                    #print(frame[h_1:h_2,w_1:w_2])
                regions.append(frame_regions)

            #print(len(regions))
            #print(len(regions[0]))
            #print(regions[0][0])
            #print(central_frame_rois)
            return regions, central_frame_rois
        else:
            raise(NotImplementedError)
            
    """
    Method to return a list of rois for each frame.
        frames : list
            List of images to get rois from.
        ---
        returns : list
            A list with a tuple of regions for each frame with the shape ((h_1,w_1),(h_2,w_2)).
            The drawned image.
    """
    @tf.function
    def get_objects_rois(self, frames, return_detections_on_image = True, filter_by_names = ["car", "bus"], split=True):

        regions_for_frame = []
        if return_detections_on_image:
            images_with_drawn_detections = []

        else:
            images_with_drawn_detections = None

        for i, img in enumerate(frames):
            #print("Processing image {} / {}".format(i+1,len(frames)))

            if split:

                splitted_img_list, split_positions = split_image(img, region_selector_input_size)

                rois = []

                for split_img, split_position in zip(splitted_img_list,split_positions):
                    detections, image_with_drawn_detections = self.detector.predict_on_image_wrapper(split_img,                     
                        draw_detections_on_image=return_detections_on_image)

                    ((y1_offset,x1_offset),(y2_offset,x2_offset)) = split_position

                    for index, row in detections.iterrows():
                        img, obj, x1, y1, x2, y2, score, *_ = row.values
                        #print(f"obj: {obj}, {type(obj)}")
                        if filter_by_names is None or (True in [name in obj for name in filter_by_names]):
                            #print("Included")
                            rois.append(((y1+y1_offset,x1+x1_offset),(y2+y1_offset,x2+x1_offset)))

            else:
                detections, image_with_drawn_detections = self.detector.predict_on_image_wrapper(img,
                        draw_detections_on_image=return_detections_on_image)

                if return_detections_on_image:
                    images_with_drawn_detections.append(image_with_drawn_detections)

                rois = []
                for index, row in detections.iterrows():
                    img, obj, x1, y1, x2, y2, score, *_ = row.values
                    #print(f"obj: {obj}, {type(obj)}")
                    if filter_by_names is None or (True in [name in obj for name in filter_by_names]):
                        #print("Included")
                        rois.append(((y1,x1),(y2,x2)))

            regions_for_frame.append(rois)

        return regions_for_frame, images_with_drawn_detections
    
