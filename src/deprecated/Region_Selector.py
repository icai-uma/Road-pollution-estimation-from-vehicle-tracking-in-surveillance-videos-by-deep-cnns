import test_time_augmentation
from general_utils import split_image
from Config import Config

"""
Class to perform the Region selector selection over each frame.
"""

class Region_Selector:

    """
    Method to return a list of regions for each frame.
        frames : list
            List of images to get regions from.
        ---
        returns : list
            A list with  a list of regions for each frame.
    """
    def get_regions(self, frames):
        raise(NotImplementedError)
        
    def get_objects_rois(frames, filter_by_class_id = [3,4,6,8], split=True, crop_position = None, crop_size = None):
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
    def get_objects_rois(self, frames, split=True, crop_position = None, crop_size = None):
        
        output_dicts=[]
        for i, img in enumerate(frames):
            
            if not crop_size is None and not crop_position is None:
                crop_p_h = crop_position[0]
                crop_p_w = crop_position[1]
                crop_s_h = crop_size[0]
                crop_s_w = crop_size[1]
                img = img[crop_p_h:(crop_p_h+crop_s_h), crop_p_w:(crop_p_w+crop_s_w)]
            
            else:
                crop_p_h = 0
                crop_p_w = 0
        
            rois = []
            scores= []
            classes = []
            #print("Processing image {} / {}".format(i+1,len(frames)))

            if split:

                splitted_img_list, split_positions = split_image(img, Config.REGION_SELECTOR_INPUT_SIZE)

                split_recognition_candidates = []       # List of recognitons near the borders so the same object can be recognized in various sections.

                for split_img, split_position in zip(splitted_img_list,split_positions):

                    ((y1_offset,x1_offset),(y2_offset,x2_offset)) = split_position

                    output_dict = self.run_inference_for_single_image(split_img)

                    for detection_box, detection_class, detection_score in zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']):
                        y1, x1, y2, x2 = detection_box
                        object_absolute_position_roi = [y1+crop_p_h+y1_offset,x1+crop_p_w+x1_offset, y2+crop_p_h+y1_offset,x2+crop_p_w+x1_offset]

                        close_to_side, side_close_to = test_time_augmentation.is_roi_too_close_image_limit(detection_box, Config.REGION_SELECTOR_INPUT_SIZE, Config.TRESHOLD_TO_BE_CONSIDERED_TOO_CLOSE_TO_SIDE)
                        
                        if close_to_side:
                            
                            object_data = (object_absolute_position_roi, side_close_to, detection_class, detection_score)
                            same_objects_from_previous_candidates_data_index = test_time_augmentation.search_for_match_side_recognitions(object_data, split_recognition_candidates, Config.TRESHOLD_TO_BE_CONSIDERED_TOO_CLOSE_TO_SIDE)

                            if len(same_objects_from_previous_candidates_data_index) > 0:
                                # The is at least 1 match!
                                # We will insert the match objects in this list:
                                same_objects_from_previous_candidates_data = []

                                for index in sorted(same_objects_from_previous_candidates_data_index, reverse = True):
                                    same_objects_from_previous_candidates_data.append(split_recognition_candidates.pop(index))

                                # We insert the incoming object.
                                same_objects_from_previous_candidates_data.append(object_data)
                                # We mix the object
                                mixed_object = test_time_augmentation.mix_adjacent_rois_objs(same_objects_from_previous_candidates_data)

                                print(mixed_object)
                                # We insert the new object since it could be at a corner and be further mixed with other incoming objects.
                                split_recognition_candidates.append(mixed_object)
                            else:
                                split_recognition_candidates.append(object_data)

                        if not close_to_side:
                            
                            rois.append(((object_absolute_position_roi[0],object_absolute_position_roi[1]),(object_absolute_position_roi[2],object_absolute_position_roi[3])))
                            scores.append(float(detection_score))
                            classes.append(int(detection_class))

                # Once we have obtained all objects, we will inser remaining objects in split_recognition_candidates
                for obj_data in split_recognition_candidates:
                    (object_absolute_position_roi, side_close_to, detection_class, detection_score) = obj_data

                    rois.append(((object_absolute_position_roi[0],object_absolute_position_roi[1]),(object_absolute_position_roi[2],object_absolute_position_roi[3])))
                    scores.append(float(detection_score))
                    classes.append(int(detection_class))

            else:
                output_dict = self.run_inference_for_single_image(img)

                for detection_box, detection_class, detection_score in zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']):
                    y1, x1, y2, x2 = detection_box
                    object_absolute_position_roi = [y1+crop_p_h,x1+crop_p_w, y2+crop_p_h,x2+crop_p_w]
                            
                    rois.append(((object_absolute_position_roi[0],object_absolute_position_roi[1]),(object_absolute_position_roi[2],object_absolute_position_roi[3])))
                    scores.append(float(detection_score))
                    classes.append(int(detection_class))         

            output_dict = {"detection_boxes" : rois, "detection_classes" : classes, "detection_scores" : scores} 
            
            output_dicts.append(output_dict)          

        return output_dicts
        
    """
    Method to return a list of rois for each frame.
        frames : list
            List of images to get rois from.
        ---
        returns : list
            A list with a tuple of regions for each frame with the shape ((h_1,w_1),(h_2,w_2)).
            The drawned image.
    """
    def get_objects_rois_with_time_augmentation(self, frames, split=True, crop_position = None, crop_size = None):

        output_dicts=[]
        for i, img in enumerate(frames):
            
            if not crop_size is None and not crop_position is None:
                crop_p_h = crop_position[0]
                crop_p_w = crop_position[1]
                crop_s_h = crop_size[0]
                crop_s_w = crop_size[1]
                img = img[crop_p_h:(crop_p_h+crop_s_h), crop_p_w:(crop_p_w+crop_s_w)]

            else:
                crop_p_h = 0
                crop_p_w = 0

            #print("Processing image {} / {}".format(i+1,len(frames)))

            if split:

                splitted_img_list, split_positions = split_image(img, Config.REGION_SELECTOR_INPUT_SIZE)

                rois = []
                scores= []
                classes = []

                for split_img, split_position in zip(splitted_img_list,split_positions):
                    ((y1_offset,x1_offset),(y2_offset,x2_offset)) = split_position
                    output_dict = self.run_inference_for_single_image(split_img)
                    flipped_output_dict = self.run_inference_for_single_image(test_time_augmentation.flip(split_img))
                    sparsed_structure = test_time_augmentation.from_tensorflow_zoo_structure_to_sparse_structure(output_dict)
                    sparsed_flipped_structure = test_time_augmentation.from_tensorflow_zoo_structure_to_sparse_structure(flipped_output_dict)
                    unflipped_sparsed_flipped_structure = test_time_augmentation.flip_roi_from_objs_list(sparsed_flipped_structure, split_img.shape, normalized=False)
                    clustered_objects = test_time_augmentation.cluster_objects_by_IOU_improved([sparsed_structure, unflipped_sparsed_flipped_structure], use_mask_to_compute_IOU = False)
                    objs = test_time_augmentation.fussion_clusters_into_objects(clustered_objects)

                    for obj in objs:
                        y1, x1, y2, x2 = obj["roi"]
                        detection_class = obj["class_id"]
                        detection_score = obj["score"]
                        
                        rois.append(((y1+crop_p_h+y1_offset,x1+crop_p_w+x1_offset),(y2+crop_p_h+y1_offset,x2+crop_p_w+x1_offset)))
                        scores.append(detection_score)
                        classes.append(detection_class)

            else:
                raise(NotImplementedError)

            output_dict = {"detection_boxes" : rois, "detection_classes" : classes, "detection_scores" : scores} 
            
            output_dicts.append(output_dict)                        

        return output_dicts
