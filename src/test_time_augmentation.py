import numpy as np
import cv2
import os
import math
import general_utils
from Config import Config

"""
Function to apply to apply test-time augmentation.
"""

# We'll guess here any ROI is defined as [min_height, min_width, max_height, max_width].

# Function to translate dictionary structure from tensorflow zoo output to this functions structure.
# Expected input estructure as a dictionary {'detection_boxes':numpy_array, 'detection_classes': [class_obj_0, class_obj_1,...,class_obj_n],'detection_scores': [score_obj_0, score_obj_1, ... , score_obj_n]}

def from_tensorflow_zoo_structure_to_sparse_structure(output_dict):
    obj_list = []
    print(output_dict)
    for detection_box, detection_class, detection_score in zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']):
        obj = {}

        if len(detection_box) == 2:                         #If the data has format [[h1,w1],[h2,w2]], we turn it to [h1,w1,h2,w2]
            p1 = detection_box[0]
            p2 = detection_box[1]
            detection_box = [p1[0], p1[1], p2[0], p2[1]]

        obj["roi"] = detection_box
        obj["score"] = detection_score
        obj["class_id"] = detection_class

        obj_list.append(obj)

    return obj_list

""" 
Function to unmade the transformation from the images data information.
    datas_dict_list : list - [img1_data, ..., imgN_data] with each img_data as a tensorflow_zoo output structure.
"""
def unmade_transformations_from_segmentation_data(datas_dict_list, transformations, img):

    for transfm, data_dict in zip(transformations, datas_dict_list):
        if transfm is None or transfm == "raw":
            pass                                # If the data has no transformation, we do nothing.
        if transfm == "flip":
            data_dict["detection_boxes"] = flip_list_of_rois(data_dict["detection_boxes"], img.shape, horizontal=True, normalized = False)
    
    return datas_dict_list

"""
Function to load images with given index with transformations from path.
"""
def load_semantic_information(path, index, transformations):
    imgs_data = []
    for transfm in transformations:
        if transfm is None or transfm == "raw":
            s = ""
        else:
            s = f"{transfm}_"

        p = os.path.join(path, f"{s}frame_{index}.json")

        img_data = general_utils.load_from_json(p)
        imgs_data.append(img_data)
    
    return imgs_data, transformations

"""
Function to join the images data information.
    imgs_data : list - [img1_data, ..., imgN_data] with each img_data as a tensorflow_zoo output structure.
"""
def join_test_time_augmentation_information_from_different_images(imgs_data):
    sparsed_imgs_data = []
    #First, we need to change the data structure.
    for img_data in imgs_data:
        sparsed_imgs_data.append(from_tensorflow_zoo_structure_to_sparse_structure(img_data))

    clustered_objects = cluster_objects_by_IOU_improved(sparsed_imgs_data, use_mask_to_compute_IOU = False)
    objs = fussion_clusters_into_objects(clustered_objects)

    rois = []
    classes = []
    scores = []
    for obj in objs:
        y1, x1, y2, x2 = obj["roi"]
        detection_class = obj["class_id"]
        detection_score = obj["score"]
                        
        rois.append(((y1,x1),(y2,x2)))
        scores.append(detection_score)
        classes.append(detection_class)

    output_dict = {"detection_boxes" : rois, "detection_classes" : classes, "detection_scores" : scores} 

    return output_dict

"""
Function to get test-augmented transformations of input image.

transformations : list containing types of image transformations.

output : (list,list) A tuple containing transformed images and transformations.
"""
def get_transformations_from_img(img, transformations):
    transformations_images = []
    for transfm in transformations:
        if transfm == "flip":
            transfm_img = flip(img)
        if transfm is None:
            transfm_img = img.copy()

        transformations_images.append(transfm_img)

    return transformations_images, transformations.copy()
    
def flip(img, horizontal = True):
    
    return cv2.flip(img, 1*(horizontal))

#Function to delete, from a list, points if they are too close.
def delete_too_close_points(points_list, threshold):
    # If we find two points with distance lesser than treshold, we delete both and add a new point with a mean position.

    new_points_list = []

    points_to_fusse = []
    for p1_index in range(len(points_list)):
        p1 = points_list[p1_index]
        for p2_index in range(len(points_list)):
            p2 = points_list[p2_index]
            if (not p1_index == p2_index) and distance_between_two_points(p1, p2) < threshold:
                points_to_fusse.append((p1,p2))


def flip_roi(roi, img_shape, horizontal = True, normalized = True):
    
    if normalized:
        img_height = 1
        img_width = 1
    else:
        img_height = img_shape[0]
        img_width = img_shape[1]
    
    if len(roi) == 2:
        roi_min_h = roi[0][0]
        roi_min_w = roi[0][1]
        roi_max_h = roi[1][0]
        roi_max_w = roi[1][1]

    if len(roi) == 4:
        roi_min_h = roi[0]
        roi_min_w = roi[1]
        roi_max_h = roi[2]
        roi_max_w = roi[3]
    
    if horizontal:
        flip_roi_min_h = roi_min_h
        flip_roi_min_w = img_width - roi_max_w
        flip_roi_max_h = roi_max_h
        flip_roi_max_w = img_width - roi_min_w
        
    else:
        flip_roi_min_h = img_height - roi_max_h
        flip_roi_min_w = roi_min_w
        flip_roi_max_h = img_height - roi_min_h
        flip_roi_max_w = roi_max_w
        
    if len(roi) == 2:
        new_roi = [[flip_roi_min_h, flip_roi_min_w], [flip_roi_max_h, flip_roi_max_w]]
    if len(roi) == 4:
        new_roi = [flip_roi_min_h, flip_roi_min_w, flip_roi_max_h, flip_roi_max_w]
    return new_roi
    
def flip_roi_from_objs_list(objs_list, img_shape, horizontal=True, normalized = True):

    new_objs_list = []
    for obj in objs_list:
        obj["roi"] = flip_roi(obj["roi"], img_shape, horizontal=horizontal, normalized=normalized)
        new_objs_list.append(obj)
        
    return new_objs_list

def flip_list_of_rois(rois, img_shape, horizontal=True, normalized = True):
    new_rois = []
    for roi in rois:
        new_roi = flip_roi(roi, img_shape, horizontal=horizontal, normalized=normalized)
        new_rois.append(new_roi)
        
    return new_rois

# Function to know if this roi is too close to the image limit.
# This information is useful if we are splitting the image before sending it to the object detection network so we can detect duplicate objects.
def is_roi_too_close_image_limit(roi, image_shape, threshold_to_be_considered_too_close):

    print("Is roi close to limit?")
    print(roi)
    print(image_shape)
    print(threshold_to_be_considered_too_close)
    min_img_h = 0
    min_img_w = 0
    max_img_h = image_shape[0]
    max_img_w = image_shape[1]

    if len(roi) == 2:
        min_h = roi[0][0]
        min_w = roi[0][1]
        max_h = roi[1][0]
        max_w = roi[1][1]
    if len(roi) == 4:
        min_h = roi[0]
        min_w = roi[1]
        max_h = roi[2]
        max_w = roi[3]

    return_value = False
    sides = []

    if abs(min_img_h-min_h) < threshold_to_be_considered_too_close:
        return_value = True
        sides.append("top")
    if abs(min_img_w-min_w) < threshold_to_be_considered_too_close:
        return_value = True
        sides.append("left")
    if abs(max_img_h-max_h) < threshold_to_be_considered_too_close:
        return_value = True
        sides.append("low")
    if abs(max_img_w-max_w) < threshold_to_be_considered_too_close:
        return_value = True
        sides.append("right")

    print(sides)
    return return_value, sides


# Function to search for two object recognitions in two adjoining images.
# each data must be a tuple with structure (absolute_position_roi, side_close_to, detection_class, detection_score) 
# We guess previous candidates are from image crops taken from top-down left-right.
def search_for_match_side_recognitions(incoming_data, previous_candidates_data, threshold_to_be_considered_too_close_to_side):

    print("search")
    print(incoming_data)
    print(previous_candidates_data)
    (absolute_position_roi, side_close_to, detection_class, detection_score) = incoming_data
    
    same_objects_from_previous_candidates_data_index = []

    # We will displace the absolute position roi (the roi it have within the complete image) acording to the sides this object is close to:
    new_absolute_position_roi = np.array(absolute_position_roi).copy()

    if "top" in side_close_to:
        new_absolute_position_roi[0] = new_absolute_position_roi[0] - 2*threshold_to_be_considered_too_close_to_side
        new_absolute_position_roi[2] = new_absolute_position_roi[2] - 2*threshold_to_be_considered_too_close_to_side

    if "low" in side_close_to:
        new_absolute_position_roi[0] = new_absolute_position_roi[0] + 2*threshold_to_be_considered_too_close_to_side
        new_absolute_position_roi[2] = new_absolute_position_roi[2] + 2*threshold_to_be_considered_too_close_to_side

    if "left" in side_close_to:
        new_absolute_position_roi[1] = new_absolute_position_roi[1] - 2*threshold_to_be_considered_too_close_to_side
        new_absolute_position_roi[3] = new_absolute_position_roi[3] - 2*threshold_to_be_considered_too_close_to_side

    if "right" in side_close_to:
        new_absolute_position_roi[1] = new_absolute_position_roi[1] + 2*threshold_to_be_considered_too_close_to_side
        new_absolute_position_roi[3] = new_absolute_position_roi[3] + 2*threshold_to_be_considered_too_close_to_side

    # Now we will check if the displaced roi is overlapping another caondidate's roi.
    
    for p_c_data_index in range(len(previous_candidates_data)):
        p_c_data = previous_candidates_data[p_c_data_index]
        (p_c_absolute_position_roi, p_c_side_close_to, p_c_detection_class, p_c_detection_score) = p_c_data
        print("-------------------------------")
        print(new_absolute_position_roi)
        print(np.array(p_c_absolute_position_roi))
        if intersection_between_two_rectangles(new_absolute_position_roi, np.array(p_c_absolute_position_roi)) > 0:
            # They are overlapping!
            if p_c_detection_class == detection_class:
                # They recognize the same class.
                # They should be the same object!

                same_objects_from_previous_candidates_data_index.append(p_c_data_index)

    print(same_objects_from_previous_candidates_data_index)
    return same_objects_from_previous_candidates_data_index

# Function to mix N objects.
def mix_adjacent_rois_objs(objs_list):

    print(objs_list)
    new_min_h = None
    new_min_w = None
    new_max_h = None
    new_max_w = None

    class_votes = {}
    class_with_max_votes = None
    max_score = None

    for obj in objs_list:
        obj_roi, _, detection_class, detection_score = obj
        obj_min_h = obj_roi[0]
        obj_min_w = obj_roi[1]
        obj_max_h = obj_roi[2]
        obj_max_w = obj_roi[3]

        if new_min_h is None or new_min_h > obj_min_h:
            new_min_h = obj_min_h

        if new_min_w is None or new_min_w > obj_min_w:
            new_min_w = obj_min_w

        if new_max_h is None or new_max_h < obj_max_h:
            new_max_h = obj_max_h

        if new_max_w is None or new_max_w < obj_max_w:
            new_max_w = obj_max_w

        if detection_class in class_votes.keys():
            class_votes[detection_class] = class_votes[detection_class]+1
        else:
            class_votes[detection_class] = 1

        if class_with_max_votes is None or class_votes[detection_class]>class_votes[class_with_max_votes]:
            class_with_max_votes = detection_class

        if max_score is None or max_score < detection_score:
            max_score = detection_score

    new_roi = [new_min_h, new_min_w, new_max_h, new_max_w]

    return (new_roi, None, class_with_max_votes, max_score)


def distance_between_two_points(p1, p2):

    c1 = math.pow(p1[0] - p2[0],2)
    c2 = math.pow(p1[1] - p2[1],2)

    return math.sqrt(c1+c2)

def distance_between_two_rois_sides(roi1, roi2):

    if len(roi1) == 2:
        roi1_min_h = roi1[0][0]
        roi1_min_w = roi1[0][1]
        roi1_max_h = roi1[1][0]
        roi1_max_w = roi1[1][1]
    if len(roi1) == 4:
        roi1_min_h = roi1[0]
        roi1_min_w = roi1[1]
        roi1_max_h = roi1[2]
        roi1_max_w = roi1[3]

    if len(roi2) == 2:
        roi2_min_h = roi2[0][0]
        roi2_min_w = roi2[0][1]
        roi2_max_h = roi2[1][0]
        roi2_max_w = roi2[1][1]
    if len(roi2) == 4:
        roi2_min_h = roi2[0]
        roi2_min_w = roi2[1]
        roi2_max_h = roi2[2]
        roi2_max_w = roi2[3]

    if intersection_between_two_rectangles([roi1_min_h, roi1_min_w, roi1_max_h, roi1_max_w],[roi2_min_h, roi2_min_w, roi2_max_h, roi2_max_w]) > 0:
        # They are overlapping!
        return -1

    minimum_value = abs(roi1_min_h - roi2_max_h)

    if abs(roi2_min_h - roi1_max_h) < minimum_value:
        minimum_value = abs(roi2_min_h - roi1_max_h)

    if abs(roi1_min_w - roi2_max_w) < minimum_value:
        minimum_value = abs(roi1_min_w - roi2_max_w)

    if abs(roi2_min_w - roi1_max_w) < minimum_value:
        minimum_value = abs(roi2_min_w - roi1_max_w)

    return minimum_value


# Function to calculate the intersection area between two rectangle.
# Squares are assumed to be given as list: [min_height, min_width, max_height, max_width]
def intersection_between_two_rectangles(sq1, sq2):
    print(sq1)
    print(sq2)
    sq1_min_h = sq1[0]
    sq1_max_h = sq1[2]
    sq1_min_w = sq1[1]
    sq1_max_w = sq1[3]
    
    sq2_min_h = sq2[0]
    sq2_max_h = sq2[2]
    sq2_min_w = sq2[1]
    sq2_max_w = sq2[3]
    
    # We will create the intersection rectangle.
    in_sq_min_h = max(sq1_min_h, sq2_min_h)
    in_sq_max_h = min(sq1_max_h, sq2_max_h)
    in_sq_min_w = max(sq1_min_w, sq2_min_w)
    in_sq_max_w = min(sq1_max_w, sq2_max_w)
    
    area = rectangle_area([in_sq_min_h, in_sq_min_w, in_sq_max_h, in_sq_max_w])
    
    if area != -1:
        return area
    else:
        return 0

# Function to calculate the intersection area between to masks.
# Masks should have the same shape and are assumed to be boolean or binary.
def intersection_between_two_masks(mask1, mask2):
    assert mask1.shape==mask2.shape

    intersection_mask = np.logical_and(mask1, mask2)*1

    return np.sum(intersection_mask)

# Function to calculate the union area between to masks.
# Masks should have the same shape and are assumed to be boolean or binary.
def union_between_two_masks(mask1, mask2):
    assert mask1.shape==mask2.shape
    
    union_mask = np.logical_or(mask1, mask2)*1

    return np.sum(union_mask)

# Function to calculate the union area of two rectangle.
# Squares are assumed to be given as list: [min_height, min_width, max_height, max_width]    
def union_of_two_rectangles(sq1, sq2):

    sq1_area = rectangle_area(sq1)
    sq2_area = rectangle_area(sq2)
    
    return sq1_area + sq2_area - intersection_between_two_rectangles(sq1, sq2)    

# Function to calculate the area of a rectangle.
# Square is assomed to be given as a list: [min_height, min_width, max_height, max_width]
def rectangle_area(sq):
    sq_min_h = sq[0]
    sq_max_h = sq[2]
    sq_min_w = sq[1]
    sq_max_w = sq[3]
    
    height = sq_max_h - sq_min_h
    
    if height < 0:
        return -1
    
    width = sq_max_w - sq_min_w
    
    if width < 0:
        return -1
    
    return height*width

# Function to calculate the intersection over union.
# Squares are assomed to be given as a list: [min_height, min_width, max_height, max_width]    
def intersection_over_union_using_squares(sq1, sq2):

    intersection = intersection_between_two_rectangles(sq1,sq2)
    union = union_of_two_rectangles(sq1,sq2)
    
    return intersection/union

def intersection_over_union_using_masks(mask1, mask2):

    intersection = intersection_between_two_masks(mask1, mask2)
    union = union_between_two_masks(mask1, mask2)

    return intersection/union
    
# Function to cluster objects from diferent semantic segmentations so each cluster is formed only by objects that should be the same (one from each segmentation).
# objs_list_from_each_image should have the structure [image1_objs_list, image2_objs_list, image3_objs_list, image4_objs_list] with length equal to the number of tiles.
# the output should have the structure [obj1_objs_list, obj2_objs_list, obj3_objs_list, obj4_objs_list, obj5_objs_list]
def cluster_objects_by_IOU(objs_list_from_each_image, use_mask_to_compute_IOU= True):
    number_of_images = len(objs_list_from_each_image)
    
    objs_clusters = []
    
    reference_image_index = 0
    
    while reference_image_index < number_of_images:
        first_not_empty_image_objs = objs_list_from_each_image[reference_image_index]
    
        for first_image_obj in first_not_empty_image_objs:
            cluster = (reference_image_index*[None])+[first_image_obj]
            first_image_obj_roi = first_image_obj["roi"]
            
            if "mask" in first_image_obj:
                first_image_obj_mask = first_image_obj["mask"]
            #print(f"From image {reference_image_index}, the object with index {first_image_obj['index']} is the reference.")
            
            for another_image_objs_list_index in range(reference_image_index+1,number_of_images):
                another_image_objs_list = objs_list_from_each_image[another_image_objs_list_index]
                image_with_maximum_IOU_index = None
                maximum_IOU = None
                
                for another_image_obj_index, another_image_obj in enumerate(another_image_objs_list):
                    if use_mask_to_compute_IOU:
                        another_image_obj_mask = another_image_obj["mask"]
                        iou = intersection_over_union_using_masks(first_image_obj_mask, another_image_obj_mask)
                    else:
                        another_image_obj_roi = another_image_obj["roi"]
                        iou = intersection_over_union_using_squares(first_image_obj_roi, another_image_obj_roi)
                    
                    if iou > Config.MINIMUM_IOU_SCORE_TO_NOT_BE_IGNORED and (maximum_IOU is None or iou > maximum_IOU):
                        maximum_IOU = iou
                        image_with_maximum_IOU_index = another_image_obj_index

                if not maximum_IOU is None:
                    another_image_obj_that_should_be_the_same_as_first_image_obj = another_image_objs_list[image_with_maximum_IOU_index]
                    cluster.append(another_image_obj_that_should_be_the_same_as_first_image_obj)
                    #print(f"The object with index {another_image_obj_that_should_be_the_same_as_first_image_obj['index']} from image {another_image_objs_list_index} has IOU {maximum_IOU}")
                    another_image_objs_list.pop(image_with_maximum_IOU_index)
                else:
                    cluster.append(None)

            objs_clusters.append(cluster)
            
        reference_image_index += 1

    return objs_clusters

# Function to cluster objects from diferent semantic segmentations so each cluster is formed only by objects that should be the same (one from each segmentation).
# objs_list_from_each_image should have the structure [image1_objs_list, image2_objs_list, image3_objs_list, image4_objs_list] with length equal to the number of tiles.
# the output should have the structure [obj1_objs_list, obj2_objs_list, obj3_objs_list, obj4_objs_list, obj5_objs_list]
def cluster_objects_by_IOU_improved(objs_list_from_each_image, use_mask_to_compute_IOU= True):
    number_of_images = len(objs_list_from_each_image)

    # List of objects clusters.
    objs_clusters = []

    # We need to get the score from each object
    # The result should be a list of numpy arrays containing the scores.
    objs_scores_np_from_each_image = [np.array([obj["score"] for obj in objs_list]) for objs_list in objs_list_from_each_image]

    # We get the image list with the most trustworthy object index.
    image_with_trustworthy_obj_index = np.argmax(np.array([np.max(ar) if ar.shape[0] > 0 else -1 for ar in objs_scores_np_from_each_image]))
    image_with_trustworthy_obj = objs_list_from_each_image[image_with_trustworthy_obj_index]

    while len(image_with_trustworthy_obj) > 0: 
        # If the image objs list is not empty, there is istill a trustworthy object.     
        trustworthy_obj_index = np.argmax(objs_scores_np_from_each_image[image_with_trustworthy_obj_index])
        trustworthy_obj = image_with_trustworthy_obj.pop(trustworthy_obj_index)

        # Empty cluster.
        cluster = []

        reference_image_index = 0

        image_with_maximum_IOU_index = None

        for image_index in range(number_of_images):

            if not image_index == image_with_trustworthy_obj_index:
                # If the most trustworthy object is not from this image...
                # We get the objects from this image.
                image_objs = objs_list_from_each_image[image_index]            
                maximum_IOU = None

                if not len(image_objs) == 0:    
                    # If still objects remains from this image...            
                    for obj_index, obj in enumerate(image_objs):
                        if use_mask_to_compute_IOU:
                            obj_mask = obj["mask"]
                            trustworthy_obj_mask = trustworthy_obj["mask"]
                            iou = intersection_over_union_using_masks(trustworthy_obj_mask, obj_mask)
                            #print("************")
                            #print(obj["index"])
                            #print(iou)

                        else:
                            obj_roi= obj["roi"]
                            trustworthy_obj_roi = trustworthy_obj["roi"]
                            iou = intersection_over_union_using_squares(trustworthy_obj_roi, obj_roi)
                        
                        if iou > Config.MINIMUM_IOU_SCORE_TO_NOT_BE_IGNORED and (maximum_IOU is None or iou > maximum_IOU):
                            maximum_IOU = iou
                            image_with_maximum_IOU_index = obj_index

                    if maximum_IOU:
                        # There is an object matching the trustworthy object.
                        obj_that_match_trustworthy_obj = image_objs.pop(image_with_maximum_IOU_index)
                        cluster.append(obj_that_match_trustworthy_obj)
                        #print("++++++++++++++")
                        #print(obj_that_match_trustworthy_obj["index"])
                        #print(maximum_IOU)
                    else:
                        # There is no object matching the trustworthy object.
                        cluster.append(None)
                else:
                    # There is no remain object in this image.
                    cluster.append(None)            
            else:
                # We insert the trustworthy object into te cluster.
                cluster.append(trustworthy_obj)
        #print("--------------------------------------")
        #print(cluster)
        # We insert the cluster.
        objs_clusters.append(cluster)
        # We need to get the score from each object
        # The result should be a list of numpy arrays containing the scores.
        objs_scores_np_from_each_image = [np.array([obj["score"] for obj in objs_list]) for objs_list in objs_list_from_each_image]

        # We get the image list with the most trustworthy object index.
        image_with_trustworthy_obj_index = np.argmax(np.array([np.max(ar) if ar.shape[0] > 0 else -1 for ar in objs_scores_np_from_each_image]))
        image_with_trustworthy_obj = objs_list_from_each_image[image_with_trustworthy_obj_index]
    return objs_clusters

# The input should have the structure [cluster_obj_1, cluster_obj_2, ..., cluster_obj_N]
# The output should be an object or None.
def fussion_cluster_into_one_object(cluster, use_mask = False):
    none_counter = 0
    cluster_size = len(cluster)
    class_id_votation = {}
    masks_matrix = None
    scores = []
    rois = []

    for obj in cluster:
        if obj is None:
            none_counter += 1
        else:
            if use_mask: obj_mask = obj["mask"]
            obj_score = obj["score"]
            obj_class_id = obj["class_id"]
            roi = obj["roi"]
            
            if obj_class_id in class_id_votation:
                class_id_votation[obj_class_id] = class_id_votation[obj_class_id] + 1
            else:
                class_id_votation[obj_class_id] = 1
            
            scores.append(obj_score)
            rois.append(roi)
            
            if use_mask: 
                if masks_matrix is None:
                    masks_matrix = np.expand_dims(obj_mask, axis=-1)
                else:
                    masks_matrix = np.concatenate([masks_matrix, np.expand_dims(obj_mask, axis=-1)], axis=-1)
    
    scores_np = np.array(scores)
    max_score = np.max(scores_np)
    max_score_obj_index = np.where(scores_np == max_score)[0][0]
    
    max_score_roi = rois[max_score_obj_index]
    
    if (Config.MINIMUM_TILES_SHARE_RECOGNIZING_THE_OBJECT_TO_ALLOW) < (1-none_counter/cluster_size):
    
        if use_mask: mask = (np.mean(masks_matrix, axis=-1) > Config.MINIMUM_TILES_SHARE_RECOGNIZING_THE_OBJECT_TO_ALLOW) * 1
        
        if (not use_mask) or (use_mask and (np.max(mask) > 0)):
            score = np.mean(np.array(obj_score))
            class_id = max(class_id_votation, key = lambda k: class_id_votation[k])
            
            roi_max_h = None
            roi_min_h = None
            roi_max_w = None
            roi_min_w = None

            if use_mask:
                for row_index, row in enumerate(mask):
                    row_where = np.where(row==1)[0]
                                
                    if row_where.shape[0] > 0:
                        min_index_in_row = min(row_where)
                        max_index_in_row = max(row_where)
                        roi_max_h = row_index
                                      
                        if roi_min_h is None:
                            roi_min_h = row_index
                            
                        if roi_min_w is None or min_index_in_row < roi_min_w:
                            roi_min_w = min_index_in_row
                            
                        if roi_max_w is None or max_index_in_row > roi_max_w:
                            roi_max_w = max_index_in_row

            obj = {}
            if use_mask:
                obj["roi"] = np.array([roi_min_h, roi_min_w, roi_max_h, roi_max_w]).astype('int64')
                obj["mask"] = mask
            else:
                obj["roi"] = max_score_roi

            obj["score"] = float(score)
            obj["class_id"] = int(class_id)
            
            return obj
            
        else:
            return None        
    else:
        return None
        
# The input should have the structure [cluster_1_objs_list, cluster_2_objs_list, ..., cluster_N_objs_list]
# The output should have the structure [obj_1, obj_2, ..., obj_M] with M=<N
def fussion_clusters_into_objects(cluster_list):
    objs = []

    for cluster_index, cluster in enumerate(cluster_list):
        #print("__________________________________________________________")
        #print(cluster)
        obj = fussion_cluster_into_one_object(cluster)
        if not obj is None:
            obj["index"] = cluster_index
            objs.append(obj)
        #print(obj)

        #print("========================================================")
        #print(obj)
            
    return objs
