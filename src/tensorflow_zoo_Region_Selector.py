import Region_Selector
import os
import sys
import math
import numpy as np
import tensorflow as tf
import pathlib
import cv2
import test_time_augmentation
import general_utils
from Config import Config

# Root directory of object detection

TENSORFLOW_RESEARCH_ROOT_DIR = os.path.join(Config.TENSORFLOW_ROOT_DIR, "models/research/")
print(TENSORFLOW_RESEARCH_ROOT_DIR)
sys.path.append(TENSORFLOW_RESEARCH_ROOT_DIR)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODELS_PATH = os.path.join(Config.TENSORFLOW_ROOT_DIR, "downloaded_models")

"""
In order to mazimize detection efectivity, we will split the image into smaller images with the shape the regio selector expects.

"""
def split_image(image, shape):

    return utils.split_image(image, shape)

"""
    Function to load model.
"""
def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    base_url = MODELS_PATH
    """
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"
    """

    model_dir = os.path.join(base_url, model_name, "saved_model")
    print(model_dir)
    model = tf.saved_model.load(str(model_dir))

    return model

"""
    Function to run inference for a single image.
"""

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    new_detection_boxes = []
    for detection_box in output_dict['detection_boxes']:

        y1, x1, y2, x2 = detection_box
        x1 = int(image.shape[1] * x1)
        x2 = int(image.shape[1] * x2)
        y1 = int(image.shape[0] * y1)
        y2 = int(image.shape[0] * y2)

        new_detection_boxes.append([y1, x1, y2, x2])

    output_dict['detection_boxes'] = new_detection_boxes
    return output_dict


"""
    Show inference for a video.
"""

def run_inference(model, cap):
    while cap.isOpened():
        ret, image_np = cap.read()
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

"""
    Show inference for a single image.
"""

def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #image_np = np.array(Image.open(image_path))
    image_np = cv2.imread(image_path)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    print(output_dict)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow("image", image_np)
    cv2.waitKey(0)

"""
    Function
"""

def draw_detections_on_image(image, detections, output_path):

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow(output_path, image)

"""
     Class to perform object detection using YOLO v3 with darknet weights and COCO clases.
"""

class Tensorflow_Zoo_Region_Selector(Region_Selector.Region_Selector):
    model = None     # Tensorflow zoo model for object detection.
    
    def __init__(self):
        model_name = Config.MODEL_NAME
        self.model = load_model(model_name)

        
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
        
        raise(NotImplementedError)
            
    def run_inference_for_single_image(self, img, BGR=True):
        if BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return run_inference_for_single_image(self.model, img)