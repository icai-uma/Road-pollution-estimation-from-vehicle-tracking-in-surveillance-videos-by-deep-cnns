import os

class Config():
    """
    Input and output information.
    """
    MAIN_INPUT_FOLDER = "/media/jrggcgz/Maxtor/Files/Work/road_pollution_stimation/input"
    MAIN_OUTPUT_FOLDER = "/media/jrggcgz/Maxtor/Files/Work/road_pollution_stimation/output"
    
    
    #Video name    
    VIDEO_NAME = "sherbrooke_video"
    REAL_REFERENCE_LENGTH = 10                                                   # Reference length in meters.
    SEQUENCE_FRAMES_PER_SECOND = 30                                         # Input video configuration.
    """
    
    VIDEO_NAME = "stmarc_video"
    REAL_REFERENCE_LENGTH = 5                                               # Reference length in meters.
    SEQUENCE_FRAMES_PER_SECOND = 30                                         # Input video configuration.

    
    VIDEO_NAME = "rouen_video"
    VIDEO_INDEX = 0
    # Reference image.
    REFERENCE_IMAGE_PATH = f"{INPUT_FOLDER}/{VIDEO_NAME}/rouen_reference_image.png"
    REAL_REFERENCE_LENGTH = 5                                                   # Reference length in meters.
    SEQUENCE_FRAMES_PER_SECOND = 25                                         # Input video configuration.

    """

    # Reference image.
    REFERENCE_IMAGE_PATH = f"{MAIN_INPUT_FOLDER}/{VIDEO_NAME}/reference_image.png"
    # Camera.
    CAMERA_VIDEO_PATH = f"{MAIN_INPUT_FOLDER}/{VIDEO_NAME}/{VIDEO_NAME}.avi"
    #CAMERA_VIDEO_PATH = f"{INPUT_FOLDER}/{VIDEO_NAME}.avi"

    """
    Tracker and track configuration data.
    """
    TRACKER_DIST_TRESHOLD = 30
    TRACKER_MAX_FRAMES_TO_SKIP = 4
    TRACKER_MAX_TRACK_LENGTH = 61
    TRACKER_VALUE_TO_USE_AS_INF = 1000000
    TRACKER_DEBUG = True

    """
    Speed stimation information.
    """
    SPEED_ESTIMATION = "optical_flow"                  # "polynomial_average", "optical_flow" or "simple_average".

    """
    Region detector configuration.
    """

    OBJECT_DETECTION_REPOSITORY_PATH = "../../object_detection/src" # Path to https://github.com/SirSykon/object_detection/tree/main/src
    MODEL_NAME = "gt_annotated"   # "gt_annotated", "faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8" or "yolov5"
    #MODEL_NAME = "gt_annotated"
    BACKEND = "none"
    #BACKEND = "none"

    DETECTION_SCORE_THRESHOLD = 0.5
    CLASS_IDS_TO_INCLUDE = [2,3,5,7]        # Others
    DETECTION_MODE = "LOAD"                                                 # Configuration value. "SAVE" will save detections from a video in order to load it. "LOAD" will load the detections previously saved.
    DETECTION_SAVE_FOLDER = f"{MAIN_OUTPUT_FOLDER}/{VIDEO_NAME}/{BACKEND}/{MODEL_NAME}/detections"              # Folder to save or load the detection information.

    """
    General configuration data.
    """
    CORRECT_IMAGE = True                                                                                        # Do we apply homography?
    TEST = False                                                                                                # Do we show test information?
    NUMBER_OF_FRAMES_TO_COMPUTE = 5000                                                                          # Maximum number of frames to compute.
    NUMBER_OF_FRAMES_TO_AVERAGE = 1                                                                             # We will use the information from the last frames to calculate average position to estabilize position.
    SELECT_NEW_COORDS = False                                                                                   # Do we select new coords as reference or we load from files?
    MINIMUM_SPEED_TRESHOLD = 1                                                                                  # Minimum speed treshold to discard fake movement.
    TIME_LAPSE_TO_GET_SPEED = 0.5                                                                                 # Seconds to calculate speed.
    NUMBER_OF_REFERENCE_COORDS = 8                                                                              # Number of coordinates used.
    METERS_BY_PIXEL = -1                                                                                        # Equivalence meters-pixel. This will be computed later.
    APPLY_GAUSSIAN_FILTER_TO_SPEED = False                                                                      # Do we apply a gaussian filter to the speed records?
    POSTPROCESS_FILTER_TO_SPEED = True                                                                          # Do we postprocess speed records to get average speed for each position?
    GAUSSIAN_FILTER_SIGMA = 1                                                                                   # The sigma used when applying gaussian filter.
    SAVE_CORRECTED_IMAGES = True                                                                                # Do we save corrected images?
    SAVE_FRAMES = True                                                                                          # Do we save the original images?   
    INPUT_FOLDER = f"{MAIN_INPUT_FOLDER}/{VIDEO_NAME}"                                                          # Video information input folder.
    OUTPUT_VIDEO_FOLDER = f"{MAIN_OUTPUT_FOLDER}/{VIDEO_NAME}"                                                  # Video information output folder.
    OUTPUT_MODEL_FOLDER = f"{OUTPUT_VIDEO_FOLDER}/{BACKEND}/{MODEL_NAME}"                                       # Model information output folder for the current video.
    CORRECTED_IMAGES_WITH_OBJECTS_FOLDER = f"{OUTPUT_MODEL_FOLDER}/corrected/"                                  # Folder to save corrected images with objects.
    ORIGINAL_IMAGES_WITH_OBJECTS_FOLDER = f"{OUTPUT_MODEL_FOLDER}/original/"                                    # Folder to save original images with objects.
    CORRECTED_IMAGES_FOLDER = f"{OUTPUT_VIDEO_FOLDER}/corrected/"                                               # Folder to save corrected images.
    ORIGINAL_IMAGES_FOLDER = f"{OUTPUT_VIDEO_FOLDER}/original/"                                                 # Folder to save original images.
    OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER = f"{OUTPUT_VIDEO_FOLDER}/corrected_optical_flow"                      # Folder to save corrected images optical flow.
    OPTICAL_FLOW_ORIGINAL_IMAGES_FOLDER = f"{OUTPUT_VIDEO_FOLDER}/original_optical_flow"                        # Folder to save original images optical flow.
    OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER_WITH_OBJECTS_FOLDER = f"{OUTPUT_MODEL_FOLDER}/corrected_optical_flow"  # Folder to save corrected image optical flow with objects.
    SPEED_ESTIMATION_FOLDER = f"{OUTPUT_MODEL_FOLDER}/{SPEED_ESTIMATION}"                                       # Folder to save speed estimation.           
    SAVE_OPTICAL_FLOW_IMAGES = True                                                                             # Do we save optical flow images?
    TRACK_IMAGES_FOLDER = f"{OUTPUT_MODEL_FOLDER}/TRACK"