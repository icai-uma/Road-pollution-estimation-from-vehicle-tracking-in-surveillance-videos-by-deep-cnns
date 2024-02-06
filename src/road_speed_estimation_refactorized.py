import cv2
import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import sleep

from general_utils import postprocess_speed_record, load_detection_and_turn_to_coco_format, fill_lost_track_points, generate_speed_based_on_optical_flow, generate_speed_based_on_simple_average, generate_speed_based_on_time_average, assign_regions_to_track_id, get_homography_calibration_points, colors_generator, get_homography_matrix, use_homography_to_warp_image_perspective, use_homography_to_warp_coords_perspective, draw_rectangle_on_image, swap, get_center, print_text_on_image, draw_track_on_image, apply_gaussian_filter, is_inside_polygon, draw_polygon_on_image
from tracker import Tracker
from Config import Config

sys.path.insert(0, Config.OBJECT_DETECTION_REPOSITORY_PATH)
print (sys.path)

if Config.DETECTION_MODE == "INFERENCE":
    if Config.BACKEND == "torch":
        from object_detectors.torch.faster_rcnn_object_detector import Faster_RCNN_Object_Detector
        from object_detectors.torch.yolo_object_detector import YOLO_Object_Detector
        from object_detectors.torch.ssd_object_detector import SSD_Object_Detector

    if 'faster' in Config.MODEL_NAME:
        print("Loading Faster RCNN torch object detector")
        object_detector = Faster_RCNN_Object_Detector()
    if 'ssd' in Config.MODEL_NAME:
        print("Loading SSD torch object detector")
        object_detector = SSD_Object_Detector()
    if 'yolo' in Config.MODEL_NAME:
        print("Loading YOLO torch object detector")
        object_detector = YOLO_Object_Detector()
matplotlib.use('TkAgg')
NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED = int(Config.SEQUENCE_FRAMES_PER_SECOND*Config.TIME_LAPSE_TO_GET_SPEED)    # Number of frames we will use to calculate speed.

print(f"We will use {Config.TIME_LAPSE_TO_GET_SPEED} seconds of video to calculate speed. This sequence has {Config.SEQUENCE_FRAMES_PER_SECOND} FPS so we will use the dfference between {NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED} frames.")

#the [x, y] for each right-click event will be stored here
coords = []
space_reference_coords = []
space_reference = None
video_reference_coords = []

# Variables to be used by interactive windows.
reference_fig = None
reference_cid = None
video_fig = None
video_cid = None


# Object to generate colors to be used later.
color_generator = colors_generator(25,230)

# Image used as reference.
reference_image = cv2.imread(Config.REFERENCE_IMAGE_PATH)

# Video.
vidcap = cv2.VideoCapture(Config.CAMERA_VIDEO_PATH)
fourCC = cv2.VideoWriter_fourcc(*'XVID')

success, camera_image = vidcap.read()

print(f"Processing {Config.CAMERA_VIDEO_PATH}")
# Tracker
tracker = Tracker(Config.TRACKER_DIST_TRESHOLD, Config.TRACKER_MAX_FRAMES_TO_SKIP, Config.TRACKER_MAX_TRACK_LENGTH)

"""
It's very improtant to note that all coords used as reference are obtained from an image shown with matplotlib and are in standard mathematical format (x,y) with x the position over horizontal axis and y the position over vertical axis.
When we work with an image, the computational standard is (h,w) with h as position over vertical axis and w as position over horizontal axis so the coords will be manipulated.
We will use term "swap" to change from one standard to another and "swapped" to note information using computational standard.
"""

corrected_polygon = []
polygon = []
all_corrected_images_paths = []
all_images_paths = []
all_tracks_speed_record = {}
all_tracks_positions = {}
all_tracks_regions = {}
all_tracks_frame_record = {}
all_tracks_color_record = {}
video_corrected = None
video_camera = None
video_union = None
union_shape = None
H = None
frame_index = 0

while success and frame_index<Config.NUMBER_OF_FRAMES_TO_COMPUTE:
    # While there are incomming frames and we have not reached the desired number of computed frames...

    print(f"Processed frames : {frame_index}/?")

    """
    FIRST FRAME INITIALIZATION
    """

    if frame_index == 0:                   # This is the frist frame so we will initialize data.

        space_reference_coords, video_reference_coords, space_reference = get_homography_calibration_points(reference_image, camera_image, Config.SELECT_NEW_COORDS, Config.REAL_REFERENCE_LENGTH, Config.INPUT_FOLDER, Config.OUTPUT_VIDEO_FOLDER)      # We get the reference information as mathematical standard.
        METERS_BY_PIXEL = Config.REAL_REFERENCE_LENGTH/space_reference                  # We obtain the meters value by pixel.
        H = get_homography_matrix(space_reference_coords, video_reference_coords)       # We obtain the homography matrix using coords with mathematical standard.

        if Config.TEST:
            print("H")
            print(H)

        if not os.path.isdir(Config.ORIGINAL_IMAGES_FOLDER):
            os.makedirs(Config.ORIGINAL_IMAGES_FOLDER)

        if not os.path.isdir(Config.ORIGINAL_IMAGES_WITH_OBJECTS_FOLDER):
            os.makedirs(Config.ORIGINAL_IMAGES_WITH_OBJECTS_FOLDER)

        if not os.path.isdir(Config.CORRECTED_IMAGES_FOLDER):
            os.makedirs(Config.CORRECTED_IMAGES_FOLDER)

        if not os.path.isdir(Config.CORRECTED_IMAGES_WITH_OBJECTS_FOLDER):
            os.makedirs(Config.CORRECTED_IMAGES_WITH_OBJECTS_FOLDER)

        if not os.path.isdir(Config.OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER):
            os.makedirs(Config.OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER)

        if not os.path.isdir(Config.OPTICAL_FLOW_ORIGINAL_IMAGES_FOLDER):
            os.makedirs(Config.OPTICAL_FLOW_ORIGINAL_IMAGES_FOLDER)

        if not os.path.isdir(Config.OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER_WITH_OBJECTS_FOLDER):
            os.makedirs(Config.OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER_WITH_OBJECTS_FOLDER)

        if not os.path.isdir(Config.TRACK_IMAGES_FOLDER):
            os.makedirs(Config.TRACK_IMAGES_FOLDER)

        corrected_image = use_homography_to_warp_image_perspective(H, camera_image, reference_image)    # We calculate the image with the correct perspective.
        corrected_coords = use_homography_to_warp_coords_perspective(H, video_reference_coords)         # We calculate the coords with the correct perspective.

        corrected_polygon = corrected_coords                                                                      # The corrected coords will be the polygin inside we will calculate speeds.
        
        plt.figure(4)                                                                                   # We show and save the image
        imgplot = plt.imshow(corrected_image[:,:,[2,1,0]])
        plt.scatter(corrected_coords[:,0], corrected_coords[:,1], c="r", s=200)
        plt.savefig(f"{Config.OUTPUT_VIDEO_FOLDER}/homography_image.png")        
        plt.show()
        cv2.imwrite(f"{Config.OUTPUT_VIDEO_FOLDER}/homography_image2.png", corrected_image)

    """
    DETECT OBJECTS
    """
    if Config.DETECTION_MODE == "LOAD":
        offset_to_classes = 1 if Config.BACKEND == 'tensorflow' and 'faster' in Config.MODEL_NAME else 0
        output_dict = load_detection_and_turn_to_coco_format(Config.DETECTION_SAVE_FOLDER, frame_index, offset_to_classes=offset_to_classes)

    else:
        images_batch = [cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)]
        preprocessed_images = object_detector.preprocess(images_batch)                                      # We preprocess the batch.
        outputs = object_detector.process(preprocessed_images)                                              # We apply the model.
        outputs = object_detector.filter_output_by_confidence_treshold(outputs, treshold = Config.DETECTION_SCORE_THRESHOLD)             # We filter output using confidence.
        output_dict = outputs[0]

    coco_format_object_regions_for_frame, classes_objects_for_frame, scores_objects_for_frame = output_dict     # with one points (upper left corner, width and height): [((w_1,h_1),(width,height))]

    """
    PROCESS DETECTED OBJECTS AND CORRECT POSITIONs USING HOMOGRAPHY H.
    """
    corrected_image = use_homography_to_warp_image_perspective(H, camera_image, reference_image)            # We warp the image the image with the correct perspective.

    object_centers = []                                                                                     # Objects Centers list
    corrected_object_centers_in_polygon = []                                                                   # corrected Objects Centers list
    corrected_object_region_corners_in_polygon = []


    original_image_path = os.path.join(Config.ORIGINAL_IMAGES_FOLDER, f"frame_{frame_index}.png")
    cv2.imwrite(original_image_path, camera_image)
    all_images_paths.append(original_image_path)

    corected_image_path = os.path.join(Config.CORRECTED_IMAGES_FOLDER, f"frame_{frame_index}.png")
    cv2.imwrite(corected_image_path, corrected_image)
    all_corrected_images_paths.append(corected_image_path)


    if Config.TEST:
        print("Objects regions:")
        print(object_regions_for_frame[0])

    for coco_format_object_region_vector, class_id, score in zip(coco_format_object_regions_for_frame, classes_objects_for_frame, scores_objects_for_frame):
        # We get a vector for each region. The vector will have structure [x,y,width,height]
        # so by computational standard the corners are left-right top-bottom (y,x), (y,x+width), (y+height,x), (y+height,x+width)
        # and the region is defined using mathematical standard by points (x,y), (x+width,y), (x,y+height) and (x+width,y+height).
        # We get also class id and score.

        #swapped_object_region_tuple = ((swapped_object_region_vector[0],swapped_object_region_vector[1]), (swapped_object_region_vector[2],swapped_object_region_vector[3]))
        [x,y,width,height] = coco_format_object_region_vector
        swapped_object_region_vector = [y,x, y+height,x+width]
        swapped_object_region = np.reshape(swapped_object_region_vector, [2,2]) # We turn the vector into a 2D matrix [[h_1,w_1],[h_2,w_2]]

        if ((Config.CLASS_IDS_TO_INCLUDE is None) or (class_id in Config.CLASS_IDS_TO_INCLUDE)): 

            #swapped_object_region = np.array(swapped_object_region_tuple)           # We turn the tuple into an numpy array [[h_1,w_1],[h_2,w_2]]

            object_region = swap(swapped_object_region, swap_inner_axis=True)       # We swap the array to get mathematical standard equivalent to [[w_1,h_1],[w_2,h_2]].
            object_center = get_center(object_region)                               # We get the region center.
            object_centers.append(object_region)                                    # We get the center into the list.

            # The region is defined by two points, the upper left corner (w1,h1) and the lower right corner (w2,h2). It is alto important that the upper is the  lower indexes position  in the image, 
            # so the lower is h and h1<h2. So the four corners from left to right and from up to down are (w1,h1), (w2,h1), (w1,h2), (w2,h2).
            w1 = object_region[0,0]
            w2 = object_region[1,0]
            h1 = object_region[0,1]
            h2 = object_region[1,1]
            upper_left_corner = [w1,h1]
            upper_right_corner = [w2,h1]
            lower_left_corner = [w1,h2]
            lower_right_corner = [w2,h2]
            # We create an array with the four points.
            object_region_corners = np.array([upper_left_corner, upper_right_corner,
                                                lower_left_corner, lower_right_corner])
                
            # We warp the points perspective.
            corrected_object_region_corners = use_homography_to_warp_coords_perspective(H, object_region_corners)

            # We swap again to get computational standard equivalent to corrected [[h1,w1],[h1,w2],[h2,w1],[h2,w1]]
            swapped_corrected_object_region = swap(corrected_object_region_corners, swap_inner_axis=True)

            corrected_object_center = get_center(corrected_object_region_corners, from_four_corners=True)                         # We get the center.
            
            if is_inside_polygon(corrected_polygon, corrected_object_center):    # If the point lies inside the corrected_polygon defined by the reference points...
                corrected_object_centers_in_polygon.append(corrected_object_center)
                corrected_object_region_corners_in_polygon.append(corrected_object_region_corners)
                if Config.TEST:
                    print("Object region")
                    print(object_region)
                    print("Object center")
                    print(object_center)
                    print("Object region")
                    print(object_region)
                    print("Object region corners.")
                    print(object_region_corners)
                    print("corrected object region corners.")
                    print(corrected_object_region_corners)
                    print("Swapped corrected object region.")
                    print(swapped_corrected_object_region)
                    print("corrected object center from corners")
                    print(corrected_object_center)

            # Draing on original image.
            camera_image = draw_rectangle_on_image(camera_image, object_region)     # We draw the region rectangle on the image.        
            cv2.circle(camera_image,object_center, 3, [255.,255.,0],-1)             # We draw the center.

    if Config.SAVE_FRAMES:
        cv2.imwrite(os.path.join(Config.ORIGINAL_IMAGES_WITH_OBJECTS_FOLDER, f"frame_{frame_index}.png"), camera_image)
    
    """
    TRACKER
    We associated track id to positions, regions, predictions and costs 
    """
    print("-----")
    associations, predictions, costs = tracker.update(swap(np.array(corrected_object_centers_in_polygon), swap_inner_axis=True))         # We give the centroids to the tracker in order to be tracked with computational standard. We get the associations.

    print("centers")
    print(swap(np.array(corrected_object_centers_in_polygon), swap_inner_axis=True))
    print("associations_positions")
    print(associations)
    print("predictions")
    print(predictions)

    corrected_object_regions_corners_associations = assign_regions_to_track_id(associations, corrected_object_region_corners_in_polygon, swap(np.array(corrected_object_centers_in_polygon), swap_inner_axis=True))

    for track_id in associations:
        print("track_id")
        print(track_id)
        # For each track id in associations...

        if not track_id in all_tracks_positions:
            # If the track is not registered, we initialize position, region, frame region lists and assign a color.
            all_tracks_positions[track_id] = []
            all_tracks_regions[track_id] = []
            all_tracks_color_record[track_id] = color_generator.next_color()
            all_tracks_frame_record[track_id] = []

        corrected_position = associations[track_id]
        corrected_object_region = corrected_object_regions_corners_associations[track_id]
        corrected_prediction = predictions[track_id]
        corrected_cost = costs[track_id]

        track_frames = all_tracks_frame_record[track_id]
        track_positions = all_tracks_positions[track_id]
        track_regions = all_tracks_regions[track_id]
        color = all_tracks_color_record[track_id]

        track_positions.append(corrected_position)
        track_regions.append(corrected_object_region)
        track_frames.append(frame_index)

        all_tracks_positions[track_id] = track_positions
        all_tracks_regions[track_id] = track_regions
        all_tracks_frame_record[track_id] = track_frames

        print("track positions")
        print(track_positions)
        print("track frames")
        print(track_frames)

        #Drawing on corrected image.
        if not corrected_object_region is None:
            corrected_image = draw_rectangle_on_image(corrected_image, np.int32(np.round(corrected_object_region)), color = [255.,0.,0], from_four_corners = True)

    corrected_image = draw_polygon_on_image(corrected_polygon, corrected_image)

    if Config.SAVE_FRAMES:
        cv2.imwrite(os.path.join(Config.CORRECTED_IMAGES_WITH_OBJECTS_FOLDER, f"frame_{frame_index}.png"), corrected_image)

    # We get the next image from the input.
    success, camera_image = vidcap.read()
    frame_index += 1

total_of_frames_computed = frame_index
print(f"Total of frames : {total_of_frames_computed}")

print("ALL FRAME POSITIONS AND REGIONS HAVE BEEN OBTAINED. NOW WE MUST FILL EMPTY POSITIONS.")

for track_id in all_tracks_positions.keys():
    print("track id")
    print(track_id)

    track_positions = all_tracks_positions[track_id]
    track_regions = all_tracks_regions[track_id]
    track_frame_record = all_tracks_frame_record[track_id]
    track_color = all_tracks_color_record[track_id]

    track_positions, track_regions, track_frame_record = fill_lost_track_points(track_positions, track_regions, track_frame_record)

    all_tracks_positions[track_id] = track_positions
    all_tracks_regions[track_id] = track_regions
    all_tracks_frame_record[track_id] = track_frame_record

    for index, frame_index in enumerate(track_frame_record):
        if os.path.isfile(os.path.join(Config.TRACK_IMAGES_FOLDER, f"frame_{frame_index}.png")):
            img = cv2.imread(os.path.join(Config.TRACK_IMAGES_FOLDER, f"frame_{frame_index}.png"))
        else:
            img = cv2.imread(os.path.join(Config.CORRECTED_IMAGES_WITH_OBJECTS_FOLDER, f"frame_{frame_index}.png"))

        img = draw_track_on_image(track_positions[:index+1], img, track_color)

        if Config.SAVE_FRAMES:
            cv2.imwrite(os.path.join(Config.TRACK_IMAGES_FOLDER, f"frame_{frame_index}.png"), img)

"""
SPEED CALCULATIONS
"""
print("NOW WE WILL APPLY SPEED CALCULATIONS.")

for track_id in all_tracks_positions:
    print(track_id)
    track_positions = all_tracks_positions[track_id]
    track_regions = all_tracks_regions[track_id]
    track_frame_record = all_tracks_frame_record[track_id]

    if Config.SPEED_ESTIMATION == "polynomial_average":
        speed_record = generate_speed_based_on_time_average(track_positions, NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED, Config.NUMBER_OF_FRAMES_TO_AVERAGE, METERS_BY_PIXEL, Config.TIME_LAPSE_TO_GET_SPEED, Config.MINIMUM_SPEED_TRESHOLD)

    if Config.SPEED_ESTIMATION == "simple_average":
        speed_record = generate_speed_based_on_simple_average(track_positions, METERS_BY_PIXEL, Config.MINIMUM_SPEED_TRESHOLD, Config.SEQUENCE_FRAMES_PER_SECOND)

    if Config.SPEED_ESTIMATION == "optical_flow":
        speed_record = generate_speed_based_on_optical_flow(track_regions, track_frame_record, all_corrected_images_paths, METERS_BY_PIXEL, Config.SEQUENCE_FRAMES_PER_SECOND, Config.SAVE_OPTICAL_FLOW_IMAGES, Config.OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER, Config.OPTICAL_FLOW_CORRECTED_IMAGES_FOLDER_WITH_OBJECTS_FOLDER)

    all_tracks_speed_record[track_id] = speed_record


"""
SAVE SPEED RESULTS.
"""
# We need to save the results in order to obtain pollution estimation.
# We create a matrix with Number of registered traces rows and number of frames columns.

speed_results = np.zeros(shape=(len(all_tracks_speed_record), total_of_frames_computed))

postprocessed_speed_results = np.zeros(shape=(len(all_tracks_speed_record), total_of_frames_computed))

# We generate the graphic and the speed record matrix.
color_list = []
speed_record_lists = []
for key_index, key in enumerate(all_tracks_speed_record):
    track_frame_record = all_tracks_frame_record[key]
    first_frame_index = track_frame_record[0]
    last_frame_index = track_frame_record[-1]    
    track_speed_record = all_tracks_speed_record[key]

    print(track_frame_record)
    print(track_speed_record)

    initial_not_useful_speed_values = np.sum(np.array(track_speed_record)==-1)
    assert not Config.SPEED_ESTIMATION=="polynomial_average" or initial_not_useful_speed_values<NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED
    print(initial_not_useful_speed_values)
    print(first_frame_index)
    print(last_frame_index)
    print(speed_results.shape)

    if not Config.SPEED_ESTIMATION=="polynomial_average" or len(track_speed_record) >= NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED:
        speed_results[key_index, first_frame_index+initial_not_useful_speed_values:last_frame_index+1] = np.array(track_speed_record[initial_not_useful_speed_values:])
    
        color = list(np.array(all_tracks_color_record[key])/255.)

        plt.plot(track_frame_record[initial_not_useful_speed_values:], track_speed_record[initial_not_useful_speed_values:], color = color)

if not os.path.isdir(Config.SPEED_ESTIMATION_FOLDER):
    os.makedirs(Config.SPEED_ESTIMATION_FOLDER)
    
np.save(f"{Config.SPEED_ESTIMATION_FOLDER}/speed_records.npy", speed_results)

plt.savefig(f"{Config.SPEED_ESTIMATION_FOLDER}/speed_records.png", dpi=2000)

plt.show()
"""
if (Config.POSTPROCESS_FILTER_TO_SPEED):
    for key_index, key in enumerate(all_tracks_speed_record):
        track_frame_record = all_tracks_frame_record[key]
        first_frame_index = track_frame_record[0]
        last_frame_index = track_frame_record[-1]
        track_speed_record = all_tracks_speed_record[key]

        print(track_speed_record)

        initial_not_useful_speed_values = np.sum(np.array(track_speed_record)==-1)
        assert initial_not_useful_speed_values<NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED
        print(initial_not_useful_speed_values)

        if not Config.SPEED_ESTIMATION=="polynomial_average" or len(track_speed_record) >= NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED:
            track_speed_record = postprocess_speed_record(track_speed_record[initial_not_useful_speed_values:], NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED)        
            postprocessed_speed_results[key_index, first_frame_index:track_frame_record[-1]+1] = np.array(track_speed_record)
            print("post")
            print(track_speed_record)
            color = list(np.array(all_tracks_color_record[key])/255.)
            plt.plot(track_frame_record, track_speed_record, color = color)
        
    np.save(f"{Config.SPEED_ESTIMATION_FOLDER}/postprocess_speed_records.npy", postprocessed_speed_results)

    plt.savefig(f"{Config.SPEED_ESTIMATION_FOLDER}/postprocess_speed_records.png", dpi=2000)
    
plt.show()
"""