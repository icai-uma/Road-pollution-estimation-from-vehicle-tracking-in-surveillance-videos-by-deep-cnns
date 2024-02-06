import cv2
import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import sleep

from general_utils import colors_generator, get_homography_matrix, use_homography_to_warp_image_perspective, use_homography_to_warp_coords_perspective, draw_rectangle_on_image, swap, get_center, print_text_on_image, draw_trace_on_image, apply_gaussian_filter, is_inside_polygon, draw_polygon_on_image
from tracker import Tracker
from Config import Config
import test_time_augmentation

if Config.MODEL_NAME == "yolov3":
    import yolov3_Region_Selector

if Config.MODEL_NAME == "yolov5":
    import yolov5_Region_Selector

    Region_Selector = yolov5_Region_Selector.Yolov5_Region_Selector

if "faster" in Config.MODEL_NAME:
    import tensorflow_zoo_Region_Selector
    import GPU_utils

    Region_Selector = tensorflow_zoo_Region_Selector.Tensorflow_Zoo_Region_Selector

matplotlib.use('TkAgg')

if "faster" in Config.MODEL_NAME:
    # We restrict video memory use.
    GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(0)

NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED = int(Config.SEQUENCE_FRAMES_PER_SECOND*Config.TIME_LAPSE_TO_GET_SPEED)    # Number of frames we will use to calculate speed.

if Config.DETECTION_MODE == "SAVE": object_detector = Region_Selector()                   # Object detector wrapper.

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

"""
CLICK Utils.
"""

#this function will be called whenever the mouse is clicked
def reference_onclick(event):

    global space_reference_coords
    global space_reference

    ix, iy = event.xdata, event.ydata
    print(int(event.button))
    print(event.key)
    print ('x = {}, y = {}'.format(ix,iy))

    if space_reference is None:
        if len(space_reference_coords) < 2:
            space_reference_coords.append(ix)
            print("x = {} saved as space reference.".format(ix))

        if len(space_reference_coords) == 2:
            space_reference = abs(int(space_reference_coords[0]) - int(space_reference_coords[1]))
            print("{} will be used as {} meters reference.".format(space_reference, Config.REAL_REFERENCE_LENGTH))
            print("Please. Close the current window.")
            space_reference_coords = []
            reference_fig.canvas.mpl_disconnect(reference_cid)

        plt.scatter(ix, iy, marker = 'x', s=20, c='r')
        plt.show()
            
    else:
        if len(space_reference_coords) < Config.NUMBER_OF_REFERENCE_COORDS:
            space_reference_coords.append((ix,iy))
            plt.scatter(ix, iy, marker = 'x', s=20, c='r')
            plt.show()
        if len(space_reference_coords) == Config.NUMBER_OF_REFERENCE_COORDS:
            space_reference_coords = np.array(space_reference_coords)
            reference_fig.canvas.mpl_disconnect(reference_cid)
            print("Please. Close the current window.")

def video_onclick(event):

    global video_reference_coords
    global space_reference_coords

    ix, iy = event.xdata, event.ydata
    print(int(event.button))
    print(event.key)
    print ('x = {}, y = {}'.format(ix,iy))

    if len(video_reference_coords) < Config.NUMBER_OF_REFERENCE_COORDS:
        video_reference_coords.append((ix,iy))
        plt.figure(1)
        plt.scatter(ix, iy, marker = 'x', s=20, c='r')
        plt.figure(0)
        if len(video_reference_coords) < Config.NUMBER_OF_REFERENCE_COORDS:
            plt.scatter(space_reference_coords[:len(video_reference_coords),0], space_reference_coords[:len(video_reference_coords),1], marker = 'x', s=20, c='b')
            plt.scatter(space_reference_coords[len(video_reference_coords),0], space_reference_coords[len(video_reference_coords),1], marker = 'x', s=20, c='r')
        plt.show()
    
    if len(video_reference_coords) == Config.NUMBER_OF_REFERENCE_COORDS:
        video_reference_coords = np.array(video_reference_coords)
        video_fig.canvas.mpl_disconnect(video_cid)
        print("Please. Close both current windows.")

def get_homography_calibration_points(reference_img, camera_img, select_new_coords, real_reference_length, preffix = ""):

    global space_reference_coords
    global video_reference_coords
    global space_reference
    global reference_fig
    global reference_cid
    global video_fig
    global video_cid

    if select_new_coords: 
        plt.figure(0)
        imgplot = plt.imshow(reference_img[:,:,[2,1,0]])

        reference_fig = plt.gcf()
        reference_cid = reference_fig.canvas.mpl_connect('button_press_event', reference_onclick)
        print(f"Please. Left click on two points as real meter reference. The horizontal difference between the two points will be used as {real_reference_length} meters")
        plt.show()

        plt.figure(1)
        imgplot = plt.imshow(reference_img[:,:,[2,1,0]])
        reference_fig = plt.gcf()
        reference_cid = reference_fig.canvas.mpl_connect('button_press_event', reference_onclick)
        print(f"Please. Left click on points to be used as homography references")
        plt.show()

        plt.figure(0)
        imgplot = plt.imshow(reference_img[:,:,[2,1,0]])
        plt.scatter(space_reference_coords[0,0], space_reference_coords[0,1], marker = 'x', s=100, c='r')
        plt.figure(1)
        imgplot = plt.imshow(camera_img[:,:,[2,1,0]])
        video_fig = plt.gcf()
        video_cid = video_fig.canvas.mpl_connect('button_press_event', video_onclick)
        print(f"Please. Left click on points to be used as camera references")
        plt.show()

        np.save(f"../input/{Config.PREFFIX}_space_reference_coords.npy", space_reference_coords)
        np.save(f"../input/{Config.PREFFIX}_video_reference_coords.npy", video_reference_coords)
        np.save(f"../input/{Config.PREFFIX}_space_real_reference.npy", np.array([space_reference])) 
    
    else:
        
        space_reference_coords = np.load(f"../input/{Config.PREFFIX}_space_reference_coords.npy")
        video_reference_coords = np.load(f"../input/{Config.PREFFIX}_video_reference_coords.npy")
        space_reference = np.load(f"../input/{Config.PREFFIX}_space_real_reference.npy")[0]

    #Establish matches image
    plt.figure(2)
    imgplot = plt.imshow(reference_img[:,:,[2,1,0]])
    plt.scatter(space_reference_coords[:,0], space_reference_coords[:,1], c="r", s=200)
    plt.savefig("../output/reference_image.png")

    plt.figure(3)
    imgplot = plt.imshow(camera_img[:,:,[2,1,0]])
    plt.scatter(video_reference_coords[:,0], video_reference_coords[:,1],  c="r", s=200)
    plt.savefig(f"../output/{Config.VIDEO_PREFFIX}_video_image.png")

    return space_reference_coords, video_reference_coords, space_reference
    
def postprocess_speed_record(record, L):
    """
        A speed record related to an object contains a list of N speed values S.
        The ith value (S_{i}) represents the average speed value between the point A_{i} (where the object was before) and the point B_{i} (where the object is when the value is stored).
        Since on this code, we do not calculate speed comparing each object position with its position in the inmediately previous frame. S_{i} is the average speed throughout a list of positions P_{i}
        with P_{i} = [p_{i}, p_{i+1}, ... , p{i+L-2}, p{i+L-1}], L = Config.TIME_LAPSE_TO_GET_SPEED*Config.SEQUENCE_FRAMES_PER_SECOND and p_{i+0} = A_{i} and p_{i+L-1} = B_{i}
        So, since the lists P overlap, we know we should have L speed values S_{i} for each position p_{j} with j>=L-1 and j<=N-1 and we can get a speed value with more precision for that points s_{j} if we average all S_{i} related to that point.
        Each point p_{j} will have K associated velocities with K=min(L, j+1, N+L-j-1) Note: N+L-j-1 = (N-1)+(L-1)-(j+1)
        s_{j} = mean([S_{k}]) \ k in [max(j-L+1,0), min(j,N-1)]
        
        Example with N=6 and L=5
        
        [p_{0}, p_{1}, p_{2}, p_{3}, p_{4}] has average speed S_{0}
        [p_{1}, p_{2}, p_{3}, p_{4}, p_{5}] has average speed S_{1}
        [p_{2}, p_{3}, p_{4}, p_{5}, p_{6}] has average speed S_{2}
        [p_{3}, p_{4}, p_{5}, p_{6}, p_{7}] has average speed S_{3}
        [p_{4}, p_{5}, p_{6}, p_{7}, p_{8}] has average speed S_{4}
        [p_{5}, p_{6}, p_{7}, p_{8}, p_{9}] has average speed S_{5}
        
        s_{0} = mean[S_{0}]
        s_{1} = mean([S_{0}, S_{1}])
        s_{2} = mean([S_{0}, S_{1}, S_{2}])
        s_{3} = mean([S_{0}, S_{1}, S_{2}, S_{3}])
        s_{4} = mean([S_{0}, S_{1}, S_{2}, S_{3}, S_{4}])
        s_{5} = mean([S_{1}, S_{2}, S_{3}, S_{4}, S_{5}])
        s_{6} = mean([S_{2}, S_{3}, S_{4}, S_{5}])
        s_{7} = mean([S_{3}, S_{4}, S_{5}])
        s_{8} = mean([S_{4}, S_{5}])
        s_{9} = mean([S_{5}])
        
        Example with N=2 and L=5
        
        [p_{0}, p_{1}, p_{2}, p_{3}, p_{4}] has average speed S_{0}
        [p_{1}, p_{2}, p_{3}, p_{4}, p_{5}] has average speed S_{1}
        
        s_{0} = mean[S_{0}]
        s_{1} = mean([S_{0}, S_{1}])
        s_{2} = mean([S_{0}, S_{1}])
        s_{3} = mean([S_{0}, S_{1}])
        s_{4} = mean([S_{0}, S_{1}])
        s_{5} = mean([S_{1}])
        s_{6} = mean([S_{1}])
        s_{7} = mean([S_{1}])
        s_{8} = mean([S_{1}])
        s_{9} = mean([S_{1}])
        
    """
    N = len(record)    
    speed_for_each_point = []
    
    for j in range(N+L-1):
    
        if j == 0:
            s_j= record[0]
        
        else:
            p_low_boundary = max(j-L+1,0)
            P_high_boundary = min(j+1,N)
            s_j = np.mean(np.array(record[p_low_boundary:P_high_boundary]))
            
        speed_for_each_point.append(s_j)
        
    return speed_for_each_point
    

# Object to generate colors to be used later.
color_generator = colors_generator(25,230)

# Image used as reference.
reference_image = cv2.imread(Config.REFERENCE_IMAGE_PATH)

# Video.
vidcap = cv2.VideoCapture(Config.CAMERA_VIDEO_PATH)
fourCC = cv2.VideoWriter_fourcc(*'XVID')

success, camera_image = vidcap.read()

print(f"Processing {Config.CAMERA_VIDEO_PATH}")
tracker = Tracker(Config.TRACKER_DIST_TRESHOLD, Config.TRACKER_MAX_FRAMES_TO_SKIP, Config.TRACKER_MAX_TRACE_LENGTH)

"""
It's very improtant to note that all coords used as reference are obtained from an image shown with matplotlib and are in standard mathematical format (x,y) with x the position over horizontal axis and y the position over vertical axis.
When we work with an image, the computational standard is (h,w) with h as position over vertical axis and w as position over horizontal axis so the coords will be manipulated.
We will use term "swap" to change from one standard to another and "swapped" to note information using computational standard.
"""

polygon = []
speed_record = {}
color_record = {}
frame_record = {}
tracks = {}
video_corrected = None
video_camera = None
video_union = None
union_shape = None
H = None
frame_index = 0

while success and frame_index<Config.NUMBER_OF_FRAMES_TO_COMPUTE:
    # While there are incomming frames and we have not reached the desired number of computed frames...

    print(f"Processed frames : {frame_index}/?")

    if H is None:                   #There is no homography matrix calculated yet.

        space_reference_coords, video_reference_coords, space_reference = get_homography_calibration_points(reference_image, camera_image, Config.SELECT_NEW_COORDS, Config.REAL_REFERENCE_LENGTH, preffix = Config.PREFFIX)      # We get the reference information.
        METERS_BY_PIXEL = Config.REAL_REFERENCE_LENGTH/space_reference
        H = get_homography_matrix(space_reference_coords, video_reference_coords)       # We obtain the homography matrix using coords with mathematical standard.

        if Config.TEST:
            print("H")
            print(H)

        OUTPUT_FOLDER = f"../output/{Config.MODEL_NAME}/{Config.VIDEO_PREFFIX}/"
        CORRECTED_IMAGES_FOLDER =f"{OUTPUT_FOLDER}/corrected/"
        FRAMES_IMAGES_FOLDER = f"{OUTPUT_FOLDER}/camera/"

        if not os.path.isdir(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        if not os.path.isdir(CORRECTED_IMAGES_FOLDER):
            os.makedirs(CORRECTED_IMAGES_FOLDER)

        corrected_image = use_homography_to_warp_image_perspective(H, camera_image, reference_image)    # We calculate the image with the correct perspective.
        corrected_coords = use_homography_to_warp_coords_perspective(H, video_reference_coords)         # We calculate the coords with the correct perspective.

        polygon = corrected_coords                                                                      # The corrected coords will be the polygin inside we will calculate speeds.
        plt.figure(4)                                                                                   # We show and save the image.
        imgplot = plt.imshow(corrected_image[:,:,[2,1,0]])
        plt.scatter(corrected_coords[:,0], corrected_coords[:,1], c="r", s=200)
        plt.savefig(f"{OUTPUT_FOLDER}/{Config.PREFFIX}_homography_image.png")
        
        plt.show()

        cv2.imwrite(f"{OUTPUT_FOLDER}/{Config.PREFFIX}_homography_image2.png", corrected_image)

        video_corrected = cv2.VideoWriter(f"{OUTPUT_FOLDER}/{Config.PREFFIX}_corrected_image.avi", fourCC, Config.SEQUENCE_FRAMES_PER_SECOND, (corrected_image.shape[1], corrected_image.shape[0]))
        video_camera = cv2.VideoWriter(f"{OUTPUT_FOLDER}/{Config.PREFFIX}_camera_image.avi", fourCC, Config.SEQUENCE_FRAMES_PER_SECOND, (camera_image.shape[1], camera_image.shape[0]))

        union_shape = (corrected_image.shape[0] + camera_image.shape[0], max(corrected_image.shape[1], camera_image.shape[1]))
        video_union = cv2.VideoWriter(f"{OUTPUT_FOLDER}/{Config.PREFFIX}_union_video.avi", fourCC, Config.SEQUENCE_FRAMES_PER_SECOND, (union_shape[1], union_shape[0]))

    if Config.DETECTION_MODE == "LOAD":
        output_dicts, _ = test_time_augmentation.load_semantic_information(Config.DETECTION_SAVE_FOLDER, frame_index, Config.TEST_TIME_TRANSFORMATIONS)
        output_dicts = test_time_augmentation.unmade_transformations_from_segmentation_data(output_dicts, Config.TEST_TIME_TRANSFORMATIONS, camera_image)
        output_dict = test_time_augmentation.join_test_time_augmentation_information_from_different_images(output_dicts)

    else:
        raise(NotImplementedError)

    swapped_object_regions_for_frame = output_dict["detection_boxes"]                                       # with two points (upper left corner and low right corner): [((h_1,w_1),(h_2,w_2))]
    classes_objects_for_frame = output_dict["detection_classes"]
    scores_objects_for_frame = output_dict["detection_scores"]

    corrected_image = use_homography_to_warp_image_perspective(H, camera_image, reference_image)            # We calculate the image with the correct perspective.

    object_centers = []                                                                                     # Objects Centers list
    warped_object_centers = []                                                                              # Warped Objects Centers list

    if Config.TEST:
        print("Objects regions:")
        print(object_regions_for_frame[0])

    for swapped_object_region_tuple, class_id, score in zip(swapped_object_regions_for_frame, classes_objects_for_frame, scores_objects_for_frame):
        # We get a tuple for region. The tuple will have structure ((h_1,w_1),(h_2,w_2))
        # so the region is defined using mathematical standard
        # by points (w_1,h_1), (w_2,h_1), (w_1,h_2) and (w_2,h_2).
        # We get also class id and score.

        if ((Config.CLASS_IDS_TO_INCLUDE is None) or (class_id in Config.CLASS_IDS_TO_INCLUDE)) and score > Config.DETECTION_SCORE_THRESHOLD: 

            swapped_object_region = np.array(swapped_object_region_tuple)           # We turn the tuple into an numpy array [[h_1,w_1],[h_2,w_2]]

            object_region = swap(swapped_object_region, swap_inner_axis=True)       # We swap the array to get mathematical standard equivalent to [[w_1,h_1],[w_2,h_2]].
                
            camera_image = draw_rectangle_on_image(camera_image, object_region)     # We draw the region rectangle on the image.        
            object_center = get_center(object_region)                               # We get the region image.
            cv2.circle(camera_image,object_center, 3, [255.,255.,0],-1)             # We draw the center.
            object_centers.append(object_region)                                    # We get the center into the list.
            
            #camera_image = draw_point_on_image(camera_image, object_center)                       # We draw the center in the image.

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
            warped_object_region_corners = use_homography_to_warp_coords_perspective(H, object_region_corners)

            # We swap again to get computational standard equivalent to warped [[h1,w1],[h1,w2],[h2,w1],[h2,w1]]
            swapped_warped_object_region = swap(warped_object_region_corners, swap_inner_axis=True)

            #corrected_image = draw_rectangle_on_image(corrected_image, swapped_warped_object_region, color = [255.,0.,0])  # We draw the regions on the corrected image.
            corrected_image = draw_rectangle_on_image(corrected_image, np.int32(np.round(warped_object_region_corners)), color = [255.,0.,0], from_four_corners = True)
            warped_object_center = get_center(warped_object_region_corners, from_four_corners=True)                         # We get the center.
            draw_polygon_on_image(polygon, corrected_image)
            if is_inside_polygon(polygon, warped_object_center):    # If the point lies inside the polygon defined by the reference points...
                warped_object_centers.append(warped_object_center)

                if Config.TEST:
                    print("Object region")
                    print(object_region)
                    print("Object center")
                    print(object_center)
                    print("Object region")
                    print(object_region)
                    print("Object region corners.")
                    print(object_region_corners)
                    print("Warped object region corners.")
                    print(warped_object_region_corners)
                    print("Swapped warped object region.")
                    print(swapped_warped_object_region)
                    print("Warped object center from corners")
                    print(warped_object_center)

    associations, predictions, costs = tracker.update(swap(np.array(warped_object_centers), swap_inner_axis=True))         # We give the centroids to the kalman filter in order to be tracked with computational standard. We get the associations.

    if Config.TEST:
        print(object_centers)

    # Now we will draw the tracks and get positions to calculate speed.

    for track_id in associations:
        # For each track id in associations...

        position = associations[track_id]
        prediction = predictions[track_id]
        cost = costs[track_id]

        if not track_id in tracks:
            # If the track id is unknown, we insert it in our dictionary to set it as known. We also asign a color to that track.
            tracks[track_id] = []
            color_record[track_id] = color_generator.next_color()

        # We get the color and the trace and append the position to that trace.
        color = color_record[track_id]
        trace = tracks[track_id]
        trace.append(position)

        # We draw the last positions of the trace over the corrected image.
        draw_trace_on_image(trace, corrected_image, color, max_number_of_trace_positions_to_draw = Config.NUMBER_OF_FRAMES_TO_AVERAGE, trace_positions_difference_to_draw = NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED, prediction=prediction, cost=cost, track_id=track_id)

        # We set the speed as a negative value in order to see it easily in the graph.
        speed = -1 * track_id

        if len(trace) > NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED:
            # If the trace has more points that the minimum number of positions to get a speed...

            # We will get NUMBER_OF_FRAMES_TO_AVERAGE positions from the NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED index and the end of the trace.
            old_traces_positions = []
            last_traces_positions = []
            for k in range (1, Config.NUMBER_OF_FRAMES_TO_AVERAGE+1):
                if not trace[-NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED + k-1] is None:
                    old_traces_positions.append(trace[-NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED + k-1])
                
                if not trace[-1*k] is None:
                    last_traces_positions.append(trace[-1*k])
                    
            old_traces_positions = np.array(old_traces_positions)
            last_traces_positions = np.array(last_traces_positions)


            # We calculate the average from last positions.
            if last_traces_positions.shape[0]:
                last_positions_average = np.mean(last_traces_positions, axis=0)
                last_positions_average_h = int(last_positions_average[0])
                last_positions_average_w = int(last_positions_average[1])
            else:
                last_positions_average = None

            # We calculate the average from old positions.
            if old_traces_positions.shape[0]:
                old_positions_average = np.mean(old_traces_positions, axis=0)
                old_positions_average_h = int(old_positions_average[0])
                old_positions_average_w = int(old_positions_average[1])
            else:
                old_positions_average = None

            # Pitagora's theorem to get the distance in pixels.
            if not last_positions_average is None and not old_positions_average is None:
                pixels_distance = math.sqrt(abs(last_positions_average_h - old_positions_average_h)**2 + abs(last_positions_average_w - old_positions_average_w)**2)
                
                # We turn pixel distance to meter distance.
                meters_distance = METERS_BY_PIXEL*pixels_distance

                # We calculate speed.
                speed = round(meters_distance / Config.TIME_LAPSE_TO_GET_SPEED,3)
                
                # If the speed is lower than a threshold, we gess the object is not moving and set the speed to 0.
                if speed < Config.MINIMUM_SPEED_TRESHOLD: speed = 0
                
                # Print info.
                print(f"Track {track_id} We set the speed using {METERS_BY_PIXEL} meters by pixel and {pixels_distance} pixel distance with resulting {meters_distance} meters and speed {speed} m/s.")
            
            else:
                speed = None           

            # We print the text on the image.
            #if not last_positions_average is None:
                #print_text_on_image(corrected_image, str(speed), (last_positions_average_h,last_positions_average_w))            

            # If the track id has speed record, we get it. Else, we create it and the associated color.
            if track_id in speed_record:
                record = speed_record[track_id]

            else:
                record = []
                color_record[track_id] = color

            # If the track id has frame record, we get it. Else, we create it.
            if track_id in frame_record:
                frames = frame_record[track_id]

            else:
                frames = []

            # We assign frame index.
            frames.append(frame_index)
            frame_record[track_id] = frames
            if speed is None:
                if len(record): speed = record[-1]
                else: speed = 0
                
            record.append(speed)
            speed_record[track_id] = record

    # We add images to videos.
    video_camera.write(camera_image)
    video_corrected.write(corrected_image)
    print_text_on_image(corrected_image, str(frame_index), (corrected_image.shape[0]-30, 0), color = (125,125,125))
    
    if Config.SAVE_CORRECTED_IMAGES:
        if not os.path.isdir(CORRECTED_IMAGES_FOLDER):
            os.makedirs(CORRECTED_IMAGES_FOLDER)
        cv2.imwrite(os.path.join(CORRECTED_IMAGES_FOLDER, "corrected_frame_{}.png".format(frame_index)), corrected_image)

    if Config.SAVE_FRAMES:
        if not os.path.isdir(FRAMES_IMAGES_FOLDER):
            os.makedirs(FRAMES_IMAGES_FOLDER)
        cv2.imwrite(os.path.join(FRAMES_IMAGES_FOLDER, "frame_{}.png".format(frame_index)), camera_image)
        
    if Config.FRAME_BY_FRAME:
        cv2.imshow("Corrected image", corrected_image)
        cv2.waitKey(0)
    
    if Config.TEST:
        plt.figure(4)
        imgplot = plt.imshow(camera_image[:,:,[2,1,0]])
        plt.figure(5)
        imgplot = plt.imshow(corrected_image[:,:,[2,1,0]])
        plt.show()

    # We union images into one image to add it to its video.
    print(union_shape)
    print(camera_image.shape)
    print(corrected_image.shape)
    union_image = np.zeros(shape=(union_shape[0], union_shape[1], 3))
    union_image[0:camera_image.shape[0], :camera_image.shape[1], :] = camera_image
    union_image[camera_image.shape[0]:, :corrected_image.shape[1], :] = corrected_image
    video_union.write(np.uint8(union_image))

    # We get the next image from the input.
    success, camera_image = vidcap.read()

    frame_index += 1

# We destroy opencv windows and save videos.
cv2.destroyAllWindows()
video_corrected.release()
video_camera.release()
video_union.release()

# We need to save the results in order to obtain pollution estimation.
# We create a matrix with Number of registered traces rows and number of frames columns.

speed_results = np.zeros(shape=(len(speed_record), frame_index))

postprocessed_speed_results = np.zeros(shape=(len(speed_record), frame_index))

# We generate the graphic.
color_list = []
speed_record_lists = []
for key_index, key in enumerate(speed_record):
    frames = frame_record[key]
    record = speed_record[key]
    
    speed_results[key_index, frames[0]:frames[-1]+1] = np.array(record)
    
    color = list(np.array(color_record[key])/255.)

    plt.plot(frames, record, color = color)
    
np.save(f"{OUTPUT_FOLDER}/{Config.VIDEO_PREFFIX}_{Config.MODEL_NAME}_speed_records.npy", speed_results)

plt.savefig(f"{OUTPUT_FOLDER}/{Config.VIDEO_PREFFIX}_{Config.MODEL_NAME}_speed_records.png", dpi=1000)

plt.show()

if (Config.POSTPROCESS_FILTER_TO_SPEED):
    for key_index, key in enumerate(speed_record):
        frames = frame_record[key]
        first_frame_index = frames[0]        
        record = speed_record[key]
        print(record)
        record = postprocess_speed_record(record, NUMBERS_OF_FRAMES_TO_GET_DISTANCE_TO_GET_SPEED)
        extra_frames_number = len(record) - len(frames)
        extended_frames = list(range(first_frame_index-extra_frames_number, first_frame_index)) + frames
        
        postprocessed_speed_results[key_index, extended_frames[0]:extended_frames[-1]+1] = np.array(record)

        color = list(np.array(color_record[key])/255.)
        plt.plot(extended_frames, record, color = color)
        
    np.save(f"{OUTPUT_FOLDER}/{Config.VIDEO_PREFFIX}_{Config.MODEL_NAME}_postprocess_speed_records.npy", postprocessed_speed_results)

    plt.savefig(f"{OUTPUT_FOLDER}/{Config.VIDEO_PREFFIX}_{Config.MODEL_NAME}_postprocess_speed_records.png", dpi=1000)
    
plt.show()

if (Config.APPLY_GAUSSIAN_FILTER_TO_SPEED):
    for key in speed_record:
        frames = frame_record[key]
        record = speed_record[key]

        record = apply_gaussian_filter([record], Config.GAUSSIAN_FILTER_SIGMA)[0]
        color = list(np.array(color_record[key])/255.)

        plt.plot(frames, record, color = color)

    plt.savefig(f"{OUTPUT_FOLDER}/{Config.VIDEO_PREFFIX}_{Config.MODEL_NAME}_filtered_speed_records.png", dpi=1000)
    
plt.show()

print(f"Total of frames : {frame_index}")
print("Cannot get video image.")
