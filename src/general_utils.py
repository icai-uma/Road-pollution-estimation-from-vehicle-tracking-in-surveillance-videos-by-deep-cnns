import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import scipy
import math
import json
import os

from Config import Config

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image

def optical_flow_to_rgb(flow, image):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(image)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return rgb

def get_optical_flow_using_gunnar_farneback(prev_image, next_image):
    flow = cv2.calcOpticalFlowFarneback(grayscale(prev_image), grayscale(next_image), None, 0.5, 5, 15, 3, 5, 1.2, 0)
    flow_image = optical_flow_to_rgb(flow,next_image)
    return flow, flow_image

def pixel_distance_between_two_points(point_1, point_2):
    # Pitagora's theorem to get the distance.
    point_1_h = point_1[0]
    point_1_w = point_1[1]
    point_2_h = point_2[0]
    point_2_w = point_2[1]
    pixels_distance = math.sqrt(abs(point_1_h - point_2_h)**2 + abs(point_1_w - point_2_w)**2)

    return pixels_distance

def pixel_per_frame_speed_to_meter_per_second(input_speed, meters_by_pixel, frames_per_second):
    return input_speed*meters_by_pixel*frames_per_second

def meters_distance_between_two_points(point_1, point_2, meters_by_pixel):
    pixels_distance = pixel_distance_between_two_points(point_1, point_2)
    print(pixels_distance)
    meters_distance = meters_by_pixel*pixels_distance

    return meters_distance

def speed_from_two_points(point_1, point_2, meters_by_pixel, time_lapse):
    meters_distance = meters_distance_between_two_points(point_1, point_2, meters_by_pixel)
    print(meters_by_pixel)
    print(meters_distance)
    print(time_lapse)
    speed = round(meters_distance / time_lapse,3)

    return speed

def get_track_average_positions(track, number_of_positions_to_average, number_of_positions_to_get_distance):
    print("--")
    print(track)
    old_tracks_positions = []
    last_tracks_positions = []
    for k in range (1, number_of_positions_to_average+1):
        if not track[-number_of_positions_to_get_distance + k-1] is None:
            old_tracks_positions.append(track[-number_of_positions_to_get_distance + k-1])
                    
        if not track[-1*k] is None:
            last_tracks_positions.append(track[-1*k])

    old_tracks_positions = np.array(old_tracks_positions)
    print(old_tracks_positions)
    last_tracks_positions = np.array(last_tracks_positions)
    print(last_tracks_positions)

    if not last_tracks_positions.shape[0]:
        last_tracks_positions = None

    if not old_tracks_positions.shape[0]:
        old_tracks_positions = None

    return last_tracks_positions, old_tracks_positions

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
                temp_img_split = np.zeros(shape=shape).astype(np.uint8)
                temp_img_split[:,:,:] = 128
                temp_img_split[:img_split.shape[0], :img_split.shape[1], :] = img_split
                img_split = temp_img_split

            # print([(low_h_index, low_w_index),(high_h_index, high_w_index)])
            split_list.append(img_split)
            split_position.append([(low_h_index, low_w_index),(high_h_index, high_w_index)])

    return split_list, split_position
    
def save_to_json(dictionary, path_to_save):
    json_dict = json.dumps(dictionary)
    f = open(path_to_save, "w")
    f.write(json_dict)
    f.close()
    
def load_from_json(path_to_load):
    with open(path_to_load) as json_file:
        data = json.load(json_file)
        return data

def get_homography_matrix(objective_reference_coords, true_coords):

    #Find homography
    h, mask = cv2.findHomography(true_coords,objective_reference_coords, cv2.RANSAC)
    return h

def use_homography_to_warp_image_perspective(H, image_to_warp, reference_img):

    reference_height, reference_width, channels = reference_img.shape
    corrected_image = cv2.warpPerspective(image_to_warp, H, (reference_width,reference_height))

    return corrected_image

def use_homography_to_warp_coords_perspective(H, coords):

    # Using as reference https://www.learnopencv.com/homography-examples-using-opencv-python-c/
    # https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143

    warped_coords = -1*np.ones(shape=coords.shape)
    # print(coords)
    for index, coord in enumerate(coords):
        # print("coord")
        # print(coord)
        warped_coord_aux = H.dot(np.array([coord[0], coord[1], 1]))
        # print("warped coords aux")
        # print(warped_coord_aux)
        warped_coord = warped_coord_aux[:2]/warped_coord_aux[2]
        # print("warped coord")
        # print(warped_coord)
        warped_coords[index, :] = warped_coord

    # print(warped_coords)
    return warped_coords
    
# Function to turn from speed in km/h to pollution per km.
# speed : float
# fuel : string "diesel" or "petrol"
def from_speed_to_pollution_km(speed, fuel = "mix"):
    # source: https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/662795/updated-emission-curves-ntm.pdf
    # using 2020 coefficents
    if speed <=1 : return 0
    
    if fuel == "mix":
        petrol_car_share = 0.44
        diesel_car_share = 0.56
        
        y = petrol_car_share*from_speed_to_pollution_km(speed, fuel="petrol") + diesel_car_share*from_speed_to_pollution_km(speed, fuel="diesel")        
    
    if fuel == "petrol":
        a = 0.01185628
        b = 0.00034047
        c = 1.2576*(10**-6)
        d = 1.0462*(10**-7)
        e = -7.216*(10**-10)
        f = 6.0976*(10**-12)
        g = 0
        
        y = (a+b*speed+c*speed**2+d*speed**3+e*speed**4+f*speed**5+g*speed**6)/speed

    if fuel == "diesel":
        a = 0.02918783
        b = 0.0013909
        c = 2.8984*(10**-5)
        d = -6.175*(10**-7)
        e = 9.9971*(10**-9)
        f = -7.31*(10**-11)
        g = 2.1786*(10**-13)        
    
        y = (a+b*speed+c*speed**2+d*speed**3+e*speed**4+f*speed**5+g*speed**6)/speed
        
    return y
    
# Function to turn from speed in km/h to pollution per frame.
# speed : float
# fuel : string "diesel" or "petrol"
def from_speed_to_pollution_per_frame(v, fps = 30, fuel = "mix"):

    H = fps
    dt = 1/H
    dr = v/3600 * dt # km/frame
    F_v_k = from_speed_to_pollution_km(v, fuel = fuel)  # g/km
    
    dz = F_v_k * dr # g/frame
    
    return dz
    
# Function to turn from speed in km/h to pollution g/frame
# speed : np array with 2 dimensions.
# fuel : string "diesel" or "petrol"
def from_speed_to_pollution_matrix(speed_matrix, fps=30, fuel = "mix"):
    # source: https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/662795/updated-emission-curves-ntm.pdf
    # using 2020 coefficents
    
    assert len(speed_matrix.shape) == 2
    
    new_speed_matrix = np.zeros(shape = speed_matrix.shape)
    
    for ii in range(speed_matrix.shape[0]):
        for jj in range(speed_matrix.shape[1]):
            new_speed_matrix[ii,jj] = from_speed_to_pollution_per_frame(speed_matrix[ii,jj], fps=fps, fuel = fuel)    
            
    return new_speed_matrix

def get_frame_rate(video):

    # Find OpenCV version

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :

        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)

        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    else :

        fps = video.get(cv2.CAP_PROP_FPS)

        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    return fps


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

def point_index_in_list(point, points_list):
    for index, p in enumerate(points_list):
        if p[0]==point[0] and p[1]==point[1]:
            return index

def assign_regions_to_track_id(associations, region_corners, objects_centers):
    regions_dict = {}
    for track_id in associations:
        position = associations[track_id]
        if position is not None:
            regions_dict[track_id] = region_corners[point_index_in_list(position, objects_centers)]
        else:
            regions_dict[track_id] = None

    print(regions_dict)
    return regions_dict
        

def get_homography_calibration_points(reference_img, camera_img, select_new_coords, real_reference_length, input_folder, output_folder, preffix = ""):

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

        np.save(f"{input_folder}/space_reference_coords.npy", space_reference_coords)
        np.save(f"{input_folder}/video_reference_coords.npy", video_reference_coords)
        np.save(f"{input_folder}/space_real_reference.npy", np.array([space_reference])) 
    
    else:
        
        space_reference_coords = np.load(f"{input_folder}/space_reference_coords.npy")
        video_reference_coords = np.load(f"{input_folder}/video_reference_coords.npy")
        space_reference = np.load(f"{input_folder}/space_real_reference.npy")[0]

    #Establish matches image
    plt.figure(2)
    imgplot = plt.imshow(reference_img[:,:,[2,1,0]])
    plt.scatter(space_reference_coords[:,0], space_reference_coords[:,1], c="r", s=200)
    plt.savefig(f"{output_folder}/reference_image.png")

    plt.figure(3)
    imgplot = plt.imshow(camera_img[:,:,[2,1,0]])
    plt.scatter(video_reference_coords[:,0], video_reference_coords[:,1],  c="r", s=200)
    plt.savefig(f"{output_folder}/video_image.png")

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
        
        s_{0} = mean([S_{0}])
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
        
        s_{0} = mean([S_{0}])
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

def draw_rectangle_on_image(img, rectangle_corners, color = [0.,255.,0.], from_four_corners = False):
    # We wait the corners as points in mathematical standard (w,h).
    img = img.copy()
    if not from_four_corners:
        if type(rectangle_corners) is tuple:

            img = cv2.rectangle(img, (int(rectangle_corners[0][0]), int(rectangle_corners[0][1])), (int(rectangle_corners[1][0]), int(rectangle_corners[1][1])), color, 1)

        if type(rectangle_corners) is np.ndarray:

            img = cv2.rectangle(img, (int(rectangle_corners[0,0]), int(rectangle_corners[0,1])), (int(rectangle_corners[1,0]), int(rectangle_corners[1,1])), color, 1)

    else:
        #This section we guess rectangle corners is a numpy array like follows: [[w1,h1], [w2,h1], [w1,h2], [w2,h2]]

        if type(rectangle_corners) is tuple:

            print("Draw rectangle on image with from four corners True and rectangle corners tuple not defined.")

            quit()

        if type(rectangle_corners) is np.ndarray:
            #This section we guess rectangle corners is a numpy array like follows: [[w1,h1], [w2,h1], [w1,h2], [w2,h2]]

            upper_left_corner = (rectangle_corners[0,0], rectangle_corners[0,1])
            upper_right_corner = (rectangle_corners[1,0], rectangle_corners[1,1])
            lower_left_corner = (rectangle_corners[2,0], rectangle_corners[2,1])
            lower_right_corner = (rectangle_corners[3,0], rectangle_corners[3,1])

        img = cv2.line(img, upper_left_corner, upper_right_corner, color, 1)
        img = cv2.line(img, upper_right_corner, lower_right_corner, color, 1)
        img = cv2.line(img, lower_right_corner, lower_left_corner, color, 1)
        img = cv2.line(img, lower_left_corner, upper_left_corner, color, 1)

    return img
    
def draw_point_on_image(img, point_coords, point_grossor = 5, color = [255.,0.,255.]):
    point_grossor_offset = int((point_grossor-1)/2)
    img[int(point_coords[0])-point_grossor_offset:int(point_coords[0])+point_grossor_offset+1,int(point_coords[1])-point_grossor_offset:int(point_coords[1])+point_grossor_offset+1,:] = color
    
    return img

def swap(vector, swap_inner_axis = False):
    new_vector = np.zeros(vector.shape)

    if not swap_inner_axis:

        new_vector[0] = vector[1]
        new_vector[1] = vector[0]

    else:

        for i in range(new_vector.shape[0]):

            new_vector[i,0] = vector[i,1]
            new_vector[i,1] = vector[i,0]

    return new_vector

def get_center(region, from_four_corners=False):

    # We expect corners as points in mathematical standard (w,h).
    if not from_four_corners:

        
        left_side_width = region[0,0]   # The left side width
        right_side_width = region[1,0]  # The right side width.
        upper_side_height = region[0,1] # The upper side height. It is important to note that the higher position in image, the lower height value.
        lower_side_height = region[1,1] # The lower side height. It is important to note that the lower position in image, the higher height value.
        
        """
        center = (int(left_side_width + (right_side_width - left_side_width)/2), 
                    int(upper_side_height + (lower_side_height - upper_side_height)/2))
        """
        
        corner_1 = [left_side_width, upper_side_height] # Up-Left
        corner_2 = [right_side_width, upper_side_height] # Up-Right
        corner_3 = [left_side_width, lower_side_height] # Down-Left
        corner_4 = [right_side_width, lower_side_height] # Down-Right
        
        corners = np.array([corner_1, corner_2, corner_3, corner_4])
        
    else:

        # print(region)
        """
        leftist_point_width = np.min(region[:,0])   # We get the width displacement from the point with the minimum width displacement.
        rightist_point_width = np.max(region[:,0])  # We get the width displacement from the point with the maximum width displacement.
        upper_point_height = np.min(region[:,1])    # We get the height displacement from the point with the minimum height displacement.
        lower_point_height = np.max(region[:,1])    # We get the height displacement from the point with the maximum height displacement.
        
        center = (int(leftist_point_width + (rightist_point_width - leftist_point_width)/2), 
                    int(upper_point_height + (lower_point_height - upper_point_height)/2))
        """
        corners = region

    center = np.mean(corners, axis=0)
    return (int(center[0]), int(center[1]))

def print_text_on_image(image, text_to_print, position, color = (125,85,255)):

    font                   = cv2.FONT_HERSHEY_PLAIN
    bottomLeftCornerOfText = (position[1],position[0])
    fontScale              = 1
    fontColor              = color
    lineType               = 2

    cv2.putText(image,
        text_to_print, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

def draw_points(list_of_points, image, color, radius):
    for point in list_of_points:
        print(point)
        if not point is None:
            h = point[0]
            w = point[1]
            cv2.circle(image,(int(w),int(h)), radius, color, -1)
    
    return image

def draw_track_on_image(track, image, color):

    image = image.copy()

    image = draw_points(track, image, color, 3)

    return image


def draw_track_on_image_old(track, image, color, max_number_of_track_positions_to_draw = 5, track_positions_difference_to_draw = 30, radius = 0, prediction = None, cost = None, track_id = None):
    image = image.copy()
    for k in range(1,min(max_number_of_track_positions_to_draw,len(track))+1):
        position = track[-k]
        if not position is None:
            h = position[0]
            w = position[1]

            if k == 1:
                size = 6
                if radius != 0:
                    cv2.circle(image,(int(w),int(h)), radius, color, 1)
            else:
                size = 3

            cv2.circle(image,(int(w),int(h)), size,color,-1)
        
        if len(track) >= track_positions_difference_to_draw: old_position = track[-track_positions_difference_to_draw + k-1]            
     
        else: old_position = None
        
        if not old_position is None:
            h = old_position[0]
            w = old_position[1]
            size = 3

            cv2.circle(image,(int(w),int(h)), size,color,-1)
     
    if len(track) >= track_positions_difference_to_draw:
        old_position = track[-track_positions_difference_to_draw]
        last_position = track[-1]
        if not last_position is None and not old_position is None:
                    
            l_p_h = last_position[0]
            l_p_w = last_position[1]
            o_p_h = old_position[0]
            o_p_w = old_position[1]
            cv2.line(image, (int(l_p_w), int(l_p_h)), (int(o_p_w), int(o_p_h)), color, 2)

    if not prediction is None:
        cv2.drawMarker(image,(int(prediction[0][1]),int(prediction[0][0])), color,cv2.MARKER_CROSS)

    if cost and cost > 5:
        pos_h = track[-1][0]
        pos_w = track[-1][1]
        pred_h = prediction[0][0]
        pred_w = prediction[0][1]
        ave_w = (pos_w+pred_w)/2
        ave_h = (pred_h+pred_h)/2
        #cv2.line(image, (int(pred_w),int(pred_h)), (int(pos_w),int(pos_h)), color, 1)
        #print_text_on_image(image, str(round(cost,4)), (int(ave_h),int(ave_w)), color = color)

    if track_id:
        print_text_on_image(image, str(track_id), (int(prediction[0][0])-5,int(prediction[0][1])), color = (255,255,255))

    return image

def draw_polygon_on_image(polygon, image, color = [0.,0,255.]):
    image = image.copy()
    first_position = polygon[0]
    for index, point in enumerate(polygon):
        if index < len(polygon)-1:
            next_point = polygon[index+1]
            cv2.line(image, (int(point[0]),int(point[1])), (int(next_point[0]),int(next_point[1])), color, 1)
        else:
            cv2.line(image, (int(point[0]),int(point[1])), (int(first_position[0]),int(first_position[1])), color, 1)
    return image

def apply_gaussian_filter(list_of_tracks, gaussian_filter_sigma = 0.25):
    list_of_filtered_tracks = []
    for track in list_of_tracks:
        filtered_track = scipy.ndimage.gaussian_filter(track, sigma=gaussian_filter_sigma)
        list_of_filtered_tracks.append(filtered_track)
    return list_of_filtered_tracks

class colors_generator(object):


    def __init__(self, low_range, top_range, color_step = 10):
        self.low_range = low_range
        self.top_range = top_range
        self.last_color = None
        self.index = 0
        self.color_step = color_step
        self.count = 0

    def next_color(self):
        
        """
        if self.last_color is None:

            self.last_color = [self.low_range, self.top_range, self.low_range]

        else:
            
            if self.count%3 == 0:
                self.last_color[0] = self.last_color[0] + self.color_step

                if self.last_color[0] > 255. or self.last_color[0] < 0.:
                    print("ERROR. COLOR OUT OF RANGE")
                    print(self.last_color)
                    quit()
            
            if self.count%3 == 1:
                self.last_color[1] = self.last_color[1] - self.color_step

                if self.last_color[1] > 255. or self.last_color[1] < 0.:
                    print("ERROR. COLOR OUT OF RANGE")
                    print(self.last_color)
                    quit()

            if self.count%3 == 2:
                self.last_color[2] = self.last_color[2] + self.color_step

                if self.last_color[2] > 255. or self.last_color[2] < 0.:
                    print("ERROR. COLOR OUT OF RANGE")
                    print(self.last_color)
                    quit()
        """

        self.last_color = [int(random.uniform(self.low_range, self.top_range)), int(random.uniform(self.low_range, self.top_range)), int(random.uniform(self.low_range, self.top_range))]
        self.count += 1
        return self.last_color.copy()

def get_values_in_polygon_from_matrix(matrix, polygon):
    # We expect polygon as alisto containing points [[h1,w1],[h2,w2],[h3,w3]]
    #print(polygon)

    values_from_matrix_in_polygon = []
    np_polygon = np.array(polygon)
    for x in range(int(np.min(np_polygon[:,0])), int(np.max(np_polygon[:,0]))+1):
        for y in range(int(np.min(np_polygon[:,1])), int(np.max(np_polygon[:,1]))+1):
            value = matrix[y,x]

            if is_inside_polygon(polygon, [x,y]):
                values_from_matrix_in_polygon.append(value)

    return values_from_matrix_in_polygon

def average_values_in_polygon_from_matrix(matrix, polygon):
    values_in_polygon = get_values_in_polygon_from_matrix(matrix, polygon)
    average = np.sum(np.array(values_in_polygon))/len(values_in_polygon)

    return average

# A Python3 program to check if a given point  
# lies inside a given polygon 
# Refer https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/ 
# for explanation of functions onSegment(), 
# orientation() and doIntersect()  
 
# Define Infinite (Using INT_MAX  
# caused overflow problems)
INT_MAX = 10000
 
# Given three colinear points p, q, r,  
# the function checks if point q lies 
# on line segment 'pr' 
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 
# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
 
def doIntersect(p1, q1, p2, q2):
     
    # Find the four orientations needed for  
    # general and special cases 
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # Special Cases 
    # p1, q1 and p2 are colinear and 
    # p2 lies on segment p1q1 
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are colinear and 
    # q2 lies on segment p1q1 
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are colinear and 
    # p1 lies on segment p2q2 
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are colinear and 
    # q1 lies on segment p2q2 
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 
# Returns true if the point p lies  
# inside the polygon[] with n vertices 
def is_inside_polygon(points:list, p:tuple) -> bool:
     
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        # Check if the line segment from 'p' to  
        # 'extreme' intersects with the line  
        # segment from 'polygon[i]' to 'polygon[next]' 
        if (doIntersect(points[i],
                        points[next], 
                        p, extreme)):
                             
            # If the point 'p' is colinear with line  
            # segment 'i-next', then check if it lies  
            # on segment. If it lies, return true, otherwise false 
            if orientation(points[i], p, 
                           points[next]) == 0:
                return onSegment(points[i], p, 
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
         
    # Return true if count is odd, false otherwise 
    return (count % 2 == 1)


def generate_speed_based_on_time_average(position_list, number_of_frames_to_get_speed, number_of_frames_to_average, meters_by_pixel, time_lapse_to_get_speed, minimum_speed_treshold):

    speed_record = (min(number_of_frames_to_get_speed-1,len(position_list)))*[-1]

    print("-")
    print(len(position_list))
    
    for frame_index in range(number_of_frames_to_get_speed-1, len(position_list)):
        print(frame_index)
        # We will use number_of_frames_to_average positions from the number_of_frames_to_get_speed index and the end of the track at that frame.
        last_positions, old_positions = get_track_average_positions(position_list[:frame_index+1], number_of_frames_to_average, number_of_frames_to_get_speed)

        print(last_positions)
        print(old_positions)

        assert not last_positions is None and not old_positions is None

        # We calculate the average from last positions.
        last_positions_average = np.mean(last_positions, axis=0)

        # We calculate the average from old positions.
        old_positions_average = np.mean(old_positions, axis=0)

        # Pitagora's theorem to get the distance in pixels.
        speed = speed_from_two_points(last_positions_average, old_positions_average, meters_by_pixel, time_lapse_to_get_speed)
                        
        # If the speed is lower than a threshold, we gess the object is not actually moving and set the speed to 0.
        if speed < minimum_speed_treshold: speed = 0

        speed_record.append(speed)

    print(len(speed_record))
    print(len(position_list))
    assert len(speed_record) == len(position_list)

    return speed_record

def generate_speed_based_on_simple_average(position_list, meters_by_pixel, minimum_speed_treshold, sequence_frames_per_second):
    last_position = position_list[-1]
    first_position = position_list[0]
    time_diference = len(position_list)/sequence_frames_per_second
    speed = speed_from_two_points(last_position, first_position, meters_by_pixel, time_diference)

    #If the speed is lower than a threshold, we gess the object is not actually moving and set the speed to 0.
    print(first_position)
    print(last_position)
    print(speed)
    if speed < minimum_speed_treshold: speed = 0

    speed_record = len(position_list)*[speed]

    #if speed == 0 and len(position_list) > 1:
        #print(position_list)
        #quit()

    return speed_record

def generate_speed_based_on_optical_flow(region_list, frame_record, images_paths, meters_by_pixel, sequence_frames_per_second, save_optical_flow_images, optical_flow_images_folder, optical_flow_with_objects):

    #We will get speed based on optical flow using Gunnar Farneback's algorithm.
    speed_record = []

    image = None
    previous_image = None

    assert len(region_list) == len(frame_record)

    for frame_index, region in zip(frame_record, region_list):
        if not image is None:
            previous_image = image            
        else:
            speed_record.append(-1)

        image = cv2.imread(images_paths[frame_index])

        if not previous_image is None:
            optical_flow, optical_flow_image = get_optical_flow_using_gunnar_farneback(previous_image, image)
            optical_flow_absolute_speeds = np.sqrt(np.power(optical_flow[:,:,0],2), np.power(optical_flow[:,:,1],1))

            if save_optical_flow_images:
                cv2.imwrite(os.path.join(optical_flow_images_folder, f"optical_flow_frame_{frame_index}.png"), optical_flow_image)

                meters_per_second_for_each_pixel = pixel_per_frame_speed_to_meter_per_second(optical_flow_absolute_speeds, meters_by_pixel, sequence_frames_per_second)
                speed = average_values_in_polygon_from_matrix(meters_per_second_for_each_pixel, region)

                optical_flow_with_folders_frame_path = os.path.join(optical_flow_with_objects, f"optical_flow_frame_{frame_index}.png")

                if os.path.isfile(optical_flow_with_folders_frame_path):
                    optical_flow_image = cv2.imread(optical_flow_with_folders_frame_path)

                optical_flow_image = draw_rectangle_on_image(optical_flow_image, np.int32(np.round(region)), color = [255.,0.,0], from_four_corners = True)

                cv2.imwrite(optical_flow_with_folders_frame_path, optical_flow_image)

                speed_record.append(speed)

    assert len(speed_record) == len(frame_record)
    return speed_record

def fill_lost_track_points(track_positions, track_regions, track_frames):

    print("-------------------")
    # We will fill none values from track by interpolating.
    assert len(track_positions) == len(track_regions) == len(track_frames)
    print(track_positions)
    print(track_regions)
    last_non_none_position = None
    last_non_none_region = None
    none_counts = 0
    nones_at_first_positions = 0
    for index_position, (position, region) in enumerate(zip(track_positions, track_regions)):
        if not position is None:

            if none_counts > 0:
                if not last_non_none_position is None:
                    #Interpolate
                    print("--")
                    print(index_position)
                    print(np.array(last_non_none_position))
                    print(np.array(position))
                    interpolated_positions = list(np.linspace(np.array(last_non_none_position), np.array(position), none_counts+2))[1:-1]
                    track_positions[index_position-none_counts:index_position] = interpolated_positions
                    print("-")
                    print(np.array(last_non_none_region))
                    print(np.array(region))
                    interpolated_regions = np.linspace(np.array(last_non_none_region), np.array(region), none_counts+2)[1:-1]
                    track_regions[index_position-none_counts:index_position] = interpolated_regions

                else:
                    #We do not have positions to interpolate so we simply ignore.
                    nones_at_first_positions = none_counts

            last_non_none_position = position
            last_non_none_region = region
            none_counts = 0
        else:
            none_counts += 1
    
    for i in range(nones_at_first_positions):
        # We will delete all Nones at initial positions.
        track_positions.pop(0)
        track_regions.pop(0)
        track_frames.pop(0)

    for i in range(none_counts):
        # We will delete all Nones at final positions.
        track_positions.pop(-1)
        track_regions.pop(-1)
        track_frames.pop(-1)

    assert len(track_positions) == len(track_regions) == len(track_frames)

    return track_positions, track_regions, track_frames


def load_detection(path, index):

    p = os.path.join(path, f"frame_{index}.json")

    img_data = load_from_json(p)
    return img_data

def load_detection_and_turn_to_coco_format(path, index, offset_to_classes = 0):
    img_data = load_detection(path, index)

    print(img_data)
    detection_boxes = img_data['detection_boxes']    
    detection_classes = img_data['detection_classes']
    detection_scores = img_data['detection_scores']

    coco_detection_boxes = []

    for detection_box in detection_boxes:
        if len(detection_box) == 2:
            coco_box = [detection_box[0][1], detection_box[0][0], detection_box[1][1] - detection_box[0][1], detection_box[1][0] - detection_box[0][0]]
        else:
            coco_box = [detection_box[1], detection_box[0], detection_box[3] - detection_box[1], detection_box[2] - detection_box[0]]
        coco_detection_boxes.append(coco_box)

    return [np.array(coco_detection_boxes), np.array(detection_classes) - offset_to_classes,  np.array(detection_scores)]
