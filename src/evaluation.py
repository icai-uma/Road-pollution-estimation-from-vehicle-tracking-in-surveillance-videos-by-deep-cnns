import matplotlib.pyplot as plt
import os
import numpy as np
import general_utils

from Config import Config

VIDEOS_TO_TEST = ["sherbrooke_video"]
#VIDEOS_TO_TEST = ["stmarc_video"]

REFERENCE = "gt_annotated"
MODELS_TO_TEST = ["yolov5", "faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8"]
MODELS_SHORT_NAMES = {"yolov5":"yolov5", "faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8": "fasterv2"}
MODELS_BACKEND = {"yolov5":"torch", "faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8":"tensorflow", "gt_annotated":"none"}
SPEED_ESTIMATION_METHODS = ["polynomial_average", "optical_flow", "simple_average"]
SPEED_ESTIMATION_METHODS_NAMES = {"polynomial_average":"Piecewise Linear Approximation", "simple_average": "Linear Approximation", "optical_flow": "Optical Flow"}
MAIN_SPEED_ESTIMATION_METHOD = "polynomial_average"
OUTPUT_FOLDER = f"{Config.MAIN_OUTPUT_FOLDER}/evaluation/"

def plot_speeds(speed_matrix, fig_name):
    fig = plt.figure(fig_name)
    print(speed_matrix.shape)
    for h in range(speed_matrix.shape[0]):
        s = speed_matrix[h,:]
        
    plt.plot(range(speed_matrix.shape[1]), s, color = "red")
    
def plot_over_two_axis(x_data, x_label, left_y_data, left_y_label, right_y_data, right_y_label, figure_name):
    
    fig = plt.figure(figure_name)

    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(left_y_label, color=color)
    ax1.plot(x_data, left_y_data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(right_y_label, color=color)  # we already handled the x-label with ax1
    ax2.plot(x_data, right_y_data, color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.yticks(range(max(right_y_data)+1))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
def get_values_greater_than_n(vector, n):
    values_greater_than_n = []
    for element in vector:
        if element > n: values_greater_than_n.append(element)
        
    return np.array(values_greater_than_n)        

if not os.path.isdir(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for video_name in VIDEOS_TO_TEST:
    print(video_name)
    print("Reference")
    reference_record_path = f"{Config.MAIN_OUTPUT_FOLDER}/{video_name}/{MODELS_BACKEND[REFERENCE]}/{REFERENCE}/{MAIN_SPEED_ESTIMATION_METHOD}/speed_records.npy"

    reference_speed_np_matrix = np.load(reference_record_path)
    reference_speed_np_matrix = reference_speed_np_matrix*3.6 # From m/s to km/h
    
    reference_pollution_by_vehicle_np_matrix = general_utils.from_speed_to_pollution_matrix(reference_speed_np_matrix)

    reference_pollution_by_frame = np.sum(reference_pollution_by_vehicle_np_matrix, axis=0)

    mooving_vehicles_per_frame = np.sum(reference_speed_np_matrix > 1, axis=0)    

    plot_over_two_axis(range(reference_pollution_by_frame.shape[0]), 'frame', reference_pollution_by_frame, 'Pollution (g/frame)', mooving_vehicles_per_frame, "Moving Vehicles", f"Reference Pollution by frame.")
    
    plt.savefig(f"{OUTPUT_FOLDER}/{video_name}_gt_pollution.png", dpi=1000)
    
    average_polllution_by_mooving_vehicle = np.mean(np.sum(reference_pollution_by_vehicle_np_matrix, axis=1))
    std_average_polllution_by_mooving_vehicle = np.std(np.sum(reference_pollution_by_vehicle_np_matrix, axis=1))
        
    reference_acumulated_pollution = []
    for i in range(reference_pollution_by_frame.shape[0]):
        reference_acumulated_pollution.append(np.sum(reference_pollution_by_frame[:i+1]))

    reference_acumulated_pollution = np.array(reference_acumulated_pollution)
    
    print(f"Total number of mooving objects detected:{reference_speed_np_matrix.shape[0]}")
    print(f"Average pollution by vehicle:{average_polllution_by_mooving_vehicle}")
    print(f"std pollution by vehicle:{std_average_polllution_by_mooving_vehicle}")
    
    average_speeds = []
    for vector_index in range(reference_speed_np_matrix.shape[0]):
        filtered_vector = get_values_greater_than_n(reference_speed_np_matrix[vector_index],0)
        if filtered_vector.shape[0]>0:
            average_speeds.append(np.mean(filtered_vector))
        
    average_speeds = np.array(average_speeds)
    
    print(f"Average speed by vehicle:{np.mean(average_speeds)}")
    print(f"Std speed by vehicle:{np.std(average_speeds)}")
    print(f"Total pollution:{reference_acumulated_pollution[-1]}")

    for model_name in MODELS_TO_TEST:
        print(model_name)
        model_record_path = f"{Config.MAIN_OUTPUT_FOLDER}/{video_name}/{MODELS_BACKEND[model_name]}/{model_name}/{MAIN_SPEED_ESTIMATION_METHOD}/speed_records.npy"
        model_speed_np_matrix = np.load(model_record_path)
        
        
        model_speed_np_matrix = model_speed_np_matrix*3.6 # From m/s to km/h

        model_pollution_by_vehicle_np_matrix = general_utils.from_speed_to_pollution_matrix(model_speed_np_matrix)

        model_pollution_by_frame = np.sum(model_pollution_by_vehicle_np_matrix, axis=0)
        
        model_absolute_error_by_frame = np.abs(reference_pollution_by_frame-model_pollution_by_frame)

        fig = plt.figure(f"Absolute error by frame.")
        plt.plot(range(model_absolute_error_by_frame.shape[0]), model_absolute_error_by_frame, label = MODELS_SHORT_NAMES[model_name])

        model_acumulated_pollution_by_frame = []
        for i in range(model_pollution_by_frame.shape[0]):
            model_acumulated_pollution_by_frame.append(np.sum(model_pollution_by_frame[:i+1]))

        model_acumulated_pollution_by_frame = np.array(model_acumulated_pollution_by_frame)

        fig = plt.figure("Accumulated Pollution by frame")
        plt.plot(range(model_acumulated_pollution_by_frame.shape[0]), model_acumulated_pollution_by_frame, label = MODELS_SHORT_NAMES[model_name])

        difference = np.abs(np.subtract(reference_acumulated_pollution, model_acumulated_pollution_by_frame))

        fig = plt.figure(f"Accumulated Pollution error by frame")
        plt.plot(range(difference.shape[0]), difference, label = MODELS_SHORT_NAMES[model_name])
        
        mooving_vehicles_per_frame = np.sum(model_speed_np_matrix > 0, axis=0)
        
        average_polllution_by_mooving_vehicle = np.mean(np.sum(model_pollution_by_vehicle_np_matrix, axis=1))
        std_average_polllution_by_mooving_vehicle = np.std(np.sum(model_pollution_by_vehicle_np_matrix, axis=1))

        print(f"Total number of mooving objects detected:{np.sum(np.sum(model_speed_np_matrix, axis=1)>0)}")
        print(f"Average pollution by vehicle:{average_polllution_by_mooving_vehicle}")
        print(f"std pollution by vehicle:{std_average_polllution_by_mooving_vehicle}")
        average_speeds = []
        for vector_index in range(model_speed_np_matrix.shape[0]):
            filtered_vector = get_values_greater_than_n(model_speed_np_matrix[vector_index],0)
            if filtered_vector.shape[0]>0:
                average_speeds.append(np.mean(filtered_vector))
            
        average_speeds = np.array(average_speeds)
        
        print(f"Average speed by vehicle:{np.mean(average_speeds)}")
        print(f"Std speed by vehicle:{np.std(average_speeds)}")
        print(f"Total pollution:{model_acumulated_pollution_by_frame[-1]}")
        print(f"% error pollution:{abs(reference_acumulated_pollution[-1]-model_acumulated_pollution_by_frame[-1])/reference_acumulated_pollution[-1]}")
        

        plot_over_two_axis(range(model_pollution_by_frame.shape[0]), 'frame', model_pollution_by_frame, 'Pollution', mooving_vehicles_per_frame, "Moving Vehicles", f"{MODELS_SHORT_NAMES[model_name]} Pollution by frame.")
    
        plt.savefig(f"{OUTPUT_FOLDER}/{video_name}_{MODELS_SHORT_NAMES[model_name]}_pollution.png", dpi=1000)

    fig = plt.figure("Accumulated Pollution error by frame")
    plt.xlabel('frame')
    plt.ylabel('Accumulated $PM_{10}$ emissions error')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{video_name}_accumulated_pollution_error.png", dpi=1000)
    
    fig = plt.figure(f"Absolute error by frame.")
    plt.xlabel('frame')
    plt.ylabel('Absolute $PM_{10}$ emissions error')
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"{OUTPUT_FOLDER}/{video_name}_absolute_pollution_error.png", dpi=1000)
    
    fig = plt.figure(f"Accumulated Pollution by frame")
    plt.plot(range(reference_acumulated_pollution.shape[0]), reference_acumulated_pollution, label = "manual")

    fig = plt.figure("Accumulated Pollution by frame")
    plt.xlabel('frame')
    plt.ylabel('Accumulated $PM_{10}$ emissions')
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"{OUTPUT_FOLDER}/{video_name}_accumulated_pollution.png", dpi=1000)
        
    plt.show()

    for speed_estimation_method in SPEED_ESTIMATION_METHODS:
        print(speed_estimation_method)
        speed_record_path = f"{Config.MAIN_OUTPUT_FOLDER}/{video_name}/{MODELS_BACKEND[REFERENCE]}/{REFERENCE}/{speed_estimation_method}/speed_records.npy"
        speed_np_matrix = np.load(speed_record_path)
        
        speed_np_matrix = speed_np_matrix*3.6 # From m/s to km/h

        speed_pollution_by_vehicle_np_matrix = general_utils.from_speed_to_pollution_matrix(speed_np_matrix)

        speed_pollution_by_frame = np.sum(speed_pollution_by_vehicle_np_matrix, axis=0)
        
        speed_acumulated_pollution_by_frame = []
        for i in range(speed_pollution_by_frame.shape[0]):
            speed_acumulated_pollution_by_frame.append(np.sum(speed_pollution_by_frame[:i+1]))

        speed_acumulated_pollution_by_frame = np.array(speed_acumulated_pollution_by_frame)

        fig = plt.figure("Accumulated Pollution by frame")
        plt.plot(range(speed_acumulated_pollution_by_frame.shape[0]), speed_acumulated_pollution_by_frame, label = SPEED_ESTIMATION_METHODS_NAMES[speed_estimation_method])
    
    fig = plt.figure("Accumulated Pollution by frame")
    plt.xlabel('frame')
    plt.ylabel('Accumulated $PM_{10}$ emissions')
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"{OUTPUT_FOLDER}/{video_name}_speed_methods_accumulated_pollution.png", dpi=1000)
  
diesel = []
petrol = []
mix = []
for i in range(1,120):
    diesel.append(general_utils.from_speed_to_pollution_km(i, fuel="diesel"))
    petrol.append(general_utils.from_speed_to_pollution_km(i, fuel="petrol"))
    mix.append(general_utils.from_speed_to_pollution_km(i, fuel="mix"))
fig = plt.figure("Emissions km/h")

plt.plot(range(1,120), diesel, label = "diesel")
plt.plot(range(1,120), petrol, label = "petrol")
plt.plot(range(1,120), mix, label = "mix")

plt.xlabel('speed (km/h)')
plt.ylabel('$PM_{10}$ emissions (g/km)')
plt.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f"{OUTPUT_FOLDER}/emissions_g_km.png", dpi=1000)


diesel = []
petrol = []
mix = []
for i in range(1,30):
    diesel.append(general_utils.from_speed_to_pollution_per_frame(3.6*i, fuel="diesel"))
    petrol.append(general_utils.from_speed_to_pollution_per_frame(3.6*i, fuel="petrol"))
    mix.append(general_utils.from_speed_to_pollution_per_frame(3.6*i, fuel="mix"))
fig = plt.figure("Emissions kpf")

plt.plot(np.array(range(1,30)), diesel, label = "diesel")
plt.plot(np.array(range(1,30)), petrol, label = "petrol")
plt.plot(np.array(range(1,30)), mix, label = "mix")

plt.xlabel('speed (meters per second)')
plt.ylabel('$PM_{10}$ emissions (g/frame)')
plt.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f"{OUTPUT_FOLDER}/emissions_g_f.png", dpi=1000)
