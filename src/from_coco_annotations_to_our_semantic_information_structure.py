import json
import os
from general_utils import load_from_json, save_to_json
from Config import Config


def from_class_id_to_coco_class_id(input_id, offset = 0):
    if input_id == 1:
        return 2 + offset
    if input_id == 2:
        return 5 + offset
    if input_id == 3:
        return 7 + offset
    if input_id == 4:
        return 3 + offset

COCO_ANNOTATIONS_FILE_PATH = f"../input/{Config.PREFFIX}/{Config.PREFFIX}_{Config.VIDEO_INDEX}/annotated_coco/annotations/instances_default.json"

OUTPUT_FOLDER = f"../input/{Config.PREFFIX}/{Config.PREFFIX}_{Config.VIDEO_INDEX}/gt_annotated"

if not os.path.isdir(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

data = load_from_json(COCO_ANNOTATIONS_FILE_PATH)
categories_dict = data["categories"]
print(data["categories"])
for ann in data["annotations"]:
    class_id = ann["category_id"]
    rois = ann["bbox"]
    image_id = ann["image_id"]
    
    frame_number = image_id-1
    
    file_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_number}.json")
    try:
        f = open(file_path)
        previous_saved_data_in_frame = json.load(f)
    except FileNotFoundError:
        print("File not accessible, we create new dict")
        previous_saved_data_in_frame = {"detection_boxes":[], "detection_classes":[], "detection_scores":[]}
        
    previous_detection_boxes = previous_saved_data_in_frame["detection_boxes"]
    previous_detection_classes = previous_saved_data_in_frame["detection_classes"]
    previous_detection_scores = previous_saved_data_in_frame["detection_scores"]
   
    # We add the information to the frame.
   
    previous_detection_boxes.append([rois[1], rois[0], rois[1]+rois[3], rois[0]+rois[2]])
    previous_detection_classes.append(from_class_id_to_coco_class_id(class_id))
    previous_detection_scores.append(1)
   
    data_in_frame = {"detection_boxes":previous_detection_boxes, "detection_classes":previous_detection_classes, "detection_scores":previous_detection_scores}
   
    save_to_json(data_in_frame, file_path)
