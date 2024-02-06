import os
import sys
import math
import numpy as np
import cv2
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from Config import Config
import Region_Selector
import general_utils
import test_time_augmentation

# Root directory of yolov3-keras-tf2 project
ROOT_DIR = Config.YOLOV5_PATH
sys.path.append(ROOT_DIR)  # To find local version of the library

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


"""
     Class to perform object detection using YOLO v5..
"""

class Yolov5_Region_Selector(Region_Selector.Region_Selector):
    detector = None     # Yolo version 5 detector.
    device = None       # Device to run inference at.
    
    def __init__(self):
        with torch.no_grad():
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
            parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
            parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default='runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
            self.opt = parser.parse_args()
            print(self.opt)
            #check_requirements()
        
            source, weights, view_img, save_txt, self.imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

            # Directories
            #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
            #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Initialize
            set_logging()
            self.device = select_device(self.opt.device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
            
            if self.half:
                self.model.half()  # to FP16

            # Second-stage classifier
            self.classify = False
            if self.classify:
                self.modelc = load_classifier(name='resnet101', n=2)  # initialize
                self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        
    #Method to apply preprocess to a np matrix in order to be processed by the Network.
    def preprocess_image(self,img0):
        assert img0 is not None, 'Image Not Found'
        # Padded resize
        img = letterbox(img0, new_shape=self.imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        return img
        
        
    def run_inference_for_single_image(self, img0):
        with torch.no_grad():
    
            self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
            
            # Run inference
            t0 = time.time()
            img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
            _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
            
            img = self.preprocess_image(img0)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, img0)

            output_dict = {}
            output_dict["detection_boxes"] = []
            output_dict["detection_classes"] = []
            output_dict["detection_scores"] = []
            
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                s = ''

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {self.names[int(c)]}s, '  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        
                        x1 = int(xyxy[0])
                        x2 = int(xyxy[2])
                        y1 = int(xyxy[1])
                        y2 = int(xyxy[3])
                        detection_box = [y1, x1, y2, x2]
                        output_dict["detection_boxes"].append(detection_box)
                        output_dict["detection_classes"].append(int(cls))
                        output_dict["detection_scores"].append(float(conf))

                        """
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        """
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                
                """
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)
                """
                
            return output_dict
