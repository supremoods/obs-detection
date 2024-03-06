import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os

from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import pyttsx3
from utils.vi_hat import main_req
import requests
import threading

os.environ['OPENBLAS_NUM_THREADS'] = '1'

class Detect:
    def __init__(self):

        self.opt = argparse.Namespace(
            weights='weights/v5lite-s.pt',
            source='sample',
            img_size=640,
            conf_thres=0.45,
            iou_thres=0.2,
            device='',
            view_img=False,
            save_txt=False,
            save_conf=False,
            nosave=False,
            classes=None,
            agnostic_nms=False,
            augment=False,
            update=False,
            project='runs/detect',
            name='exp',
            exist_ok=False,
            read=False,
        )
        self.width_in_rf = 0
        self.label = ''
        self.KNOWN_DISTANCE = 25.0
        self.PERSON_WIDTH = 40
        self.DOG_WIDTH = 55
        self.CAR_WIDTH = 10
        self.CAT_WIDTH = 30
        self.BICYCLE_WIDTH = 55
        self.BENCH_WIDTH = 100
        self.CONFIDENCE_THRESHOLD = 0.4
        self.NMS_THRESHOLD = 0.3
        self.distance = 0

    def focalLength(self, width_in_rf, width):
        focal_length = (width_in_rf * self.KNOWN_DISTANCE) / width
        return focal_length

    def distanceEstimate(self, focal_length, width_in_rf):
        distance = (focal_length * self.KNOWN_DISTANCE) / width_in_rf
        
        distance = distance / 100
        
        return distance

    def detect(self, save_img=False):
        engine = pyttsx3.init()
        voice = engine.getProperty('voices')  # get the available voices
        engine.setProperty('voice', voice[1].id)
        
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        save_img = not self.opt.nosave and not source.endswith(
            '.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(self.opt.project) /
                        self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                       classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + \
                    ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string

                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    detected_classes = []
                    detected_distance = []
                    detected_area = []
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                                1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # label format
                            line = (
                                cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() %
                                        line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # get the width and height of the bounding box
                            self.width_in_rf = xyxy[2] - xyxy[0]
                            # Get the x-coordinate of the center of the bounding box
                            bbox_center = (xyxy[0] + xyxy[2]) / 2
                            # Get the x-coordinate of the center of the image
                            image_center = im0.shape[1] / 2

                            # Determine if the bounding box is on the left, center, or right of the image
                            if bbox_center < image_center - 50:
                                detected_area.append("left")
                            elif bbox_center > image_center + 50:
                                detected_area.append("right")
                            else:
                                detected_area.append("center")

                            self.label = f'{names[int(cls)]} {int(cls)}'

                            if (self.opt.read == False):

                                if names[int(cls)] == 'person':
                                    self.distance = self.distanceEstimate(focal_person, self.width_in_rf)
                                elif names[int(cls)] == 'dog':
                                    self.distance = self.distanceEstimate(focal_dog, self.width_in_rf)
                                elif names[int(cls)] == 'cat':
                                    self.distance = self.distanceEstimate(focal_cat, self.width_in_rf)
                                elif names[int(cls)] == 'car':
                                    self.distance = self.distanceEstimate(focal_car, self.width_in_rf)
                                elif names[int(cls)] == 'bicycle':
                                    self.distance = self.distanceEstimate(focal_bicycle, self.width_in_rf)
                                    
                                if self.distance < 4:
                                    # set colors to red
                                    if self.distance < 2:
                                        detected_classes.append(names[int(cls)])
                                        detected_distance.append(self.distance)

                                        label = f'{names[int(cls)]} {conf:.2f} {self.distance:.2f} m'
                                        colors[int(cls)] = [0, 0, 255]
                                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                    else:
                                        detected_classes.append(names[int(cls)])
                                        detected_distance.append(self.distance)
                                        label = f'{names[int(cls)]} {conf:.2f} {self.distance:.2f} m'
                                        colors[int(cls)] = [0, 255, 0]
                                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    # Construct detected_string
                    distance_strings = [f"is too close to you" if distance < 2 else f"{round(float(distance), 1)} feet away" for distance in detected_distance]
                    position_strings = ["on your left" if detected_area[i] == 'left' else "in front of you" if detected_area[i] == 'center' else "on your right" for i in range(len(detected_area))]

                    detected_string = ", ".join([f"{clazz} {distance_strings[i]} {position_strings[i]}" for i, clazz in enumerate(detected_classes)])
                    
                    # Construct speech output
                    if len(detected_string) > 0: 
                        for position in detected_area:
                            if position == "left":
                                print("Left")
                            elif position == "right":
                                print("Right")
                            else:
                                print("Center")
                        if len(detected_classes) > 0:
                            speech = f"I detected! A {detected_string}"
                            print(f'Speech: {speech}')
                            # Start a new thread to run the speak_warning function                    
                            tts_thread = threading.Thread(target=self.speak_warning, args=(speech,engine))
                            tts_thread.start()
                    
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                key = cv2.waitKey(1)
                if key == ord('q'):
                    engine.stop()
                    break

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        
    def speak_warning(self, str, engine):
        if not engine._inLoop:
            engine.say(str)
            engine.runAndWait()

    def config(self, weights, source, classes, read, view_img):
        self.opt.weights = weights
        self.opt.source = source
        self.opt.classes = classes
        self.opt.read = read
        self.opt.view_img = view_img
        
    def read_focal(self, model, src, classes, width):

        self.config(model, src, classes, True, False)

        self.detect()

        focal_length = self.focalLength(self.width_in_rf, width)

        return self.width_in_rf, focal_length

focal_person = 0
focal_dog = 0
focal_cat = 0
focal_car = 0
focal_bicycle = 0

def inference():
    global focal_person, focal_dog, focal_cat, focal_car, focal_bicycle
    
    obs = Detect()
    
    person_width_px, focal_person = obs.read_focal('weights/v5lite-g.pt', 'ref/50.jpg', 0, obs.PERSON_WIDTH)
    dog_width_px, focal_dog = obs.read_focal('weights/v5lite-g.pt', 'ref/dog50.jpg', 16, obs.DOG_WIDTH)
    cat_width_px, focal_cat = obs.read_focal('weights/v5lite-g.pt', 'ref/dog50.jpg', 16, obs.CAT_WIDTH)
    car_width_px, focal_car = obs.read_focal('weights/v5lite-g.pt', 'ref/car.jpg', 2, obs.CAR_WIDTH)
    bicycle_width_px, focal_bicycle = obs.read_focal('weights/v5lite-g.pt', 'ref/bicycle.jpg', 1, obs.BICYCLE_WIDTH)
    
    print(f'Person focal length: {focal_person}')
    print(f'Dog focal length: {focal_dog}')
    
    print(f'Person width: {person_width_px}')
    print(f'Dog width: {dog_width_px}')
    
    print(f'Cat focal length: {focal_cat}')
    print(f'Cat width: {cat_width_px}')
    
    print(f'Car focal length: {focal_car}')
    print(f'Car width: {car_width_px}')
    
    print(f'Bicycle focal length: {focal_bicycle}')
    print(f'Bicycle width: {bicycle_width_px}')
    
    obs.config('weights/v5lite-s.pt', '0', [0,1,2,13,15,16], False, True)

    obs.detect()


if __name__ == "__main__":
    inference()
