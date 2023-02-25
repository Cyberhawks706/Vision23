# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import ntcore
from ntcore import NetworkTableInstance
from cscore import *
import numpy as np
import json
import time
from multiprocessing import Pool,Process
import robotpy_apriltag
from robotpy_apriltag import *
import cv2

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        pi_num=1
):
    with open('/boot/frc.json') as f:
        cameraConfig = json.load(f)
    camera = cameraConfig['cameras'][0]
    pi_num = str(pi_num)
    ntinst = NetworkTableInstance.getDefault()
    ntinst.startClient4(identity=("wpilibpi" + pi_num))
    ntinst.startDSClient()
    ntinst.setServerTeam(team=706)
    ntinst.getTable("CameraPublisher").getSubTable("rawCam" + pi_num).getEntry("streams").setStringArray(["mjpeg:http://wpilibpi" + pi_num + ".local:1181/?action=stream"])
    width = camera['width']
    height = camera['height']

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    
    usbCam = CameraServer.startAutomaticCapture(name=("rawCam" + pi_num), path="/dev/video0")
    usbCam.setResolution(width, height)
    input_stream = CameraServer.getVideo(camera=usbCam)
    output_stream = CameraServer.putVideo("Processed" + pi_num, width, height)
    source = "http://wpilibpi" + pi_num + ".local:1181/stream.mjpg"
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    time.sleep(0.5)
    black_img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    detector = AprilTagDetector()
    detector.addFamily("tag16h5")
    detectorConfig = detector.getConfig()
    quad_params = detector.getQuadThresholdParameters()
    detectorConfig.numThreads = 4
    quad_params.maxLineFitMSE = 3
    quad_params.minWhiteBlackDiff = 70
    quad_params.criticalAngle = 20
    detector.setConfig(detectorConfig)
    detector.setQuadThresholdParameters(quad_params)
    estimatorConfig = AprilTagPoseEstimator.Config(tagSize=0.1524, fx=1050, fy=1020, cx=width/2, cy=height/2)
    poseEstimator = AprilTagPoseEstimator(estimatorConfig)

    # Run inference
    model.warmup(imgsz=(len(dataset), 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:


        _, input_img = input_stream.grabFrame(black_img)
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray_img)
        corners = [[0,0],[0,0],[0,0],[0,0]]
        
        

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
                # batch_size >= 1
            im0 = im0s[i].copy()
            s += f'{i}: '

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                counter = 0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    c = int(cls)
                    cls = str(c)
                    xCenter = (int(xyxy[0]) + int(xyxy[2])) / 2
                    yCenter = (int(xyxy[1]) + int(xyxy[3])) / 2
                    print("cls: " + cls + " count: " + str(counter))
                    print("xCenter: " + str(xCenter))
                    print("yCenter: " + str(yCenter))
                    cv2.circle(im0, (int(xCenter), int(yCenter)), 3, (0,255,0), 5)
                    ntinst.getTable("SmartDashboard").getSubTable("processed" + pi_num).getSubTable(cls + str(counter)).putValue("xCenter", ((int(xyxy[0]) + int(xyxy[2])) / 2))
                    ntinst.getTable("SmartDashboard").getSubTable("processed" + pi_num).getSubTable(cls + str(counter)).putValue("yCenter", ((int(xyxy[1]) + int(xyxy[3])) / 2))
                    # Add bbox to image
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    counter += 1

            # Stream results
            im0 = annotator.result()
            if(detections):
                for k in range(len(detections)):
                    for i in range(len(corners)):
                        corners[i][0] = detections[k].getCorner(i).x
                        corners[i][1] = detections[k].getCorner(i).y
                    center = (int(corners[0][0]), int(corners[0][1]))
                    poseEstimate = poseEstimator.estimate(detections[k])
                    cv2.polylines(im0, np.int32([corners]), True, (255,0,0), 5)
                    cv2.putText(im0, str(detections[k].getId()), center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                    ntinst.getTable("SmartDashboard").getSubTable("processed" + pi_num).putValue("tag" + str(detections[k].getId()), str(poseEstimate))
                    print("tag" + str(detections[k].getId()) + ": ", str(poseEstimate))
            output_stream.putFrame(im0)
            for tagID in range(32):
                table = ntinst.getTable("SmartDashboard").getSubTable("processed" + pi_num)
                if (table.containsKey("tag" + str(tagID))) and (ntcore._now() - table.getEntry("tag" + str(tagID)).getLastChange()) > 50000:
                    table.getEntry("tag" + str(tagID)).unpublish()

            
        # put video back to processed0

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--pi-num', type=int, default=2)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

