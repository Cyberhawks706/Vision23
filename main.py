import ntcore
from ntcore import NetworkTableInstance
from cscore import CameraServer
import robotpy_apriltag
from robotpy_apriltag import *

import cv2
import json
import numpy as np
import time

def main():
   with open('/boot/frc.json') as f:
      detectorConfig = json.load(f)
   camera = detectorConfig['cameras'][0]
   numCameras = len(detectorConfig['cameras'])

   width = camera['width']
   height = camera['height']

   #start networktables
   #inst = CameraServer.getInstance()
   ntinst = NetworkTableInstance.getDefault()
   ntinst.startClient4(identity="wpilibpi")
   ntinst.startDSClient()
   ntinst.setServerTeam(team=706)

   #force CameraServer to start looking for cameras
   ntinst.getTable("CameraPublisher").getSubTable("rawCam0").getEntry("streams").setStringArray(["mjpeg:http://wpilibpi.local:1181/?action=stream"])

   usbCams = {}
   input_streams = {}
   output_streams = {}

   for camID in range(numCameras):
      usbCams[camID] = CameraServer.startAutomaticCapture(name=("rawCam" + str(camID)), path=("/dev/video" + str(camID * 2)))
      usbCams[camID].setResolution(width, height)
      input_streams[camID] = CameraServer.getVideo(camera=usbCams[camID])
      output_streams[camID] = CameraServer.putVideo(("Processed" + str(camID)), width, height)
   
   # Table for vision output information
   vision_nt = ntinst.getTable('Vision')

   # Allocating new images is very expensive, always try to preallocate
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
   # Wait for NetworkTables to start
   time.sleep(0.5)

   while True:
      start_time = time.time()

      input_imgs = {}
      output_imgs = {}
      frame_times = {}

      for camID in range(numCameras):
         frame_times[camID], input_imgs[camID] = input_streams[camID].grabFrame(black_img)
         output_imgs[camID] = np.copy(input_imgs[camID])
      
      if frame_times[0] == 0:
         output_streams[0].notifyError(input_streams[0].getError())
         continue

      for camID in range(numCameras):
         # Convert to HSV and threshold image
         gray_img = cv2.cvtColor(input_imgs[camID], cv2.COLOR_BGR2GRAY)
         
         detections = detector.detect(gray_img)
         corners = [[0,0],[0,0],[0,0],[0,0]]
         if(detections):
            for k in range(len(detections)):
               for i in range(len(corners)):
                  corners[i][0] = detections[k].getCorner(i).x
                  corners[i][1] = detections[k].getCorner(i).y
               center = (int(corners[0][0]), int(corners[0][1]))
               poseEstimate = poseEstimator.estimate(detections[k])
               cv2.polylines(output_imgs[camID], np.int32([corners]), True, (255,0,0), 5)
               cv2.putText(output_imgs[camID], str(detections[k].getId()), center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
               ntinst.getTable("SmartDashboard").getSubTable("processed" + str(camID)).putValue("tag" + str(detections[k].getId()), str(poseEstimate))
         
         for tagID in range(32):
            table = ntinst.getTable("SmartDashboard").getSubTable("processed" + str(camID))
            if (table.containsKey("tag" + str(tagID))) and (ntcore._now() - table.getEntry("tag" + str(tagID)).getLastChange()) > 50000:
               table.getEntry("tag" + str(tagID)).unpublish()

      processing_time = time.time() - start_time
      fps = 1 / processing_time
      for camID in range(numCameras):
         cv2.putText(output_imgs[camID], str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
         output_streams[camID].putFrame(output_imgs[camID])

main()