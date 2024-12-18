"""
========================================================================================================================================
Script Name:            detectOpenVINO.py
Description:            This script uses the PySpin API to acquire frames from the Spinnaker Blackfly camera
                        and then performs object detection on those frames using an OpenVINO YOLOv8 model. 
Author:                 Luke Thompson 
Created Date:           04/08/2024
Last Modified By:       Luke Thompson
Last Modified Date:     04/08/2024
Version:                1.0
Dependencies:           | OpenCV | Numpy | PySpin | Ultralytics | OpenVINO
Usage:                  Requires the PySpin API from https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/
                        The Spinnaker SDK must be installed prior to installing PySpin (pretty sure anyways). 
                        Use the appropriate version for your CPU architecture: Spinnaker 4.0.0.116 for Ubuntu 22.04 Python (December 21, 2023)
License:                
========================================================================================================================================
"""

import PySpin
import cv2
import numpy as np
from ultralytics import YOLO
import time

def run_object_detection(camera, model):
    cv2.namedWindow("YOLOv8 Detection", flags=cv2.WINDOW_GUI_NORMAL)
    #cv2.resizeWindow("YOLOv8 Detection", 800, 600)
    prev_time = 0  # Initialize previous time for FPS calculation

    try:
        while True:
            # Capture frame from camera
            frame = camera.GetNextImage()
            
            if frame is None:
                print("Source ended")
                break
            frame_data = frame.GetNDArray()
            #frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BAYER_RG2BGR)
            frame.Release()

            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BAYER_RG2RGB)
            # Perform object detection
            results = model.predict(frame_data)

            # Annotate frame with results
            annotated_frame = results[0].plot()

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Draw FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame with detections
            #cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
            #cv2.setWindowProperty("YOLOv8 Detection", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break
            
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # Cleanup
        camera.EndAcquisition()
        camera.DeInit()
        del camera
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()

# Initialize Spinnaker camera
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
camera = cam_list[0]
camera.Init()


# # Get the NodeMap to access camera settings
# nodemap = camera.GetNodeMap()

# # Set the width and height
# width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
# height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))

# Set desired frame size
# width_node.SetValue(3072)
# height_node.SetValue(1686)

# width_node.SetValue(1920)
# height_node.SetValue(1080)

camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
camera.BeginAcquisition()


# Load YOLOv8 model
#model = YOLO("models/yolov8n.pt") 
#model.export(format='openvino')

#model = YOLO("yolov8n-seg.pt")
model = YOLO("best.pt")

run_object_detection(camera, model)
