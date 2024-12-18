"""
========================================================================================================================================
Script Name:            exportOpenVINO.py
Description:            An example script that loads the default YOLOv8 Model and exports it as an OpenVINO model. 
                        
                         
Author:                 Luke Thompson 
Created Date:           04/08/2024
Last Modified By:       Luke Thompson
Last Modified Date:     04/08/2024
Version:                1.0
Dependencies:           | OpenVINO | Ultralytics |
Usage:                  Exporting a GPU model requires CUDA however you can export the model as CPU and then run inference on an Intel GPU at runtime.
License:                
========================================================================================================================================
"""

from ultralytics import YOLO

device_type = "CPU"                                             # Use either "CPU" or "GPU" 
model = YOLO("models/yolov8n.pt")                               # Load YOLOv8 model
model.export(format='openvino', device=device_type)             # Export YOLOv8 model

