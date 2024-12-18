from ultralytics import YOLO
import cv2
import time
from openvino.runtime import Core, get_version

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")

# Export the model to OpenVINO format
model.export(format="openvino")

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize OpenVINO Core
core = Core()

# Print OpenVINO version
print(f"OpenVINO version: {get_version()}")

# Get available devices
devices = core.available_devices

print(f"Available devices: {devices}")

# Get device properties
device_properties = core.get_property('GPU', "FULL_DEVICE_NAME")
print(f"Device properties: {device_properties}")

# Print the device being used
print(f"Using device: {model.device}")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    start_time = time.time()
    results = ov_model(frame)
    end_time = time.time()

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Calculate and display FPS
    fps = 1 / (end_time - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()