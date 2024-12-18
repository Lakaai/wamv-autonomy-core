from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('seg.pt')  # Replace 'yolov8n.pt' with the path to your YOLOv8 model if different

# Load the image
image_path = "code.png"  # Replace with the path to your image
image = cv2.imread(image_path)

# Perform inference
results = model.predict(source=image, save=True)  # 'save=True' saves the results

# Display the results on the image
annotated_image = results[0].plot()  # This draws the bounding boxes and labels on the image

# Show the image with bounding boxes
cv2.imshow('YOLOv8 Inference', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# from roboflow import Roboflow
# # rf = Roboflow(api_key="WU49vbgHUCDAR0DAVn9p")
# # project = rf.workspace().project("matrix-model")
# # model = project.version(1).model


# from roboflow import Roboflow

# # Initialize Roboflow with your API key
# rf = Roboflow(api_key="WU49vbgHUCDAR0DAVn9p")

# # Access your workspace and project
# project = rf.workspace("matrix").project("matrix-model")

# # Specify the model version you want to download
# version = project.version("1")

# # Download the YOLOv8 model in the format you need
# model = version.download("yolov8")

# # # This will download the model files (e.g., 'best.pt') into a directory named "yolov8"

# # # infer on a local image
# print(model.predict("matrix.png", confidence=40, overlap=30).json())

# # # visualize your prediction
# # model.predict("matrix.png", confidence=40, overlap=30).save("prediction.png")

# # # infer on an image hosted elsewhere
# # # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())