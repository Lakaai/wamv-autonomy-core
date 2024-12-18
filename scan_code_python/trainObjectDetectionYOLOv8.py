from ultralytics import YOLO

# Load a model
model = YOLO('data/data.yaml')  # build a new model from scratch
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='data/data.yaml', epochs=3)  # train the model
results = model.val()  # evaluate model performance on the validation set
results = model('matrix.png')  # predict on an image
results = model.export(format='yolov8')  # export the model to ONNX format