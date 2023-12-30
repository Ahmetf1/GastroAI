from ultralytics import YOLO
import os
import pathlib

# Load a pretrained YOLOv8n model

weights = pathlib.Path(__file__).resolve().parent.joinpath("results").joinpath("epoch1900.pt").as_posix()
model = YOLO(weights)

# Run inference on 'bus.jpg' with arguments
for media in os.listdir(pathlib.Path(__file__).resolve().parent.joinpath("datasets").joinpath("test")):
    print(media)
    results = model.predict(pathlib.Path(__file__).resolve().parent.joinpath("datasets").joinpath("test").joinpath(media), save=True, stream=True, device='0')
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        print(boxes)