from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Change device = mps in win or linux
results = model.train(data='dataset.yaml', epochs=100, patience = 10, save = True, save_period = 10, imgsz=512, device='mps')