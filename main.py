from ultralytics import YOLO

# Loads a model pretrained on the COCO dataset
model = YOLO('yolov8n.pt')

# Tracks and shows boxes around objects found in the COCO dataset in a video
results = model.track(source="https://www.youtube.com/watch?v=aQwajwY2gp4", show=True, show_conf=False)