import ultralytics
from ultralytics import YOLO
import comet_ml



if __name__ == "__main__":
    ultralytics.checks()
    comet_ml.init(project_name="human-detection")

    model = YOLO("yolov8n.pt")
    results = model.train(project='human-detection', data='datasets/cfg/VisDrone.yaml', epochs=300, imgsz=640, single_cls=True, save_period=1, save_json=True, batch=32)

    print(results)

