import comet_ml
import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
    ultralytics.checks()
    comet_ml.init(project_name="human-detection")

    model = YOLO("yolov8n.pt")
    model.train(
        project="human-detection",
        data="datasets/cfg/VisDrone.yaml",
        epochs=256,
        single_cls=True,
        save_period=5,
        batch=16,
        imgsz=840,
        name="yolov8_840px",
    )
