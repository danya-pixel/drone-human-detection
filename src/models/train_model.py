import argparse

import ultralytics
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model with specified imgsz and batch."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size for training."
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size for training."
    )
    args = parser.parse_args()

    ultralytics.checks()

    model = YOLO("yolov8n.pt")

    model.train(
        project="human-detection",
        data="datasets/cfg/VisDrone.yaml",
        epochs=256,
        single_cls=True,
        save_period=5,
        batch=args.batch,
        imgsz=args.imgsz,
        name="yolov8_1050px",
    )


if __name__ == "__main__":
    main()
