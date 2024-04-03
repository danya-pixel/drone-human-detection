import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on an image or video source and save the result."
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the YOLOv8 model file."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image or video file.",
    )

    args = parser.parse_args()

    model = YOLO(args.model_path)

    results = model(args.source, save=True)


if __name__ == "__main__":
    main()
