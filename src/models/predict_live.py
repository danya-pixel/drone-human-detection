import argparse

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Perform YOLOv8 inference on a video.")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the YOLOv8 model file."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video file."
    )

    args = parser.parse_args()

    model = YOLO(args.model_path)

    cap = cv2.VideoCapture(args.video_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.predict(frame, conf=0.5)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
