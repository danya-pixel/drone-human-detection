import argparse

from sahi import AutoDetectionModel
from sahi.predict import predict


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference using a detection model."
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image file for inference.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (cpu or cuda).",
    )
    parser.add_argument(
        "--slice_height",
        type=int,
        default=256,
        help="Height of each slice for tiled inference.",
    )
    parser.add_argument(
        "--slice_width",
        type=int,
        default=256,
        help="Width of each slice for tiled inference.",
    )
    parser.add_argument(
        "--overlap_height_ratio",
        type=float,
        default=0.2,
        help="Overlap height ratio for tiled inference.",
    )
    parser.add_argument(
        "--overlap_width_ratio",
        type=float,
        default=0.2,
        help="Overlap width ratio for tiled inference.",
    )
    parser.add_argument(
        "--model_confidence_threshold",
        type=float,
        default=0.4,
        help="Model confidence threshold for detections.",
    )

    args = parser.parse_args()

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )

    result = predict(
        detection_model=detection_model,
        source=args.image_path,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        model_confidence_threshold=args.model_confidence_threshold,
        return_dict=True,
    )

    print(result)


if __name__ == "__main__":
    main()
