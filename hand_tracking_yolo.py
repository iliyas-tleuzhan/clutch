import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(source_value: str):
    """Use camera index when numeric; otherwise treat as file/URL."""
    if source_value.isdigit():
        return int(source_value)
    return source_value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO hand detection + tracking")
    parser.add_argument(
        "--model",
        type=str,
        default="hand.pt",
        help="Path to hand-detection YOLO weights (e.g., hand.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: webcam index (0) or video file path/URL.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config: bytetrack.yaml or botsort.yaml",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live annotated frames in a window.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output video to ./outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Torch device, e.g. "0" (GPU), "cpu", or "0,1".',
    )
    return parser


def main():
    args = build_argparser().parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Download a hand-detection YOLO checkpoint and pass --model /path/to/weights.pt"
        )

    source = parse_source(args.source)
    model = YOLO(str(model_path))

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "hand_tracking.mp4"
    writer = None
    last_frame_time = time.time()

    track_stream = model.track(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tracker=args.tracker,
        persist=True,
        stream=True,
        verbose=False,
        device=args.device,
    )

    for result in track_stream:
        frame = result.plot()

        now = time.time()
        fps = 1.0 / max(1e-6, now - last_frame_time)
        last_frame_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (w, h),
                )
            writer.write(frame)

        if args.show:
            cv2.imshow("YOLO Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if args.save:
        print(f"Saved: {output_video_path.resolve()}")


if __name__ == "__main__":
    main()
