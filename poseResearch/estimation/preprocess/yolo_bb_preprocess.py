from .preprocess_estimation import PreprocessEstimation
import torch
from ultralytics import YOLO
import cv2
from typing import Literal


available__yolo_pose_models = Literal[
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
]


class YOLOBoundingBoxPreprocess(PreprocessEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "YOLOBoundingBoxPreprocess"

    def __init__(
        self,
        model: available__yolo_pose_models,
        video_path: str | None = None,
    ):
        super().__init__()
        self.model = YOLO(model)
        self.video_path = video_path

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Masks out everything outside the bounding boxes of people.
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: Output images of shape (T, H, W, C)
        """
        images_in_shape = images.permute(0, 3, 1, 2)  # (T, C, H, W)
        T, C, H, W = images_in_shape.shape
        results = self.model(images_in_shape)
        output = torch.zeros_like(images)  # (T, H, W, C)

        last_valid_mask = None  # Store the last valid mask
        last_valid_mask_t = None  # Store the last valid mask's frame index
        frames_with_detection = 0  # Counter for frames with person detections

        for t, r in enumerate(results):
            # r.boxes: bounding boxes for this image
            boxes = r.boxes
            current_mask = torch.zeros((H, W), dtype=torch.bool, device=images.device)

            if boxes is not None and boxes.shape[0] > 0:
                # Process current frame detections
                for i in range(boxes.shape[0]):
                    cls = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    if cls == 0 and conf > 0.9:  # class 0 is 'person'
                        x1, y1, x2, y2 = boxes.xyxy[i]
                        # Clamp to image bounds and convert to int
                        x1 = max(0, int(torch.floor(x1).item()))
                        y1 = max(0, int(torch.floor(y1).item()))
                        x2 = min(W, int(torch.ceil(x2).item()))
                        y2 = min(H, int(torch.ceil(y2).item()))
                        current_mask[y1:y2, x1:x2] = True

                # If we found valid detections, update last_valid_mask
                if current_mask.any():
                    last_valid_mask = current_mask.clone()
                    frames_with_detection += 1
                    last_valid_mask_t = t

            # Use last_valid_mask if current frame has no detections
            if not current_mask.any() and last_valid_mask is not None:
                current_mask = last_valid_mask
                print(
                    f"Frame {t}: No detections, using mask from frame {last_valid_mask_t}."
                )
            elif not current_mask.any():
                print(f"Frame {t}: No detections and no previous mask available")

            # Apply mask: keep only pixels inside any person bbox, black out others
            img = images[t]  # (H, W, C)
            masked_img = torch.zeros_like(img)
            for c in range(img.shape[2]):
                masked_img[..., c][current_mask] = img[..., c][current_mask]
            output[t] = masked_img

        if self.video_path is not None:
            # Write the output tensor to a video file for visualization
            try:
                # Convert output to uint8 and numpy for video writing
                output_np = output.cpu().numpy()

                # Normalize to 0-255 range if needed
                if output_np.max() <= 1.0:
                    output_np = (output_np * 255).astype("uint8")
                else:
                    output_np = (output_np.clip(0, 255)).astype("uint8")

                # Write to video file using OpenCV
                frames, height, width, _ = output_np.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = 30

                out = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))

                for t in range(frames):
                    frame = output_np[t]
                    # OpenCV expects BGR format, convert from RGB if needed
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()
                print(f"{self.identifier}: Output video written to {self.video_path}")

            except ImportError:
                raise

        # Print detection statistics
        print(
            f"{self.identifier}: Person detected in {frames_with_detection}/{T} of {T} frames ({frames_with_detection/T*100:.1f}%)"
        )

        return output
