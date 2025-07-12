from .preprocess_estimation import PreprocessEstimation
import torch
from ultralytics import YOLO
import cv2


class YOLOBoundingBoxPreprocess(PreprocessEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "YOLOBoundingBoxPreprocess"

    def __init__(self, model_path: str):
        super().__init__()
        self.model = YOLO(model_path)

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

            # Use last_valid_mask if current frame has no detections
            if not current_mask.any() and last_valid_mask is not None:
                current_mask = last_valid_mask
                print(f"Frame {t}: Using mask from previous frame (no detections)")
            elif not current_mask.any():
                print(f"Frame {t}: No detections and no previous mask available")

            # Apply mask: keep only pixels inside any person bbox, black out others
            img = images[t]  # (H, W, C)
            masked_img = torch.zeros_like(img)
            for c in range(img.shape[2]):
                masked_img[..., c][current_mask] = img[..., c][current_mask]
            output[t] = masked_img

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
            video_path = "yolo_bb_preprocess_fem1_t1_output.avi"
            height, width = output_np.shape[1], output_np.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
            fps = 30

            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for t in range(output_np.shape[0]):
                frame = output_np[t]
                # OpenCV expects BGR format, convert from RGB if needed
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)

            out.release()
            print(f"{self.identifier}: Output video written to {video_path}")

        except ImportError:
            raise

        return output
