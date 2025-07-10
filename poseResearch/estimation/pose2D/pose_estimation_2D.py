from abc import abstractmethod
import torch
from ..util import Estimation


class TwoDPoseEstimation(Estimation):
    """
    Abstract base class for 2D pose estimation.
    Input: images as a tensor of shape (T, H, W, C)
    Output: 2D poses as a tensor of shape (P, T, Nk, D) in the h36m format (17 keypoints).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D)
        """
        pass

    def _post_process(self, output: torch.Tensor) -> torch.Tensor:
        """
        Post-process the output to ensure consistent h36m format.
        Override this method if your model outputs in a different format that needs conversion.

        Args:
            output (torch.Tensor): Output from _forward method

        Returns:
            torch.Tensor: Post-processed output in h36m format
        """
        # Default implementation: assume output is already in h36m format
        return output

    def _normalization(self, output: torch.Tensor) -> torch.Tensor:
        """
        Normalize 2D poses by centering root at (0,0) and scaling root-belly distance to 1.

        Args:
            output (torch.Tensor): Output from _post_process method (P, T, 17, 3)

        Returns:
            torch.Tensor: Normalized output where root is at (0,0) and root-belly distance is 1
        """
        P, T, Nk, D = output.shape
        normalized_output = output.clone()

        # h36m format: root=0, belly=7
        root_idx = 0
        belly_idx = 7

        for p in range(P):
            for t in range(T):
                # Check if both root and belly have valid positions (non-zero confidence)
                root_conf = normalized_output[p, t, root_idx, 2]
                belly_conf = normalized_output[p, t, belly_idx, 2]

                if (
                    root_conf > 0 and belly_conf > 0
                ):  # Only normalize if both are detected
                    # Get root and belly positions (x, y coordinates only)
                    root_pos = normalized_output[
                        p, t, root_idx, :2
                    ].clone()  # (2,) - x, y
                    belly_pos = normalized_output[
                        p, t, belly_idx, :2
                    ].clone()  # (2,) - x, y

                    # Calculate root-belly distance before translation
                    root_belly_vector = belly_pos - root_pos
                    root_belly_dist = torch.norm(root_belly_vector)

                    if root_belly_dist > 1e-6:  # Avoid division by zero
                        # Step 1: Translate to center root at (0, 0)
                        normalized_output[p, t, :, :2] = normalized_output[
                            p, t, :, :2
                        ] - root_pos.unsqueeze(0)

                        # Step 2: Scale to make root-belly distance = 1
                        scale_factor = 1.0 / root_belly_dist
                        normalized_output[p, t, :, :2] = (
                            normalized_output[p, t, :, :2] * scale_factor
                        )

        return normalized_output

    def output_check(self, output) -> bool:
        """
        Checks if the output tensor has the correct shape: (P, T, Nk, D)
        Returns True if valid, False otherwise.
        """
        if not isinstance(output, torch.Tensor):
            print(f"Warning: {self.identifier} returned non-tensor output.")
            return False
        if output.ndim != 4:
            print(
                f"Warning: {self.identifier} returned tensor with {output.ndim} dimensions. Expected 4."
            )
            return False
        P, T, Nk, D = output.shape
        print(f"P: {P}, T: {T}, Nk: {Nk}, D: {D}")
        expected_Nk = getattr(self, "num_keypoints", 17)
        expected_D = getattr(self, "num_dims", 3)
        if D != expected_D or Nk != expected_Nk:
            print(
                f"Warning: {self.identifier} returned tensor with {Nk} keypoints and {D} dimensions. Expected {expected_Nk} keypoints and {expected_D} dimensions."
            )
            if Nk == 0 and D == expected_Nk * expected_D:
                print(
                    f"Warning: {self.identifier} expected output tensor with {expected_Nk} keypoints and {expected_D} dimensions, but got {Nk} keypoints and {D} dimensions. This might occur when no person is detected."
                )
                return True
            return False
        if T < 1:
            print(f"Warning: {self.identifier} returned no frames.")
            return False
        if P < 1:
            print(f"Warning: {self.identifier} returned no persons.")
        return True
