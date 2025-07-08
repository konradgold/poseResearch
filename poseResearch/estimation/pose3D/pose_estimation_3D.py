from abc import abstractmethod
import torch
from ..util import Estimation


class ThreeDPoseEstimation(Estimation):
    """
    Abstract base class for 3D pose estimation.
    Input: 2D poses as a tensor of shape (P, T, Nk, D) in the h36m format (17 keypoints).
    Output: 3D poses as a tensor of shape (P, T, Nk, D) in the h36m format (17 keypoints).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            poses_2d (torch.Tensor): Input 2D poses of shape (P, T, Nk, D)
        Returns:
            torch.Tensor: Output tensor (shape defined by subclass)
        """
        pass

    def _post_process(self, output: torch.Tensor) -> torch.Tensor:
        """
        Post-process the output to ensure it's in h36m format.
        Override this method if your model outputs in a different format.

        Args:
            output (torch.Tensor): Output from _forward method

        Returns:
            torch.Tensor: Post-processed output in h36m format
        """
        # By default, assume the output is already in h36m format
        # Override this method if conversion is needed
        return output

    def _normalization(self, output: torch.Tensor) -> torch.Tensor:
        """
        Normalize 3D poses by centering root at (0,0,0) and scaling root-belly distance to 1.

        Args:
            output (torch.Tensor): Output from _post_process method (P, T, 17, 3)

        Returns:
            torch.Tensor: Normalized output where root is at (0,0,0) and root-belly distance is 1
        """
        P, T, Nk, D = output.shape
        normalized_output = output.clone()

        # h36m format: root=0, belly=7
        root_idx = 0
        belly_idx = 7

        for p in range(P):
            for t in range(T):
                # Get root and belly positions (x, y, z coordinates)
                root_pos = normalized_output[
                    p, t, root_idx, :3
                ].clone()  # (3,) - x, y, z
                belly_pos = normalized_output[
                    p, t, belly_idx, :3
                ].clone()  # (3,) - x, y, z

                # Calculate root-belly distance before translation
                root_belly_vector = belly_pos - root_pos
                root_belly_dist = torch.norm(root_belly_vector)

                if root_belly_dist > 1e-6:  # Avoid division by zero
                    # Step 1: Translate to center root at (0, 0, 0)
                    normalized_output[p, t, :, :3] = normalized_output[
                        p, t, :, :3
                    ] - root_pos.unsqueeze(0)

                    # Step 2: Scale to make root-belly distance = 1
                    scale_factor = 1.0 / root_belly_dist
                    normalized_output[p, t, :, :3] = (
                        normalized_output[p, t, :, :3] * scale_factor
                    )

        return normalized_output

    def output_check(self, input_tensor) -> bool:
        """
        Checks if the input tensor has the correct shape: (P, T, Nk, D)
        Returns True if valid, False otherwise.
        """
        if not isinstance(input_tensor, torch.Tensor):
            print(f"Warning: {self.identifier} returned non-tensor output.")
            return False
        if input_tensor.ndim != 4:
            print(
                f"Warning: {self.identifier} returned tensor with {input_tensor.ndim} dimensions. Expected 4."
            )
            return False
        P, T, Nk, D = input_tensor.shape
        expected_Nk = getattr(self, "num_keypoints", 17)
        expected_D = getattr(self, "num_dims", 3)
        if D != expected_D or Nk != expected_Nk:
            print(
                f"Warning: {self.identifier} returned tensor with {Nk} keypoints and {D} dimensions. Expected {expected_Nk} keypoints and {expected_D} dimensions."
            )
            return False
        if T < 1:
            print(f"Warning: {self.identifier} returned no frames.")
            return False
        if P < 1:
            print(f"Warning: {self.identifier} returned no persons.")
        return True
