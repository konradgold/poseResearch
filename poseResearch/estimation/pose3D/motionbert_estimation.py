import torch
from .pose_estimation_3D import ThreeDPoseEstimation


class MotionBERTEstimation(ThreeDPoseEstimation):
    """
    Abstract base class for 3D pose estimation.
    Input: 2D poses as a tensor of shape (P, T, Nk, D)
    Output: (to be defined by subclasses)
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(
            self.checkpoint_path, map_location=lambda storage, loc: storage
        )

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "MotionBERTEstimation"

    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            poses_2d (torch.Tensor): Input 2D poses of shape (P, T, Nk, D)
        Returns:
            torch.Tensor: Output tensor (shape defined by subclass)
        """
        # return poses_2d
        model_backbone.load_state_dict(self.checkpoint["model_pos"], strict=True)
        model_pos = model_backbone
        model_pos.eval()
        testloader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 8,
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "drop_last": False,
        }

        vid = imageio.get_reader(opts.vid_path, "ffmpeg")
        fps_in = vid.get_meta_data()["fps"]
        vid_size = vid.get_meta_data()["size"]
        os.makedirs(opts.out_path, exist_ok=True)

        if opts.pixel:
            # Keep relative scale with pixel coornidates
            wild_dataset = WildDetDataset(
                opts.json_path,
                clip_len=opts.clip_len,
                vid_size=vid_size,
                scale_range=None,
                focus=opts.focus,
            )
        else:
            # Scale to [-1,1]
            wild_dataset = WildDetDataset(
                opts.json_path,
                clip_len=opts.clip_len,
                scale_range=[1, 1],
                focus=opts.focus,
            )

        test_loader = DataLoader(wild_dataset, **testloader_params)

        results_all = []
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                N, T = batch_input.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                if args.no_conf:
                    batch_input = batch_input[:, :, :, :2]
                if args.flip:
                    batch_input_flip = flip_data(batch_input)
                    predicted_3d_pos_1 = model_pos(batch_input)
                    predicted_3d_pos_flip = model_pos(batch_input_flip)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
                else:
                    predicted_3d_pos = model_pos(batch_input)
                if args.rootrel:
                    predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
                else:
                    predicted_3d_pos[:, 0, 0, 2] = 0
                    pass
                if args.gt_2d:
                    predicted_3d_pos[..., :2] = batch_input[..., :2]
                results_all.append(predicted_3d_pos.cpu().numpy())

        results_all = np.hstack(results_all)
        results_all = np.concatenate(results_all)
        render_and_save(
            results_all, "%s/X3D.mp4" % (opts.out_path), keep_imgs=False, fps=fps_in
        )
        if opts.pixel:
            # Convert to pixel coordinates
            results_all = results_all * (min(vid_size) / 2.0)
            results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0
        np.save("%s/X3D.npy" % (opts.out_path), results_all)
