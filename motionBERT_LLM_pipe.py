import argparse
import torch
from ultralytics import YOLO
import yaml
import os
from MotionAgent.options.option_llm import get_args_parser

import MotionBERT.lib.utils.learning as motionbert_learning
from PoseGPT.models.poseGPT import GPT
from dataset.dataloader import VideoHandler  
from MotionAgent.models.mllm import MotionLLM



def parse_args():
    parser = argparse.ArgumentParser(description='Motion BERT Pipeline')
    parser.add_argument('--config', type=str, required=False, default='configs/pipeline.yml',
                        help='Path to the config file in configs directory')
    parser.add_argument('--video_path', type=str, required=False, default='/Users/konradgoldenbaum/Downloads/interactive4_t3-cam16-2.mp4',
                        help='Path to the input video')
    args = parser.parse_args()
    
    # Verify config file exists and is in yaml format
    if not os.path.exists(args.config) or not args.config.endswith(('.yml', '.yaml')):
        raise ValueError('Config file must exist and be a YAML file')
    
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    class Config:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)
    return Config(config)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Your pipeline code here

    print(f"Loaded config from: {args.config}")
    print(f"Video path: {args.video_path}")

    # Load video
    video_path = args.video_path
    if not os.path.exists(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")
    
    # Initialize video dataset
    video_dataset = VideoHandler(video_path)
    video_dataset.set_batch_size(config.batch_size)  # Set batch size to 1 for single frame processing


    # generate motions (from pretrained motionBERT model)
    model_backbone = motionbert_learning.load_backbone(config)
    if torch.cuda.is_available():
        model_backbone = torch.nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    checkpoint = torch.load(config.checkpoint_path, map_location=lambda storage, loc: storage)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_pos'].items():  # or checkpoint if no 'state_dict' key
        new_key = k.replace('module.', '')  # remove 'module.' prefix
        new_state_dict[new_key] = v

    model_backbone.load_state_dict(new_state_dict, strict=True)
    motionbert_model = model_backbone
    motionbert_model.eval()

    # Optional: Export video with generated motions

    # Transform motions to MotionLLM format
    yolo_model = YOLO(config.yolo_model_path)
    yolo_poses = []
    for batch in video_dataset:
        yolo_poses.append(yolo_model(torch.tensor(batch, dtype=torch.float32)))  # Add batch dimension
        break
    
    # do transformation
    keypoints_list = []
    for batch in yolo_poses:
        for batch_item in batch:
            if batch_item.keypoints is not None:
                print(f"Detected {batch_item.keypoints.data} keypoints in frame")
                keypoints_list.append(torch.Tensor(batch_item.keypoints.data))
            else:
                # If no keypoints detected, create empty placeholder
                keypoints_list.append(None)

    poses_2d = torch.cat(keypoints_list)     # type: ignore

    poses3d = motionbert_model(poses_2d.unsqueeze(1))


    # Load MotionLLM encoder and decoder
    if not config.custom_motion_llm_args:
        motion_llm_args = get_args_parser()

    mllm = MotionLLM(motion_llm_args)
    # Tokenize Video using MotionLLM encoder
    # Pad poses3d from (b,3,17) to (b,3,263) 
    padding = torch.zeros(3, 263-17, device=poses3d.device, dtype=poses3d.dtype)
    for pose in poses3d:
        if pose is None:
            raise ValueError("Pose data is None, ensure keypoints are detected correctly.")
        pose = pose.squeeze()
        pose = pose.transpose(0, 1)  # Change from (3, 17) to (17, 3)
        pose = torch.cat([pose, padding], dim=1)

        pose_encoding = mllm.caption(pose)
        print(f"Encoded pose shape: {pose_encoding}")


    #pose_decoding = mllm.net.embeddings_decode(pose_encoding)

    # Save the trained poseGPT model

    # Training loop
    optimizer = torch.optim.Adam(mllm.llm.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        mllm.train()
        total_loss = 0
        
        # Use pose_encoding as input and target with teacher forcing
        outputs = mllm.llm(pose_encoding)
        loss = criterion(outputs.view(-1, outputs.size(-1)), pose_encoding.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {total_loss:.4f}')

    # Save model
    torch.save(mllm.llm.state_dict(), config.save_path)

    # Save pose_decoding to file
    pose_save_path = os.path.join(os.path.dirname(config.save_path), 'pose_decoding.pt')
    torch.save(pose_decoding, pose_save_path)
    print(f"Saved pose decoding to: {pose_save_path}")



if __name__ == '__main__':
    main()