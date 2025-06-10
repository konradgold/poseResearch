import argparse
import torch
from ultralytics import YOLO
import yaml
import os

import MotionBERT.lib.utils.learning as motionbert_learning
from PoseGPT.models.poseGPT import GPT
from dataset.dataloader import VideoHandler  
from MotionAgent.models.mllm import MotionLLM



def parse_args():
    parser = argparse.ArgumentParser(description='Motion BERT Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file in configs directory')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the input video')
    args = parser.parse_args()
    
    # Verify config file exists and is in yaml format
    if not os.path.exists(args.config) or not args.config.endswith(('.yml', '.yaml')):
        raise ValueError('Config file must exist and be a YAML file')
    
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    video_dataset.set_batch_size(20)  # Set batch size to 1 for single frame processing


    # generate motions (from pretrained motionBERT model)
    model_backbone = motionbert_learning.load_backbone(config)
    if torch.cuda.is_available():
        model_backbone = torch.nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    checkpoint = torch.load(config['checkpoint_path'], map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()

    # Optional: Export video with generated motions

    # Transform motions to MotionLLM format
    yolo_model = YOLO(config['yolo_model_path'])
    yolo_poses = []
    for batch in video_dataset:
        yolo_poses.append(yolo_model(batch, 
                   batch_size=1, 
                   flip=False, 
                   synthetic=False, 
                   data_stride=1, 
               clip_len=16, 
               num_workers=4))
    
    # do transformation
    poses_2d = yolo_poses # type: ignore

    poses3d = model_pos(poses_2d)


    # Load MotionLLM encoder and decoder

    mllm = MotionLLM(config)
    # Tokenize Video using MotionLLM encoder

    pose_encoding = mllm.net.encode(poses3d)
    pose_decoding = mllm.net.embeddings_decode(pose_encoding)

    pose_gpt = GPT(config['vocal_size'], config['num_layers'], config['num_heads'], config['dropout'])

    def set_codebook(mllm, pose_gpt):
        # TODO: Implement codebook setting logic
        return pose_gpt

    pose_gpt = set_codebook(mllm, pose_gpt)
    # Save the trained poseGPT model

    # Training loop
    optimizer = torch.optim.Adam(pose_gpt.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config['num_epochs']):
        pose_gpt.train()
        total_loss = 0
        
        # Use pose_encoding as input and target with teacher forcing
        outputs = pose_gpt(pose_encoding)
        loss = criterion(outputs.view(-1, outputs.size(-1)), pose_encoding.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {total_loss:.4f}')

    # Save model
    torch.save(pose_gpt.state_dict(), config['save_path'])



if __name__ == '__main__':
    main()