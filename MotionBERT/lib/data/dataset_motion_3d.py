import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset
from MotionBERT.lib.data.augmentation import Augmenter3D
from MotionBERT.lib.utils.tools import read_pkl
from MotionBERT.lib.utils.utils_data import flip_data
import cv2
from ultralytics import YOLO
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        self.clip_len = args.clip_len
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 
    
class MotionDataset2D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset2D, self).__init__(args, subset_list, data_split)
        self.flip: bool = args.flip
        self.synthetic: bool = args.synthetic
        self.clip_len = args.clip_len
        self.data_stride = args.data_stride
        self.yolo_model = YOLO()
        for dir in self.file_list:
            # Count the number of files in the directory
            if os.path.isdir(dir):
                num_files = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
            else:
                num_files = 1  # If it's a file, count as 1
            if not hasattr(self, 'cumsum'):
                self.cumsum = [0]
                self.total = 0
            else:
                self.total = self.cumsum[-1]
            self.cumsum.append(self.total + num_files)

    def __len__(self):
        return self.total // self.data_stride - 1

    def __getitem__(self, index):
        assert index < self.total // self.data_stride - 1 
        i = 0
        while self.cumsum[i+1] <= index * self.data_stride:
            i += 1
        
        offset = index * self.data_stride - self.cumsum[i]
        img_dir = self.file_list[i]
        img_files = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        # Select a sequence of images; if not enough, go back from the end
        if offset + self.clip_len <= len(img_files):
            selected_imgs = img_files[offset:offset + self.clip_len]
        else:
            # Not enough images, take the last clip_len images
            selected_imgs = img_files[-self.clip_len:]

        imgs = []
        for img_name in selected_imgs:
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)  # (self.clip_len, h, w, 3)
        
        # Extract keypoints using YOLO11n-pose model
        keypoints = []
        # Run YOLO inference on all images at once (batch inference)
        results = self.yolo_model(imgs)
        for res in results:
            if hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints.data) > 0:
            # Get the first person's keypoints (shape: [17, 3] for COCO format)
                kpts = res.keypoints.data[0].cpu().numpy()
                if kpts.shape[0] == 17:
                    keypoints.append(kpts)
                else:
                    padded_kpts = np.zeros((17, 3))
                    min_kpts = min(kpts.shape[0], 17)
                    padded_kpts[:min_kpts] = kpts[:min_kpts]
                    keypoints.append(padded_kpts)
            else:
                keypoints.append(np.zeros((17, 3)))
        keypoints = np.stack(keypoints, axis=0)  # (self.clip_len, 17, 3)
        return imgs, keypoints

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)