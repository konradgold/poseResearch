
<h1 align="center"> <a href="https://arxiv.org/abs/2501.01770"> TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation [AAAI 2025 ]</a></h1>

| ![skating](figure/video2.gif)  | ![anime](figure/video3.gif) |
| ------------- | ------------- |



This is the official implementation of the approach described in the paper of TCPFormer :

> [**TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation**](https://arxiv.org/abs/2501.01770) 
            
> Jiajie Liu<sup>1</sup>, Mengyuan Liu<sup>1</sup>, Hong Liu<sup>1</sup>, Wenhao Li<sup>2</sup>

> <sup>1</sup>State Key Laboratory of General Artificial Intelligence, Peking University, Shenzhen Graduate School, <sup>2</sup>Nanyang Technological University


## 💡 Environment
The project is developed under the following environment:
- Python 3.10.x
- PyTorch 2.2.1
- CUDA 12.1

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 

## 🐳 Dataset
### Human3.6M
#### Preprocessing
1. We follow the previous state-of-the-art method [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md) for dataset setup. Download the [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:

**For our model with T = 243**:
```text
python h36m.py  --n-frames 243
```
**or T = 81**
```text
python h36m.py  --n-frames 81
```
**or T = 27**
```text
python h36m.py  --n-frames 81
```


### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

## ✨ Training
After dataset preparation, you can train the model as follows:
### Human3.6M
You can train Human3.6M with the following command:
```
python train.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/h36m`. 
```
python train.py --config configs/h36m/TCPFormer_h36m_243.yaml 
```
### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python train_3dhp.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/mpi`. 
```
python train_3dhp.py --config configs/mpi/TCPFormer_mpi_81.yaml 
```
### Human3.6M (GT)

Please refer to [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md).

## 🚅 Evaluation
| Dataset  | frames | Checkpoint|
|----------|--------|-----------|
|Human3.6M|81|[download](https://drive.google.com/file/d/14D_gfCflgl67-nl0L2MJijbARizbphnP/view?usp=drive_link)|
|Human3.6M|243|[download](https://drive.google.com/file/d/1xiCQaYOWlNBR4uZVGmFJ644mB4tPH-Gq/view?usp=drive_link)|
|MPI-INF-3DHP|9|[download](https://drive.google.com/file/d/1z_foxtKFxz1_g8jOfP-_cqv7ciptpJNo/view?usp=drive_link)|
|MPI-INF-3DHP|27|[download](https://drive.google.com/file/d/1EHl7IFud3JkDmDsDK6vad7O4STAMp9T_/view?usp=drive_link)|
|MPI-INF-3DHP|81|[download](https://drive.google.com/file/d/1ST3NYm-xlgkrMhs3nHm6_WVt6jvCzL-e/view?usp=drive_link)|




After downloading the weight from table above, you can evaluate Human3.6M models by:
```
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
For example if TCPFormer with T = 243 of H.36M is downloaded and put in `checkpoint` directory, then you can run:
```
python train.py --eval-only  --checkpoint checkpoint --checkpoint-file TCPFormer_h36m_243_379.pth.tr --config configs/h36m/TCPFormer_h36m_243.yaml
```

Similarly, TCPFormer with T = 81 of H.36M is downloaded and put in `checkpoint` directory, then you can run:
```
python train.py --eval-only  --checkpoint checkpoint --checkpoint-file TCPFormer_h36m_81_405.pth.tr --config configs/h36m/TCPFormer_h36m_81.yaml
```



For MPI-INF-3DHP dataset, you can download the checkpoint with T = 81 and put in `checkpoint_mpi` directory, then you can run:
```
python train_3dhp.py --eval-only  --checkpoint checkpoint_mpi --checkpoint-file TCPFormer_mpi_81.pth.tr --config configs/mpi/TCPFormer_mpi_81.yaml
```

## 👀 Visualization

For the 3D human pose estimation visualization, please refer to [MHFormer](https://github.com/Vegetebird/MHFormer).

For the attention matrix visualization, this is just a 243x243 matrix, and you can easily visualize it. Let GPT/DeepSeek help you!



## ✏️ Citation

If you find our work useful in your research, please consider citing:

    @article{liu2025tcpformer,
        title={TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation},
        author={Liu, Jiajie and Liu, Mengyuan and Liu, Hong and Li, Wenhao},
        journal={arXiv preprint arXiv:2501.01770},
        year={2025}
    }



## 👍 Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)

## 🔒 Licence

This project is licensed under the terms of the MIT license.



