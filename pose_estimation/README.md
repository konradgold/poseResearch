- Zuerst aktivieren vom environment in pose_estimation mit source pose_estimation/.venv/bin/activate
- uv sync ausführen
- Probieren obs geht :D
Wenn nicht, dann:
- uv pip install wheel muss manuell ausgeführt werden, um chumpy zu installieren
- Danach kann uv sync ausgeführt werden, um chumpy 0.70 zu bauen
- Install uv pip install cython numpy wheel setuptools pip (oder nur cython bei mir, um ein tool von xtcocotools zu bauen)
Jetzt muss mmcv installiert werden:
- in pose_estimation clonen mit git clone https://github.com/open-mmlab/mmcv.git
- reingehen mit cd mmcv
- Version wechseln mit git checkout v2.1.0 (da 2.2.0 nicht funkt)
- In Ordner pose_estimation/mmcv führt das aus: uv pip install MMCV_WITH_OPS=1 uv pip install -e .

die models müssen runtergeladen werden:

person detection (making squares around them): https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
2d keypoint detection: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
2d to 3d lift: https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth