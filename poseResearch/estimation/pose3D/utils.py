import numpy as np

s_org_36_jt_num = 32
s_36_root_jt_idx = 0
s_36_lsh_jt_idx = 11
s_36_rsh_jt_idx = 14
s_36_jt_num = 18
s_36_flip_pairs = np.array(
    [[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int32
)
s_36_parent_ids = np.array(
    [0, 0, 1, 2, 0, 4, 5, 0, 17, 17, 8, 17, 11, 12, 17, 14, 15, 0], dtype=np.int32
)
s_36_bone_jts = np.array(
    [
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16],
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
    ]
)
s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8, -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_hm36_2_mpii_jt = [3, 2, 1, 4, 5, 6, 0, 17, 8, 10, 16, 15, 14, 11, 12, 13]

s_coco_2_hm36_jt = [-1, 12, 14, 16, 11, 13, 15, -1, -1, 0, -1, 5, 7, 9, 6, 8, 10, -1]

s_posetrack_2_hm36_jt = [-1, 2, 1, 0, 3, 4, 5, -1, 12, 13, 14, 9, 10, 11, 8, 7, 6, -1]


def from_coco_to_hm36_single(pose, pose_vis):
    res_jts = np.zeros((s_36_jt_num, 3), dtype=np.float32)
    res_vis = np.zeros((s_36_jt_num, 3), dtype=np.float32)

    for i in range(0, s_36_jt_num):
        id1 = i
        id2 = s_coco_2_hm36_jt[i]
        if id2 >= 0:
            res_jts[id1] = pose[id2].copy()
            res_vis[id1] = pose_vis[id2].copy()

    return res_jts.copy(), res_vis.copy()


def from_coco_to_hm36(db):
    for n_sample in range(0, len(db)):
        res_jts, res_vis = from_coco_to_hm36_single(
            db[n_sample]["joints_3d"], db[n_sample]["joints_3d_vis"]
        )
        db[n_sample]["joints_3d"] = res_jts
        db[n_sample]["joints_3d_vis"] = res_vis
    return db
