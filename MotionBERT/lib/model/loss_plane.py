import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def torch_project_to_plane(predicted, target):
    assert len(predicted.shape()) == 3
    assert len(target.shape()) == 3

    assert predicted.shape(0) == target.shape(0)
    assert predicted.shape(1) ==  target.shape(1)
    assert predicted.shape(2) == target.shape(2) + 1

    image_size = (640, 480)  # Width x Height
    focal_length = image_size[0]

    out = torch.zeros(target.shape())

    camera_matrix = np.array([
        [focal_length, 0, image_size[0] / 2],
        [0, focal_length, image_size[1] / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    for i, keypoints_3d in enumerate(predicted):

        # Convert tensors to numpy arrays and ensure correct dtype/shape
        keypoints_3d_np = keypoints_3d.detach().cpu().numpy().astype(np.float64)
        target_np = target[i].detach().cpu().numpy().astype(np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            keypoints_3d_np,
            target_np,
            camera_matrix,
            dist_coeffs
        )

        if not success:
            continue  # Skip this pair if PnP fails

        projected_points, _ = cv2.projectPoints(
            keypoints_3d,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        out[i] = torch.tensor(projected_points.reshape(-1, 2))

    return out





def numpy_project_to_plane(predicted, target):
    ...

# Numpy-based errors

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    predicted = numpy_project_to_plane(predicted, target)
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    predicted = numpy_project_to_plane(predicted, target)
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    predicted = torch_project_to_plane(predicted, target)
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    predicted = torch_project_to_plane(predicted, target)
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    predicted = torch_project_to_plane(predicted, target)
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var

def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt) # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)

def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)

