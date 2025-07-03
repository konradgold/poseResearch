import os
import random
import numpy as np
import argparse
import errno
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import prettytable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from MotionBERT.lib.utils.tools import get_config
from MotionBERT.lib.utils.learning import AverageMeter, load_backbone
from MotionBERT.lib.utils.utils_data import flip_data
from MotionBERT.lib.data.dataset_motion_3d import MotionDataset2D
from MotionBERT.lib.data.augmentation import Augmenter2D
from MotionBERT.lib.model.loss import loss_2d_weighted, mpjpe, p_mpjpe


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print("Saving checkpoint to", chk_path)
    torch.save(
        {
            "epoch": epoch + 1,
            "lr": lr,
            "optimizer": optimizer.state_dict(),
            "model_pos": model_pos.state_dict(),
            "min_loss": min_loss,
        },
        chk_path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/motionbert/train/overfit.yml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="MotionBERT/checkpoint/pose3d/overfitTT",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        default="MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite",
        type=str,
        metavar="PATH",
        help="pretrained checkpoint directory",
    )
    parser.add_argument(
        "-ms",
        "--selection",
        default="best_epoch.bin",
        type=str,
        metavar="FILENAME",
        help="checkpoint to finetune (file name)",
    )
    parser.add_argument("-sd", "--seed", default=0, type=int, help="random seed")
    opts = parser.parse_args()
    return opts


def evaluate(args, model_pos, test_loader, datareader):
    print("INFO: Testing")
    results_all = []
    model_pos.eval()
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                batch_gt[:, 0, 0, 2] = 0

            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset["test"]["action"])
    factors = np.array(datareader.dt_dataset["test"]["2.5d_factor"])
    gts = np.array(datareader.dt_dataset["test"]["joints_2.5d_image"])
    sources = np.array(datareader.dt_dataset["test"]["source"])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset["test"]["action"]))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = [
        "s_09_act_05_subact_02",
        "s_09_act_10_subact_02",
        "s_09_act_13_subact_01",
    ]
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ["test_name"] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(["P1"] + final_result)
    summary_table.add_row(["P2"] + final_result_procrustes)
    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    print("Protocol #1 Error (MPJPE):", e1, "mm")
    print("Protocol #2 Error (P-MPJPE):", e2, "mm")
    print("----------")
    return e1, e2, results_all


def train_epoch(args, model_pos, train_loader, losses, optimizer, has_gt):
    model_pos.train()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            conf = copy.deepcopy(
                batch_input[:, :, :, 2:]
            )  # For 2D data, weight/confidence is at the last channel
            batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(
                    batch_input, noise=(args.noise and has_gt), mask=args.mask
                )
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input)  # (N, T, 17, 3)

        optimizer.zero_grad()
        loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
        loss_total = loss_2d_proj
        losses["2d_proj"].update(loss_2d_proj.item(), batch_size)
        losses["total"].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()


def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                "Unable to create checkpoint directory:", opts.checkpoint
            )
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    print("Loading dataset...")
    trainloader_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 12,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    }

    testloader_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 12,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    }

    subset_list = args.subset_list

    if args.expand_subset_list:
        expanded_list = []
        for subset in subset_list:
            base_path = os.path.join(args.data_root, subset)
            for root, dirs, files in os.walk(base_path):
                # Only add paths that contain files, not just subdirectories
                if files:
                    rel_path = os.path.relpath(root, args.data_root)
                    expanded_list.append(rel_path)
        subset_list = expanded_list

    train_dataset = MotionDataset2D(args, subset_list, "")
    test_dataset = MotionDataset2D(args, subset_list, "")
    train_loader_2d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)

    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print("INFO: Trainable parameter count:", model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    chk_filename = os.path.join(opts.pretrained, opts.selection)
    print("Loading checkpoint", chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_pos"].items():  # or checkpoint if no 'state_dict' key
        new_key = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[new_key] = v

    model_backbone.load_state_dict(new_state_dict, strict=True)
    model_pos = model_backbone

    if True:  # not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model_pos.parameters()),
            lr=lr,
            weight_decay=args.weight_decay,
        )
        lr_decay = args.lr_decay
        st = 0

        print("INFO: Training on {} batches".format(len(train_loader_2d)))

        args.mask = args.mask_ratio > 0 and args.mask_T_ratio > 0
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)

        # Training
        for epoch in range(st, args.epochs):
            print("Training epoch %d." % epoch)
            start_time = time()
            losses = {}
            losses["3d_pos"] = AverageMeter()
            losses["3d_scale"] = AverageMeter()
            losses["2d_proj"] = AverageMeter()
            losses["lg"] = AverageMeter()
            losses["lv"] = AverageMeter()
            losses["total"] = AverageMeter()
            losses["3d_velocity"] = AverageMeter()
            losses["angle"] = AverageMeter()
            losses["angle_velocity"] = AverageMeter()

            train_epoch(
                args, model_pos, train_loader_2d, losses, optimizer, has_gt=True
            )

            elapsed = (time() - start_time) / 60

            # if args.no_eval:
            e1 = np.inf
            print(
                "[%d] time %.2f lr %f 3d_train %f"
                % (epoch + 1, elapsed, lr, losses["3d_pos"].avg)
            )
            if False:
                e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)
                print(
                    "[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f"
                    % (epoch + 1, elapsed, lr, losses["3d_pos"].avg, e1, e2)
                )
                train_writer.add_scalar("Error P1", e1, epoch + 1)
                train_writer.add_scalar("Error P2", e2, epoch + 1)
                train_writer.add_scalar("loss_3d_pos", losses["3d_pos"].avg, epoch + 1)
                train_writer.add_scalar(
                    "loss_2d_proj", losses["2d_proj"].avg, epoch + 1
                )
                train_writer.add_scalar(
                    "loss_3d_scale", losses["3d_scale"].avg, epoch + 1
                )
                train_writer.add_scalar(
                    "loss_3d_velocity", losses["3d_velocity"].avg, epoch + 1
                )
                train_writer.add_scalar("loss_lv", losses["lv"].avg, epoch + 1)
                train_writer.add_scalar("loss_lg", losses["lg"].avg, epoch + 1)
                train_writer.add_scalar("loss_a", losses["angle"].avg, epoch + 1)
                train_writer.add_scalar(
                    "loss_av", losses["angle_velocity"].avg, epoch + 1
                )
                train_writer.add_scalar("loss_total", losses["total"].avg, epoch + 1)

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, "epoch_{}.bin".format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, "latest_epoch.bin")
            chk_path_best = os.path.join(opts.checkpoint, "best_epoch.bin")

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(
                    chk_path_best, epoch, lr, optimizer, model_pos, min_loss
                )

    if False:  # opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)


if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
