import argparse
import torch
import numpy as np
import json
import logging
from util import utils
from datasets.RelPoseDataset import RelPoseDataset
from models.pose_losses import CameraPoseLoss
from os.path import join
from models.RPR import RPR
import transforms3d as t3d
import time

def compute_rel_pose(poses1, poses2):
    # p1 p_rel = p2
    rel_pose = torch.zeros_like(poses1)
    poses1 = poses1.cpu().numpy()
    poses2 = poses2.cpu().numpy()
    for i, p1 in enumerate(poses1):
        p2 = poses2[i]
        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

        t2 = p2[:3]
        q2 = p2[3:]
        rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

        t_rel = t2 - t1
        rot_rel = np.dot(np.linalg.inv(rot1), rot2)
        q_rel = t3d.quaternions.mat2quat(rot_rel)
        rel_pose[i][:3] = torch.Tensor(t_rel).to(device)
        rel_pose[i][3:] = torch.Tensor(q_rel).to(device)

    return rel_pose

def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)
    return q

def compute_abs_pose_torch(rel_pose, abs_pose_neighbor):
    abs_pose_query = torch.zeros_like(rel_pose)
    abs_pose_query[:, :3] = abs_pose_neighbor[:, :3] + rel_pose[:, :3]
    abs_pose_query[:, 3:] = qmult(abs_pose_neighbor[:, 3:], rel_pose[:, 3:])
    return abs_pose_query

def compute_abs_pose(rel_pose, abs_pose_neighbor, device):
    # p_neighbor p_rel = p_query
    # p1 p_rel = p2
    abs_pose_query = torch.zeros_like(rel_pose)
    rel_pose = rel_pose.cpu().numpy()
    abs_pose_neighbor = abs_pose_neighbor.cpu().numpy()
    for i, rpr in enumerate(rel_pose):
        p1 = abs_pose_neighbor[i]

        t_rel = rpr[:3]
        q_rel = rpr[3:]
        rot_rel = t3d.quaternions.quat2mat(q_rel/ np.linalg.norm(q_rel))

        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1/ np.linalg.norm(q1))

        t2 = t1 + t_rel
        rot2 = np.dot(rot1,rot_rel)
        q2 = t3d.quaternions.mat2quat(rot2)
        abs_pose_query[i][:3] = torch.Tensor(t2).to(device)
        abs_pose_query[i][3:] = torch.Tensor(q2).to(device)

    return abs_pose_query



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file with image pairs, their poses and gt poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("backbone_path",
                            help="path to a pre-trained backbone (e.g. resnet50)")
    arg_parser.add_argument("--rpr_encoder_path", help="paired simsiam")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained RPR model")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} experiment for RPR".format(args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.fdeterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)


    # TODO Create the encoder from SimSiam and load the backbone
    encoder = None

    if args.rpr_encoder_path:
        encoder.load_state_dict(torch.load(args.rpr_encoder_path, map_location=device_id))
        logging.info("Initializing encoder from: {}".format(args.rpr_encoder_path))

    # Create the RPR model - regressor only or with image-based encoder
    model = RPR(config, args.rpr_backbone_path, args.pae_checkpoint_path).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        transform = utils.train_transforms.get('baseline')
        train_dataset = RelPoseDataset(args.dataset_path, args.labels_file, transform)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(train_dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        # Resetting temporal loss used for logging
        running_loss = 0.0
        n_samples = 0

        for epoch in range(n_epochs):
            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_rel_poses = minibatch['rel_pose'].to(dtype=torch.float32)
                batch_size = gt_rel_poses.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                neighbor_poses = minibatch['pose2'].to(device).to(dtype=torch.float32)

                # Estimate the relative pose
                # Zero the gradients
                optim.zero_grad()
                est_rel_poses = model(minibatch['img1'], minibatch['img2']).get('rel_pose')
                criterion = pose_loss(est_rel_poses, gt_rel_poses)
                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_rel_poses.detach(), gt_rel_poses.detach())
                    msg = "[Batch-{}/Epoch-{}] running relative camera pose loss: {:.3f}, camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_freq_print),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item())
                    posit_err, orient_err = utils.pose_err(neighbor_poses.detach(), minibatch['pose1'].to(dtype=torch.float32).detach())
                    msg = msg + ", distance from neighbor images: {:.2f}[m], {:.2f}[deg]".format(posit_err.mean().item(),
                                                                        orient_err.mean().item())
                    logging.info(msg)
                    # Resetting temporal loss used for logging
                    running_loss = 0.0
                    n_samples = 0

            # Save checkpoint3n
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_rpr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_rpr_final.pth')


    else: # Test


        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = RelPoseDataset(args.dataset_path, args.labels_file, transform, False)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        abs_stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                gt_pose = minibatch.get('pose1').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                est_rel_pose = model(minibatch['img1'], minibatch['img2']).get('rel_pose')

                # Flip to get the relative from neighbor to query
                est_rel_pose[:, :3] = -est_rel_pose[:, :3]
                est_rel_pose[:, 4:] = -est_rel_pose[:, 4:]

                est_pose = compute_abs_pose(est_rel_pose, minibatch.get('pose2'), device)

                torch.cuda.synchronize()
                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                abs_stats[i, 0] = posit_err.item()
                abs_stats[i, 1] = orient_err.item()
                abs_stats[i, 2] = (toc - tic) * 1000

                logging.info("Absolute Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    abs_stats[i, 0], abs_stats[i, 1], abs_stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median absolute pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(abs_stats[:, 0]),
                                                                                 np.nanmedian(abs_stats[:, 1])))
        logging.info("Mean pose inference time:{:.2f}[ms]".format(np.mean(abs_stats[:, 2])))



