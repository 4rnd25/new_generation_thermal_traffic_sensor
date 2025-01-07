"""
Created on Nov 25 11:28

@author: ISAC - pettirsch
"""

import torch

from Utils.Cuboid_calc.calc_3d_corners import get_3d_corners

from pytorch3d.ops import box3d_overlap

def cuboid_iou_3d(target_bc, target_dim, target_yaw, pred_bc_img, pred_dim, pred_obs_angle_sin, pred_obs_angle_cos, bottom_map,
                  cam_pos, cls, cls_mean_lookup, target_bc_img = None, distance_threshold = 100, dist_threshold_low = 0):

    valid_targets_idx = torch.arange(target_bc.shape[0], device=target_bc.device)
    num_targets = target_bc.shape[0]
    num_preds = pred_bc_img.shape[0]

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
        distance = torch.sqrt(torch.sum((target_bc - cam_pos_repeated) ** 2, dim=1))
        valid_targets = (distance < distance_threshold) #& (distance >= dist_threshold_low)
        target_bc = target_bc[valid_targets]
        target_dim = target_dim[valid_targets]
        target_yaw = target_yaw[valid_targets]
        valid_targets_idx = valid_targets_idx[valid_targets]

    # Get the target corners
    target_corners = get_3d_corners(target_bc, target_dim, target_yaw, local=False)

    # Get the predicted corners
    pred_bc_img_x = torch.round(pred_bc_img[:, 0])
    pred_bc_img_y = torch.round(pred_bc_img[:, 1])
    out_of_bounds = (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) | (pred_bc_img_y < 0) | (
            pred_bc_img_y >= bottom_map.shape[1])
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds].long()
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds].long()

    # Include out-of-bounds predictions in valid_preds
    valid_preds_idx = torch.arange(pred_bc_img.shape[0], device=pred_bc_img.device)
    valid_preds_idx_out_of_bounds = valid_preds_idx[out_of_bounds]
    valid_preds_idx_in_bounds = valid_preds_idx[~out_of_bounds]

    # Get BC world
    bottom_map = bottom_map.to(pred_bc_img.device)
    cam_pos = cam_pos.to(pred_bc_img.device)
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid,:]+cam_pos

    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)
    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]
    pred_bc_world_valid = torch.stack([pred_bc_world_x_valid, pred_bc_world_y_valid, pred_bc_world_z_valid], dim=1)

    valid_preds_idx = valid_preds_idx_in_bounds[in_bounds_world]

    # Expand camera position
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_x_valid.shape[0], 1).to(pred_bc_img.device)
    osin_valid = pred_obs_angle_sin[~out_of_bounds][in_bounds_world]
    ocos_valid = pred_obs_angle_cos[~out_of_bounds][in_bounds_world]
    magnitude = torch.sqrt(osin_valid ** 2 + ocos_valid ** 2 + 1e-8)  # Add small epsilon for numerical stability
    sin_alpha = osin_valid / magnitude
    cos_alpha = ocos_valid / magnitude
    obs_angle = torch.atan2(sin_alpha, cos_alpha)
    atan_xy = torch.atan2(pred_bc_world_x_valid - cam_pos_expanded[:, 0], pred_bc_world_y_valid - cam_pos_expanded[:, 1])
    yaw_angle = atan_xy + obs_angle

    # dim 3d
    delta_lenght = pred_dim[:, 0][~out_of_bounds][in_bounds_world]
    delta_width = pred_dim[:, 1][~out_of_bounds][in_bounds_world]
    delta_height = pred_dim[:, 2][~out_of_bounds][in_bounds_world]

    # Get mean values
    cls_mean_lookup = cls_mean_lookup.to(pred_bc_img.device)
    mean_values = cls_mean_lookup[cls.long()]
    mean_lenght = mean_values[0]
    mean_width = mean_values[1]
    mean_height = mean_values[2]
    length = mean_lenght * torch.exp(delta_lenght)
    width = mean_width * torch.exp(delta_width)
    height = mean_height * torch.exp(delta_height)
    dims_pred = torch.stack([length, width, height], dim=1)

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
        distance = torch.sqrt(torch.sum((pred_bc_world_valid - cam_pos_repeated) ** 2, dim=1))
        valid_preds = (distance < distance_threshold) #& (distance >= dist_threshold_low)
        pred_bc_world_valid = pred_bc_world_valid[valid_preds]
        dims_pred = dims_pred[valid_preds]
        yaw_angle = yaw_angle[valid_preds]
        cam_pos_expanded = cam_pos_expanded[valid_preds]
        valid_preds_idx = valid_preds_idx[valid_preds]

    # get the predicted corners
    pred_corners = get_3d_corners(pred_bc_world_valid, dims_pred, yaw_angle, local=False)

    # Subtract the camera position from the corners
    cam_pos_repeated = cam_pos_expanded.unsqueeze(1).repeat(1, 8, 1)
    pred_corners = pred_corners - cam_pos_repeated
    cam_pos_expanded_t = cam_pos.unsqueeze(0).repeat(target_corners.shape[0], 1).to(target_corners.device)
    cam_pos_repeated_t = cam_pos_expanded_t.unsqueeze(1).repeat(1, 8, 1)
    target_corners = target_corners - cam_pos_repeated_t

    # Compute the intersection volume
    #pred_corners = pred_corners.to('cpu')
    #target_corners = target_corners.to('cpu')
    pred_corners = pred_corners.float()
    target_corners = target_corners.float()

    if pred_corners.shape[0] == 0 or target_corners.shape[0] == 0:
        return torch.zeros((num_preds, num_targets), dtype=torch.float32).to(pred_bc_img.device), valid_targets_idx, valid_preds_idx

    try:
        intersection_vol, iou_3d = box3d_overlap(pred_corners, target_corners)
    except:
        # Set all bottom z values in pred of each box to the same value
        pred_corners[:, 0:4, 2] = pred_corners[:, 0, 2].unsqueeze(1).repeat(1, 4)
        target_corners[:, 0:4, 2] = target_corners[:, 0, 2].unsqueeze(1).repeat(1, 4)

        # Set all top z values in pred of each box to the same value
        pred_corners[:, 4:8, 2] = pred_corners[:, 4, 2].unsqueeze(1).repeat(1, 4)
        target_corners[:, 4:8, 2] = target_corners[:, 4, 2].unsqueeze(1).repeat(1, 4)
        try:
            intersection_vol, iou_3d = box3d_overlap(pred_corners, target_corners)
        except:
            return torch.zeros((num_preds, num_targets), dtype=torch.float32).to(pred_bc_img.device), valid_targets_idx, valid_preds_idx

    return iou_3d, valid_targets_idx, valid_preds_idx

def cuboid_iou_bev(target_bc, target_dim, target_yaw, pred_bc_img, pred_dim, pred_obs_angle_sin, pred_obs_angle_cos, bottom_map,
                  cam_pos, cls, cls_mean_lookup, target_bc_img = None, distance_threshold = None):

    valid_targets_idx = torch.arange(target_bc.shape[0], device=target_bc.device)
    num_targets = target_bc.shape[0]
    num_preds = pred_bc_img.shape[0]

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
        distance = torch.sqrt(torch.sum((target_bc - cam_pos_repeated) ** 2, dim=1))
        valid_targets = distance < distance_threshold
        target_bc = target_bc[valid_targets]
        target_dim = target_dim[valid_targets]
        target_yaw = target_yaw[valid_targets]
        valid_targets_idx = valid_targets_idx[valid_targets]

    # Get the target corners
    target_corners = get_3d_corners(target_bc, target_dim, target_yaw, local=False)

    # Get the predicted corners
    pred_bc_img_x = torch.round(pred_bc_img[:, 0])
    pred_bc_img_y = torch.round(pred_bc_img[:, 1])
    out_of_bounds = (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) | (pred_bc_img_y < 0) | (
            pred_bc_img_y >= bottom_map.shape[1])
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds].long()
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds].long()

    # Include out-of-bounds predictions in valid_preds
    valid_preds_idx = torch.arange(pred_bc_img.shape[0], device=pred_bc_img.device)
    valid_preds_idx_out_of_bounds = valid_preds_idx[out_of_bounds]
    valid_preds_idx_in_bounds = valid_preds_idx[~out_of_bounds]

    # Get BC world
    bottom_map = bottom_map.to(pred_bc_img.device)
    cam_pos = cam_pos.to(pred_bc_img.device)
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid,:]+cam_pos

    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)
    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]
    pred_bc_world_valid = torch.stack([pred_bc_world_x_valid, pred_bc_world_y_valid, pred_bc_world_z_valid], dim=1)

    valid_preds_idx = valid_preds_idx_in_bounds[in_bounds_world]

    # Expand camera position
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_x_valid.shape[0], 1).to(pred_bc_img.device)
    osin_valid = pred_obs_angle_sin[~out_of_bounds][in_bounds_world]
    ocos_valid = pred_obs_angle_cos[~out_of_bounds][in_bounds_world]
    magnitude = torch.sqrt(osin_valid ** 2 + ocos_valid ** 2 + 1e-8)  # Add small epsilon for numerical stability
    sin_alpha = osin_valid / magnitude
    cos_alpha = ocos_valid / magnitude
    obs_angle = torch.atan2(sin_alpha, cos_alpha)
    atan_xy = torch.atan2(pred_bc_world_x_valid - cam_pos_expanded[:, 0], pred_bc_world_y_valid - cam_pos_expanded[:, 1])
    yaw_angle = atan_xy + obs_angle

    # dim 3d
    delta_lenght = pred_dim[:, 0][~out_of_bounds][in_bounds_world]
    delta_width = pred_dim[:, 1][~out_of_bounds][in_bounds_world]
    delta_height = pred_dim[:, 2][~out_of_bounds][in_bounds_world]

    # Get mean values
    cls_mean_lookup = cls_mean_lookup.to(pred_bc_img.device)
    mean_values = cls_mean_lookup[cls.long()]
    mean_lenght = mean_values[0]
    mean_width = mean_values[1]
    mean_height = mean_values[2]
    length = mean_lenght * torch.exp(delta_lenght)
    width = mean_width * torch.exp(delta_width)
    height = mean_height * torch.exp(delta_height)
    dims_pred = torch.stack([length, width, height], dim=1)

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
        distance = torch.sqrt(torch.sum((pred_bc_world_valid - cam_pos_repeated) ** 2, dim=1))
        valid_preds = distance < distance_threshold
        pred_bc_world_valid = pred_bc_world_valid[valid_preds]
        dims_pred = dims_pred[valid_preds]
        yaw_angle = yaw_angle[valid_preds]
        cam_pos_expanded = cam_pos_expanded[valid_preds]
        valid_preds_idx = valid_preds_idx[valid_preds]

    # get the predicted corners
    pred_corners = get_3d_corners(pred_bc_world_valid, dims_pred, yaw_angle, local=False)

    # Subtract the camera position from the corners
    cam_pos_repeated = cam_pos_expanded.unsqueeze(1).repeat(1, 8, 1)
    pred_corners = pred_corners - cam_pos_repeated
    cam_pos_expanded_t = cam_pos.unsqueeze(0).repeat(target_corners.shape[0], 1).to(target_corners.device)
    cam_pos_repeated_t = cam_pos_expanded_t.unsqueeze(1).repeat(1, 8, 1)
    target_corners = target_corners - cam_pos_repeated_t

    # Compute the intersection volume
    pred_corners = pred_corners.float()
    target_corners = target_corners.float()

    # Set all bottom z values in pred of each box to 0 and top z values to 1e-8
    pred_corners[:, 0:4, 2] = 0
    pred_corners[:, 4:8, 2] = 1
    target_corners[:, 0:4, 2] = 0
    target_corners[:, 4:8, 2] = 1


    if pred_corners.shape[0] == 0 or target_corners.shape[0] == 0:
        return torch.zeros((num_preds, num_targets), dtype=torch.float32).to(pred_bc_img.device), valid_targets_idx, valid_preds_idx

    intersection_vol, iou_3d = box3d_overlap(pred_corners, target_corners)

    return iou_3d, valid_targets_idx, valid_preds_idx


def cuboid_dist(target_bc, target_dim, target_yaw, pred_bc_img, pred_dim, pred_obs_angle_sin, pred_obs_angle_cos, bottom_map,
                  cam_pos, cls, cls_mean_lookup, target_bc_img = None, distance_threshold = 50, dist_threshold_low = 30):

    valid_targets_idx = torch.arange(target_bc.shape[0], device=target_bc.device)
    num_targets = target_bc.shape[0]
    num_preds = pred_bc_img.shape[0]

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
        distance = torch.sqrt(torch.sum((target_bc - cam_pos_repeated) ** 2, dim=1))
        valid_targets = (distance < distance_threshold) & (distance >= dist_threshold_low)
        target_bc = target_bc[valid_targets]
        valid_targets_idx = valid_targets_idx[valid_targets]

    # Get the predicted corners
    pred_bc_img_x = torch.round(pred_bc_img[:, 0])
    pred_bc_img_y = torch.round(pred_bc_img[:, 1])
    out_of_bounds = (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) | (pred_bc_img_y < 0) | (
            pred_bc_img_y >= bottom_map.shape[1])
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds].long()
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds].long()

    # Include out-of-bounds predictions in valid_preds
    valid_preds_idx = torch.arange(pred_bc_img.shape[0], device=pred_bc_img.device)
    valid_preds_idx_out_of_bounds = valid_preds_idx[out_of_bounds]
    valid_preds_idx_in_bounds = valid_preds_idx[~out_of_bounds]

    # Get BC world
    bottom_map = bottom_map.to(pred_bc_img.device)
    cam_pos = cam_pos.to(pred_bc_img.device)
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid,:]+cam_pos

    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)
    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]
    pred_bc_world_valid = torch.stack([pred_bc_world_x_valid, pred_bc_world_y_valid, pred_bc_world_z_valid], dim=1)

    valid_preds_idx = valid_preds_idx_in_bounds[in_bounds_world]

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
        distance = torch.sqrt(torch.sum((pred_bc_world_valid - cam_pos_repeated) ** 2, dim=1))
        valid_preds = (distance < distance_threshold) & (distance >= dist_threshold_low)
        pred_bc_world_valid = pred_bc_world_valid[valid_preds]
        valid_preds_idx = valid_preds_idx[valid_preds]

    # Subtract the camera position from the corners
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
    pred_bc_world_valid = pred_bc_world_valid - cam_pos_expanded
    cam_pos_expanded_t = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
    target_bc = target_bc - cam_pos_expanded_t

    pred_bc_world_valid = pred_bc_world_valid.float()
    target_bc = target_bc.float()

    # Compute the intersection volume
    if pred_bc_world_valid.shape[0] == 0 or target_bc.shape[0] == 0:
        return torch.ones((num_preds, num_targets), dtype=torch.float32).to(
            pred_bc_img.device) * 1000, valid_targets_idx, valid_preds_idx

    # Compute pairwise Euclidean distances between predicted and target bottom centers
    distances = torch.cdist(pred_bc_world_valid, target_bc, p=2)  # Shape: (num_preds, num_targets)

    return distances, valid_targets_idx, valid_preds_idx

def cuboid_dist_all_data(target_bc, target_dim, target_yaw, pred_bc_img, pred_dim, pred_obs_angle_sin, pred_obs_angle_cos, bottom_map,
                  cam_pos, cls, cls_mean_lookup, target_bc_img = None, distance_threshold = None, dist_threshold_low = 0):

    valid_targets_idx = torch.arange(target_bc.shape[0], device=target_bc.device)
    num_targets = target_bc.shape[0]
    num_preds = pred_bc_img.shape[0]

    cam_pos_repeated = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
    distance_cam= torch.sqrt(torch.sum((target_bc - cam_pos_repeated) ** 2, dim=1))

    if distance_threshold is not None:
        # Remove all targets that are too far away
        valid_targets = (distance_cam < distance_threshold) & (distance_cam >= dist_threshold_low)
        target_bc = target_bc[valid_targets]
        target_dim = target_dim[valid_targets]
        target_yaw = target_yaw[valid_targets]
        valid_targets_idx = valid_targets_idx[valid_targets]

    # Get the target corners
    target_corners = get_3d_corners(target_bc, target_dim, target_yaw, local=False)

    # Get the predicted corners
    pred_bc_img_x = torch.round(pred_bc_img[:, 0])
    pred_bc_img_y = torch.round(pred_bc_img[:, 1])
    out_of_bounds = (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) | (pred_bc_img_y < 0) | (
            pred_bc_img_y >= bottom_map.shape[1])
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds].long()
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds].long()

    # Include out-of-bounds predictions in valid_preds
    valid_preds_idx = torch.arange(pred_bc_img.shape[0], device=pred_bc_img.device)
    valid_preds_idx_out_of_bounds = valid_preds_idx[out_of_bounds]
    valid_preds_idx_in_bounds = valid_preds_idx[~out_of_bounds]

    # Get BC world
    bottom_map = bottom_map.to(pred_bc_img.device)
    cam_pos = cam_pos.to(pred_bc_img.device)
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid,:]+cam_pos

    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)
    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]
    pred_bc_world_valid = torch.stack([pred_bc_world_x_valid, pred_bc_world_y_valid, pred_bc_world_z_valid], dim=1)

    valid_preds_idx = valid_preds_idx_in_bounds[in_bounds_world]

    # Expand camera position
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_x_valid.shape[0], 1).to(pred_bc_img.device)
    osin_valid = pred_obs_angle_sin[~out_of_bounds][in_bounds_world]
    ocos_valid = pred_obs_angle_cos[~out_of_bounds][in_bounds_world]
    magnitude = torch.sqrt(osin_valid ** 2 + ocos_valid ** 2 + 1e-8)  # Add small epsilon for numerical stability
    sin_alpha = osin_valid / magnitude
    cos_alpha = ocos_valid / magnitude
    obs_angle = torch.atan2(sin_alpha, cos_alpha)
    atan_xy = torch.atan2(pred_bc_world_x_valid - cam_pos_expanded[:, 0],
                          pred_bc_world_y_valid - cam_pos_expanded[:, 1])
    yaw_angle = atan_xy + obs_angle

    # dim 3d
    delta_lenght = pred_dim[:, 0][~out_of_bounds][in_bounds_world]
    delta_width = pred_dim[:, 1][~out_of_bounds][in_bounds_world]
    delta_height = pred_dim[:, 2][~out_of_bounds][in_bounds_world]

    # Get mean values
    cls_mean_lookup = cls_mean_lookup.to(pred_bc_img.device)
    mean_values = cls_mean_lookup[cls.long()]
    mean_lenght = mean_values[0]
    mean_width = mean_values[1]
    mean_height = mean_values[2]
    length = mean_lenght * torch.exp(delta_lenght)
    width = mean_width * torch.exp(delta_width)
    height = mean_height * torch.exp(delta_height)
    dims_pred = torch.stack([length, width, height], dim=1)

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
        distance = torch.sqrt(torch.sum((pred_bc_world_valid - cam_pos_repeated) ** 2, dim=1))
        valid_preds = (distance < distance_threshold) & (distance >= dist_threshold_low)
        pred_bc_world_valid = pred_bc_world_valid[valid_preds]
        valid_preds_idx = valid_preds_idx[valid_preds]
        dims_pred = dims_pred[valid_preds]
        yaw_angle = yaw_angle[valid_preds]
        cam_pos_expanded = cam_pos_expanded[valid_preds]

    # get the predicted corners
    pred_corners = get_3d_corners(pred_bc_world_valid, dims_pred, yaw_angle, local=False)

    # Subtract the camera position from the corners
    cam_pos_repeated = cam_pos_expanded.unsqueeze(1).repeat(1, 8, 1)
    pred_corners = pred_corners - cam_pos_repeated
    cam_pos_expanded_t = cam_pos.unsqueeze(0).repeat(target_corners.shape[0], 1).to(target_corners.device)
    cam_pos_repeated_t = cam_pos_expanded_t.unsqueeze(1).repeat(1, 8, 1)
    target_corners = target_corners - cam_pos_repeated_t

    # Compute the intersection volume
    pred_corners = pred_corners.float()
    target_corners = target_corners.float()

    # Subtract the camera position from the corners
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
    pred_bc_world_valid = pred_bc_world_valid - cam_pos_expanded
    cam_pos_expanded_t = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
    target_bc = target_bc - cam_pos_expanded_t

    pred_bc_world_valid = pred_bc_world_valid.float()
    target_bc = target_bc.float()

    # Compute the intersection volume
    if pred_bc_world_valid.shape[0] == 0 or target_bc.shape[0] == 0:
        return torch.ones((num_preds, num_targets), dtype=torch.float32).to(
            pred_bc_img.device) * 1000, valid_targets_idx, valid_preds_idx

    # Compute pairwise Euclidean distances between predicted and target bottom centers
    distances = torch.cdist(pred_bc_world_valid, target_bc, p=2)  # Shape: (num_preds, num_targets)

    # Reshape tensors for broadcasting
    yaq_pred = yaw_angle.view(-1, 1)  # Shape (n, 1)
    yaw_t = target_yaw.view(1, -1)  # Shape (1, m)

    # Compute periodic angular differences (generalized for any range)
    angle_diff = torch.remainder(yaq_pred - yaw_t + torch.pi, 2 * torch.pi) - torch.pi  # Wrap to [-pi, pi]
    angle_diff = torch.abs(angle_diff)  # Take the absolute value to get the smallest difference

    # Compute squared differences
    head_ang_diff_mat = angle_diff ** 2  # Shape (n, m)

    vol_pred = dims_pred[:, 0] * dims_pred[:, 1] * dims_pred[:, 2]
    vol_target = target_dim[:, 0] * target_dim[:, 1] * target_dim[:, 2]

    # Compute the mean absolute percentage error (MAPE) between predicted and target volumes
    mape_mat = torch.abs((vol_pred.view(-1, 1) - vol_target.view(1, -1)) / vol_target.view(1, -1))  # Shape (n, m)

    return distances, valid_targets_idx, valid_preds_idx, distances, head_ang_diff_mat, mape_mat, distance_cam



def cuboid_iou_3d_all_data(target_bc, target_dim, target_yaw, pred_bc_img, pred_dim, pred_obs_angle_sin, pred_obs_angle_cos, bottom_map,
                  cam_pos, cls, cls_mean_lookup, target_bc_img = None, distance_threshold = None):

    valid_targets_idx = torch.arange(target_bc.shape[0], device=target_bc.device)
    num_targets = target_bc.shape[0]
    num_preds = pred_bc_img.shape[0]


    # Remove all targets that are too far away
    cam_pos_repeated = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
    distance_cam= torch.sqrt(torch.sum((target_bc - cam_pos_repeated) ** 2, dim=1))


    if distance_threshold is not None:
        valid_targets = distance_cam < distance_threshold
        target_bc = target_bc[valid_targets]
        target_dim = target_dim[valid_targets]
        target_yaw = target_yaw[valid_targets]
        valid_targets_idx = valid_targets_idx[valid_targets]

    # Get the target corners
    target_corners = get_3d_corners(target_bc, target_dim, target_yaw, local=False)

    # Get the predicted corners
    pred_bc_img_x = torch.round(pred_bc_img[:, 0])
    pred_bc_img_y = torch.round(pred_bc_img[:, 1])
    out_of_bounds = (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) | (pred_bc_img_y < 0) | (
            pred_bc_img_y >= bottom_map.shape[1])
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds].long()
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds].long()

    # Include out-of-bounds predictions in valid_preds
    valid_preds_idx = torch.arange(pred_bc_img.shape[0], device=pred_bc_img.device)
    valid_preds_idx_out_of_bounds = valid_preds_idx[out_of_bounds]
    valid_preds_idx_in_bounds = valid_preds_idx[~out_of_bounds]

    # Get BC world
    bottom_map = bottom_map.to(pred_bc_img.device)
    cam_pos = cam_pos.to(pred_bc_img.device)
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid,:]+cam_pos

    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)
    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]
    pred_bc_world_valid = torch.stack([pred_bc_world_x_valid, pred_bc_world_y_valid, pred_bc_world_z_valid], dim=1)

    valid_preds_idx = valid_preds_idx_in_bounds[in_bounds_world]

    # Expand camera position
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_x_valid.shape[0], 1).to(pred_bc_img.device)
    osin_valid = pred_obs_angle_sin[~out_of_bounds][in_bounds_world]
    ocos_valid = pred_obs_angle_cos[~out_of_bounds][in_bounds_world]
    magnitude = torch.sqrt(osin_valid ** 2 + ocos_valid ** 2 + 1e-8)  # Add small epsilon for numerical stability
    sin_alpha = osin_valid / magnitude
    cos_alpha = ocos_valid / magnitude
    obs_angle = torch.atan2(sin_alpha, cos_alpha)
    atan_xy = torch.atan2(pred_bc_world_x_valid - cam_pos_expanded[:, 0], pred_bc_world_y_valid - cam_pos_expanded[:, 1])
    yaw_angle = atan_xy + obs_angle

    # dim 3d
    delta_lenght = pred_dim[:, 0][~out_of_bounds][in_bounds_world]
    delta_width = pred_dim[:, 1][~out_of_bounds][in_bounds_world]
    delta_height = pred_dim[:, 2][~out_of_bounds][in_bounds_world]

    # Get mean values
    cls_mean_lookup = cls_mean_lookup.to(pred_bc_img.device)
    mean_values = cls_mean_lookup[cls.long()]
    mean_lenght = mean_values[0]
    mean_width = mean_values[1]
    mean_height = mean_values[2]
    length = mean_lenght * torch.exp(delta_lenght)
    width = mean_width * torch.exp(delta_width)
    height = mean_height * torch.exp(delta_height)
    dims_pred = torch.stack([length, width, height], dim=1)

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
        distance = torch.sqrt(torch.sum((pred_bc_world_valid - cam_pos_repeated) ** 2, dim=1))
        valid_preds = distance < distance_threshold
        pred_bc_world_valid = pred_bc_world_valid[valid_preds]
        dims_pred = dims_pred[valid_preds]
        yaw_angle = yaw_angle[valid_preds]
        cam_pos_expanded = cam_pos_expanded[valid_preds]
        valid_preds_idx = valid_preds_idx[valid_preds]

    # get the predicted corners
    pred_corners = get_3d_corners(pred_bc_world_valid, dims_pred, yaw_angle, local=False)

    # Subtract the camera position from the corners
    cam_pos_repeated = cam_pos_expanded.unsqueeze(1).repeat(1, 8, 1)
    pred_corners = pred_corners - cam_pos_repeated
    cam_pos_expanded_t = cam_pos.unsqueeze(0).repeat(target_corners.shape[0], 1).to(target_corners.device)
    cam_pos_repeated_t = cam_pos_expanded_t.unsqueeze(1).repeat(1, 8, 1)
    target_corners = target_corners - cam_pos_repeated_t

    # Compute the intersection volume
    pred_corners = pred_corners.to('cpu')
    target_corners = target_corners.to('cpu')
    pred_corners = pred_corners.float()
    target_corners = target_corners.float()

    if pred_corners.shape[0] == 0 or target_corners.shape[0] == 0:
        return torch.zeros((num_preds, num_targets), dtype=torch.float32).to(pred_bc_img.device), valid_targets_idx, valid_preds_idx, None, None, None, distance_cam

    try:
        intersection_vol, iou_3d = box3d_overlap(pred_corners, target_corners)
    except:
        # Set all bottom z values in pred of each box to the same value
        pred_corners[:, 0:4, 2] = pred_corners[:, 0, 2].unsqueeze(1).repeat(1, 4)
        target_corners[:, 0:4, 2] = target_corners[:, 0, 2].unsqueeze(1).repeat(1, 4)

        # Set all top z values in pred of each box to the same value
        pred_corners[:, 4:8, 2] = pred_corners[:, 4, 2].unsqueeze(1).repeat(1, 4)
        target_corners[:, 4:8, 2] = target_corners[:, 4, 2].unsqueeze(1).repeat(1, 4)
        try:
            intersection_vol, iou_3d = box3d_overlap(pred_corners, target_corners)
        except:
            return torch.zeros((num_preds, num_targets), dtype=torch.float32).to(pred_bc_img.device), valid_targets_idx, valid_preds_idx, None, None, None, distance_cam

    # Compute pairwise Euclidean distances between predicted and target bottom centers
    dist_mat = torch.cdist(pred_bc_world_valid, target_bc, p=2)  # Shape: (num_preds, num_targets)

    # Reshape tensors for broadcasting
    yaq_pred = yaw_angle.view(-1, 1)  # Shape (n, 1)
    yaw_t = target_yaw.view(1, -1)  # Shape (1, m)

    # Compute periodic angular differences (generalized for any range)
    angle_diff = torch.remainder(yaq_pred - yaw_t + torch.pi, 2 * torch.pi) - torch.pi  # Wrap to [-pi, pi]
    angle_diff = torch.abs(angle_diff)  # Take the absolute value to get the smallest difference

    # Compute squared differences
    head_ang_diff_mat = angle_diff ** 2  # Shape (n, m)

    vol_pred = dims_pred[:, 0] * dims_pred[:, 1] * dims_pred[:, 2]
    vol_target = target_dim[:, 0] * target_dim[:, 1] * target_dim[:, 2]

    # Compute the mean absolute percentage error (MAPE) between predicted and target volumes
    mape_mat = torch.abs((vol_pred.view(-1, 1) - vol_target.view(1, -1)) / vol_target.view(1, -1))  # Shape (n, m)

    return iou_3d, valid_targets_idx, valid_preds_idx, dist_mat, head_ang_diff_mat, mape_mat, distance_cam

def cuboid_iou_bev_all_data(target_bc, target_dim, target_yaw, pred_bc_img, pred_dim, pred_obs_angle_sin, pred_obs_angle_cos, bottom_map,
                  cam_pos, cls, cls_mean_lookup, target_bc_img = None, distance_threshold = None):

    valid_targets_idx = torch.arange(target_bc.shape[0], device=target_bc.device)
    num_targets = target_bc.shape[0]
    num_preds = pred_bc_img.shape[0]

    cam_pos_repeated = cam_pos.unsqueeze(0).repeat(target_bc.shape[0], 1).to(target_bc.device)
    distance_cam = torch.sqrt(torch.sum((target_bc - cam_pos_repeated) ** 2, dim=1))

    if distance_threshold is not None:
        # Remove all targets that are too far away
        valid_targets = distance_cam < distance_threshold
        target_bc = target_bc[valid_targets]
        target_dim = target_dim[valid_targets]
        target_yaw = target_yaw[valid_targets]
        valid_targets_idx = valid_targets_idx[valid_targets]

    # Get the target corners
    target_corners = get_3d_corners(target_bc, target_dim, target_yaw, local=False)

    # Get the predicted corners
    pred_bc_img_x = torch.round(pred_bc_img[:, 0])
    pred_bc_img_y = torch.round(pred_bc_img[:, 1])
    out_of_bounds = (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) | (pred_bc_img_y < 0) | (
            pred_bc_img_y >= bottom_map.shape[1])
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds].long()
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds].long()

    # Include out-of-bounds predictions in valid_preds
    valid_preds_idx = torch.arange(pred_bc_img.shape[0], device=pred_bc_img.device)
    valid_preds_idx_out_of_bounds = valid_preds_idx[out_of_bounds]
    valid_preds_idx_in_bounds = valid_preds_idx[~out_of_bounds]

    # Get BC world
    bottom_map = bottom_map.to(pred_bc_img.device)
    cam_pos = cam_pos.to(pred_bc_img.device)
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid,:]+cam_pos

    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)
    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]
    pred_bc_world_valid = torch.stack([pred_bc_world_x_valid, pred_bc_world_y_valid, pred_bc_world_z_valid], dim=1)

    valid_preds_idx = valid_preds_idx_in_bounds[in_bounds_world]

    # Expand camera position
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_x_valid.shape[0], 1).to(pred_bc_img.device)
    osin_valid = pred_obs_angle_sin[~out_of_bounds][in_bounds_world]
    ocos_valid = pred_obs_angle_cos[~out_of_bounds][in_bounds_world]
    magnitude = torch.sqrt(osin_valid ** 2 + ocos_valid ** 2 + 1e-8)  # Add small epsilon for numerical stability
    sin_alpha = osin_valid / magnitude
    cos_alpha = ocos_valid / magnitude
    obs_angle = torch.atan2(sin_alpha, cos_alpha)
    atan_xy = torch.atan2(pred_bc_world_x_valid - cam_pos_expanded[:, 0], pred_bc_world_y_valid - cam_pos_expanded[:, 1])
    yaw_angle = atan_xy + obs_angle

    # dim 3d
    delta_lenght = pred_dim[:, 0][~out_of_bounds][in_bounds_world]
    delta_width = pred_dim[:, 1][~out_of_bounds][in_bounds_world]
    delta_height = pred_dim[:, 2][~out_of_bounds][in_bounds_world]

    # Get mean values
    cls_mean_lookup = cls_mean_lookup.to(pred_bc_img.device)
    mean_values = cls_mean_lookup[cls.long()]
    mean_lenght = mean_values[0]
    mean_width = mean_values[1]
    mean_height = mean_values[2]
    length = mean_lenght * torch.exp(delta_lenght)
    width = mean_width * torch.exp(delta_width)
    height = mean_height * torch.exp(delta_height)
    dims_pred = torch.stack([length, width, height], dim=1)

    if distance_threshold is not None:
        # Remove all targets that are too far away
        cam_pos_repeated = cam_pos.unsqueeze(0).repeat(pred_bc_world_valid.shape[0], 1).to(pred_bc_world_valid.device)
        distance = torch.sqrt(torch.sum((pred_bc_world_valid - cam_pos_repeated) ** 2, dim=1))
        valid_preds = distance < distance_threshold
        pred_bc_world_valid = pred_bc_world_valid[valid_preds]
        dims_pred = dims_pred[valid_preds]
        yaw_angle = yaw_angle[valid_preds]
        cam_pos_expanded = cam_pos_expanded[valid_preds]
        valid_preds_idx = valid_preds_idx[valid_preds]

    # get the predicted corners
    pred_corners = get_3d_corners(pred_bc_world_valid, dims_pred, yaw_angle, local=False)

    # Subtract the camera position from the corners
    cam_pos_repeated = cam_pos_expanded.unsqueeze(1).repeat(1, 8, 1)
    pred_corners = pred_corners - cam_pos_repeated
    cam_pos_expanded_t = cam_pos.unsqueeze(0).repeat(target_corners.shape[0], 1).to(target_corners.device)
    cam_pos_repeated_t = cam_pos_expanded_t.unsqueeze(1).repeat(1, 8, 1)
    target_corners = target_corners - cam_pos_repeated_t

    # Compute the intersection volume
    pred_corners = pred_corners.to('cpu')
    target_corners = target_corners.to('cpu')
    pred_corners = pred_corners.float()
    target_corners = target_corners.float()

    # Set all bottom z values in pred of each box to 0 and top z values to 1e-8
    pred_corners[:, 0:4, 2] = 0
    pred_corners[:, 4:8, 2] = 1
    target_corners[:, 0:4, 2] = 0
    target_corners[:, 4:8, 2] = 1

    # pred_corners = pred_corners.to('cpu')
    # target_corners = target_corners.to('cpu')

    if pred_corners.shape[0] == 0 or target_corners.shape[0] == 0:
        return torch.zeros((num_preds, num_targets), dtype=torch.float32).to(pred_bc_img.device), valid_targets_idx, valid_preds_idx, None, None, None, distance_cam

    intersection_vol, iou_3d = box3d_overlap(pred_corners, target_corners)

    # Compute pairwise Euclidean distances between predicted and target bottom centers
    dist_mat = torch.cdist(pred_bc_world_valid[:,0:2], target_bc[:,0:2], p=2)  # Shape: (num_preds, num_targets)

    # Reshape tensors for broadcasting
    yaq_pred = yaw_angle.view(-1, 1)  # Shape (n, 1)
    yaw_t = target_yaw.view(1, -1)  # Shape (1, m)

    # Compute periodic angular differences (generalized for any range)
    angle_diff = torch.remainder(yaq_pred - yaw_t + torch.pi, 2 * torch.pi) - torch.pi  # Wrap to [-pi, pi]
    angle_diff = torch.abs(angle_diff)  # Take the absolute value to get the smallest difference

    # Compute squared differences
    head_ang_diff_mat = angle_diff ** 2  # Shape (n, m)

    vol_pred = dims_pred[:, 0] * dims_pred[:, 1] * dims_pred[:, 2]
    vol_target = target_dim[:, 0] * target_dim[:, 1] * target_dim[:, 2]

    # Compute the mean absolute percentage error (MAPE) between predicted and target volumes
    mape_mat = torch.abs((vol_pred.view(-1, 1) - vol_target.view(1, -1)) / vol_target.view(1, -1))  # Shape (n, m)

    return iou_3d, valid_targets_idx, valid_preds_idx, dist_mat, head_ang_diff_mat, mape_mat, distance_cam

