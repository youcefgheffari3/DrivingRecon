import torch
import torch.nn.functional as F
from utils.projection import project_world_to_image
from utils.warp_feature import warp_feature_with_depth


def warp_feature_to_ref_frame(feat_tgt, depth_tgt, pose_tgt2ref, intrinsics):
    """
    Warp feature map from target frame to reference frame using depth and relative pose.

    Args:
        feat_tgt (Tensor): Feature map from target frame [B, C, H, W]
        depth_tgt (Tensor): Depth map of target frame [B, 1, H, W]
        pose_tgt2ref (Tensor): Transformation matrix T_ref <- T_tgt [B, 4, 4]
        intrinsics (Tensor): Camera intrinsics [B, 3, 3]

    Returns:
        feat_warped (Tensor): Warped feature map aligned to reference frame
    """
    return warp_feature_with_depth(
        feat_tgt, depth_tgt, pose_tgt2ref, intrinsics
    )


def fuse_features(ref_feat, warped_feats):
    """
    Fuse reference frame features with a list of warped features from other frames.

    Args:
        ref_feat (Tensor): Feature map from reference frame [B, C, H, W]
        warped_feats (List[Tensor]): List of warped feature maps [B, C, H, W]

    Returns:
        Tensor: Fused feature map [B, C, H, W]
    """
    all_feats = [ref_feat] + warped_feats
    fused = torch.stack(all_feats, dim=0).mean(dim=0)  # Simple average fusion
    return fused


def compute_relative_pose(pose_ref, pose_tgt):
    """
    Compute relative transformation from target to reference.

    Args:
        pose_ref (Tensor): [B, 4, 4] Pose of reference frame
        pose_tgt (Tensor): [B, 4, 4] Pose of target frame

    Returns:
        Tensor: Relative pose from target to reference [B, 4, 4]
    """
    return torch.bmm(torch.inverse(pose_ref), pose_tgt)
