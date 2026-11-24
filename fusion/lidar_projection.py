import os
import numpy as np
import cv2
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud


class LidarToCameraProjector:
    """
    Projects nuScenes LiDAR point clouds into camera image space
    using per-frame calibrated_sensor and ego_pose.
    """

    def __init__(self, nuscenes):
        """
        nuscenes: NuScenes object
        """
        self.nusc = nuscenes

    def project(
        self,
        sample,
        cam_channel="CAM_FRONT",
        out_h=640,
        out_w=960,
        min_depth=1.0,
        blur=True,
    ):
        """
        Returns:
            lidar_img: (H, W) float32 normalized depth-density map
        """

        # -----------------------------
        # Load sample data
        # -----------------------------
        cam_sd = self.nusc.get("sample_data", sample["data"][cam_channel])
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

        # Load point cloud
        lidar_path = os.path.join(self.nusc.dataroot, lidar_sd["filename"])
        pc = LidarPointCloud.from_file(lidar_path)
        pts = pc.points[:3]  # (3, N)

        # -----------------------------
        # Load calibration
        # -----------------------------
        cs_lid = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_lid = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        cs_cam = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        ego_cam = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

        # -----------------------------
        # LiDAR → ego (lidar timestamp)
        # -----------------------------
        pts = (
            Quaternion(cs_lid["rotation"]).rotation_matrix @ pts
            + np.array(cs_lid["translation"]).reshape(3, 1)
        )

        # -----------------------------
        # ego → global
        # -----------------------------
        pts = (
            Quaternion(ego_lid["rotation"]).rotation_matrix @ pts
            + np.array(ego_lid["translation"]).reshape(3, 1)
        )

        # -----------------------------
        # global → ego (camera timestamp)
        # -----------------------------
        pts = Quaternion(ego_cam["rotation"]).inverse.rotation_matrix @ (
            pts - np.array(ego_cam["translation"]).reshape(3, 1)
        )

        # -----------------------------
        # ego → camera
        # -----------------------------
        pts = Quaternion(cs_cam["rotation"]).inverse.rotation_matrix @ (
            pts - np.array(cs_cam["translation"]).reshape(3, 1)
        )

        # -----------------------------
        # Depth filtering
        # -----------------------------
        z = pts[2]
        valid = z > min_depth
        pts = pts[:, valid]

        if pts.shape[1] == 0:
            return np.zeros((out_h, out_w), dtype=np.float32)

        # -----------------------------
        # Camera intrinsics (scaled)
        # -----------------------------
        K = np.array(cs_cam["camera_intrinsic"])
        orig_w, orig_h = cam_sd["width"], cam_sd["height"]

        K[0] *= out_w / orig_w
        K[1] *= out_h / orig_h

        # -----------------------------
        # Projection
        # -----------------------------
        uv = K @ pts
        uv = uv[:2] / uv[2:]

        px = np.round(uv.T).astype(np.int32)

        # -----------------------------
        # Rasterization (density map)
        # -----------------------------
        lidar_img = np.zeros((out_h, out_w), dtype=np.float32)

        for u, v in px:
            if 0 <= u < out_w and 0 <= v < out_h:
                lidar_img[v, u] += 1.0

        if blur:
            lidar_img = cv2.GaussianBlur(lidar_img, (5, 5), 0)

        # Normalize
        maxv = lidar_img.max()
        if maxv > 0:
            lidar_img /= maxv

        return lidar_img
