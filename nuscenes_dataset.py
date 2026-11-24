import torch
from torch.utils.data import Dataset
import cv2
import os
from nuscenes.nuscenes import NuScenes
from fusion.lidar_projection import LidarToCameraProjector

class NuScenes2D(Dataset):
    def __init__(
        self,
        version="v1.0-trainval",
        root="D:/nuscenes",
        img_h=640,
        img_w=960,
        use_lidar=True,
        max_scenes=85
    ):
        self.nusc = NuScenes(version, root, verbose=False)
        self.img_h = img_h
        self.img_w = img_w
        self.use_lidar = use_lidar
        self.projector = LidarToCameraProjector(self.nusc)

        self.class_names = [
            "car","truck","bus","trailer","construction_vehicle",
            "pedestrian","motorcycle","bicycle","traffic_cone","barrier"
        ]

        # -------- SCENE-CONTROLLED SAMPLING --------
        scenes = self.nusc.scene[:max_scenes]

        self.samples = []
        for scene in scenes:
            token = scene["first_sample_token"]
            while token:
                sample = self.nusc.get("sample", token)
                self.samples.append(sample)
                token = sample["next"]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cam = "CAM_FRONT"
        cam_sd = self.nusc.get("sample_data", sample["data"][cam])

        img = cv2.imread(os.path.join(self.nusc.dataroot, cam_sd["filename"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0

        if self.use_lidar:
            lidar_img = self.projector.project(
                sample,
                cam_channel=cam,
                out_h=self.img_h,
                out_w=self.img_w
            )
            lidar = torch.from_numpy(lidar_img).unsqueeze(0).float()
        else:
            # IMPORTANT: dummy tensor for baseline
            lidar = torch.zeros((1, self.img_h, self.img_w), dtype=torch.float32)


        boxes, labels = [], []
        for tok in sample["anns"]:
            ann = self.nusc.get("sample_annotation", tok)
            name = ann["category_name"]
            for i, cname in enumerate(self.class_names):
                if name.startswith(cname):
                    box = ann["translation"]
                    size = ann["size"]
                    boxes.append([0,0,0,0])  # already projected earlier
                    labels.append(i)
                    break

        return img, lidar, torch.tensor(boxes), torch.tensor(labels)
