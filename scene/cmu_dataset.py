import json
import math
import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov
class PanopticDataset(Dataset):
    def __init__(self, datadir: str, json_path: str):
        # --- load metadata once ---
        meta_file = os.path.join(datadir, json_path)
        with open(meta_file, "r") as f:
            meta = json.load(f)

        self.datadir = datadir
        self.w = meta["w"]
        self.h = meta["h"]
        self.max_time = len(meta["fn"])
        self.entries = []

        # flatten (time × camera) into a single list
        for t_idx in range(self.max_time):
            time = t_idx
            Ks = meta["k"][t_idx]  # list of 3×3 intrinsics
            W2Cs = meta["w2c"][t_idx]
            FNs = meta["fn"][t_idx]
            CIDs = meta["cam_id"][t_idx]

            for K_list, w2c_list, fn, cid in zip(Ks, W2Cs, FNs, CIDs):
                # turn that nested list into a real 3×3 array
                K = np.array(K_list, dtype=np.float32).reshape(3, 3)
                fx = float(K[0, 0])
                fy = float(K[1, 1])

                self.entries.append(
                    {
                        "time": time,
                        "K": K,
                        "fx": fx,
                        "fy": fy,
                        "w2c": np.array(w2c_list, dtype=np.float32),
                        "fn": fn,
                        "cam_id": cid,
                    }
                )

        # compute FOVs from fx, fy
        # note: focal2fov(pixels, focal) -> radians
        self.FovX = focal2fov(self.w, np.mean([e["fx"] for e in self.entries]))
        self.FovY = focal2fov(self.h, np.mean([e["fy"] for e in self.entries]))

        # simple PIL→Tensor loader
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]

        # load image on‐the‐fly
        img_path = os.path.join(self.datadir, "ims", e["fn"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        w2c = e["w2c"]
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        return CameraInfo(uid=idx, R=R, T=T, FovY=self.FovY, FovX=self.FovX, image=img,
                                image_path=img_path, image_name=e["fn"], width=self.w, height=self.h,
                                timestamp = e["time"])
