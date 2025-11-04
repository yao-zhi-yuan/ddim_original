# datasets/mpos.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def _pad_to_square(img, fill=0):
    w, h = img.size
    if w == h: return img
    if w > h:
        pad = (0, (w-h)//2, 0, w-h-(w-h)//2)
    else:
        pad = ((h-w)//2, 0, h-w-(h-w)//2, 0)
    return TF.pad(img, pad, fill=fill)

class MPOSPairs(Dataset):
    """
    期望目录：
      root/
        CFP/  xxx.png|jpg
        FFA/  xxx.png|jpg
        splits/
          train.txt
          val.txt   # 文件名(不含后缀)逐行
    """
    def __init__(self, root, split='train', size=256, ffa_gray=True, random_flip=False):
        self.root = Path(root)
        self.size = size
        self.ffa_gray = ffa_gray
        self.random_flip = random_flip

        with open(self.root/'splits'/f'{split}.txt') as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]

        self.to_tensor_rgb = T.Compose([
            T.Lambda(_pad_to_square),
            T.Resize(size),
            T.ToTensor(),                   # [0,1]

        ])
        self.to_tensor_gray = T.Compose([
            T.Lambda(_pad_to_square),
            T.Resize(size),
            T.ToTensor(),                   # [0,1], shape(1,H,W)

        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]  # id_ 示例: '1_DR/11.png'

        # --- FIX: The ID from train.txt already contains the full relative path.
        # --- We must join it directly, *not* use glob.
        cfp_path = self.root / 'CFP' / id_
        ffa_path = self.root / 'FFA' / id_

        # --- Add a clear check in case the path is *still* wrong ---
        if not cfp_path.exists():
            raise FileNotFoundError(
                f"File not found. Looked for: {cfp_path}\n"
                f"Please check: \n"
                f"  1. Your 'data_root' in config.yml is: {self.root}\n"
                f"  2. The file '{id_}' exists inside 'CFP/' directory.")
        if not ffa_path.exists():
            raise FileNotFoundError(
                f"File not found. Looked for: {ffa_path}\n"
                f"Please check: \n"
                f"  1. Your 'data_root' in config.yml is: {self.root}\n"
                f"  2. The file '{id_}' exists inside 'FFA/' directory.")
        # --- END FIX ---

        # 读图
        cfp = Image.open(cfp_path).convert('RGB')
        ffa = Image.open(ffa_path)
        ffa = ffa.convert('L') if self.ffa_gray else ffa.convert('RGB')

        if self.random_flip:
            if torch.rand(1) < 0.5:
                cfp = TF.hflip(cfp);
                ffa = TF.hflip(ffa)

        cfp_t = self.to_tensor_rgb(cfp)  # (3,H,W)
        ffa_t = self.to_tensor_gray(ffa) if self.ffa_gray else self.to_tensor_rgb(ffa)

        # 多返回原图路径，供可视化用
        meta = {"cond_path": str(cfp_path), "gt_path": str(ffa_path), "id": str(id_)}
        return cfp_t, ffa_t, meta

        # 训练时候用
        # return cfp_t, ffa_t