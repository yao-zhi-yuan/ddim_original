# import os, sys, glob
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
#
# # ==== 路径按你的截图修改 ====
# PRED_ROOT = "exp/mpos256_ch96_s2025/image_samples/samples_mpos256_ch96_s2025/pred"
# GT_ROOT   = "./exp/mpos/FFA"   # ← 如与你实际不同，改成你的GT根目录
# SAVE_CSV  = "./metrics_psnr_ssim_32000.csv"
#
# # 可选：去边（0.90=只保留中心90%）
# CENTER_KEEP = 1.00
#
# def center_crop_keep(img: Image.Image, keep=1.0):
#     if keep >= 0.999: return img
#     w, h = img.size; cw, ch = int(w*keep), int(h*keep)
#     l = (w-cw)//2; t = (h-ch)//2
#     return img.crop((l, t, l+cw, t+ch))
#
# def load_gray_uint8(path):
#     im = Image.open(path).convert("L")
#     im = center_crop_keep(im, CENTER_KEEP)
#     return np.array(im)
#
# pred_files = sorted([p for p in glob.glob(os.path.join(PRED_ROOT, "**", "*.*"), recursive=True)
#                      if os.path.isfile(p)])
#
# rows = []
# missing = []
# for p in tqdm(pred_files, desc="PSNR/SSIM"):
#     rel = os.path.relpath(p, PRED_ROOT)  # 例如 "5_CSC/2.png"
#     gt_path = os.path.join(GT_ROOT, rel)
#     if not os.path.exists(gt_path):
#         missing.append(rel); continue
#
#     pred = load_gray_uint8(p)
#     gt   = load_gray_uint8(gt_path)
#     if pred.shape != gt.shape:
#         pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR))
#
#     p_val = psnr(gt, pred, data_range=255)
#     s_val = ssim(gt, pred, data_range=255)  # 灰度图 channel_axis=None
#     rows.append((rel, p_val, s_val))
#
# if not rows:
#     print("No matched pairs. Check PRED_ROOT / GT_ROOT."); sys.exit(1)
#
# arr = np.array(rows, dtype=object)
# mean_psnr = float(np.mean(arr[:,1].astype(float)))
# mean_ssim = float(np.mean(arr[:,2].astype(float)))
# print(f"AVG PSNR: {mean_psnr:.3f} dB  |  AVG SSIM: {mean_ssim:.4f}   (N={len(rows)})")
# if missing:
#     print(f"[WARN] {len(missing)} GT not found, e.g. {missing[:5]} ...")
#
# with open(SAVE_CSV, "w") as f:
#     f.write("rel_path,psnr,ssim\n")
#     for rel,pv,sv in rows:
#         f.write(f"{rel},{pv:.6f},{sv:.6f}\n")
#     f.write(f"__MEAN__,{mean_psnr:.6f},{mean_ssim:.6f}\n")
# print(f"Saved -> {SAVE_CSV}")
import os, sys, glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch  # <--- NEW
import lpips  # <--- NEW

# ==== 路径按你的截图修改 ====
PRED_ROOT = "exp/mpos256_ch96_s2025/image_samples/samples_mpos256_ch96_s2025/pred"
GT_ROOT = "./exp/mpos/FFA"
SAVE_CSV = "./metrics_psnr_ssim_lpips_44000.csv"  # <--- NEW (文件名修改)

# 可选：去边（0.90=只保留中心90%）
CENTER_KEEP = 1.00

# --- LPIPS模型初始化 --- # <--- NEW (整个区块)
USE_GPU = torch.cuda.is_available()
torch.cuda.empty_cache()
lpips_net = lpips.LPIPS(net='alex')  # 'alex'或'vgg'
if USE_GPU:
    lpips_net = lpips_net.cuda()
print(f"LPIPS using: {'GPU' if USE_GPU else 'CPU'}")


# --- LPIPS模型初始化结束 --- #

def center_crop_keep(img: Image.Image, keep=1.0):
    if keep >= 0.999: return img
    w, h = img.size;
    cw, ch = int(w * keep), int(h * keep)
    l = (w - cw) // 2;
    t = (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch))


# 用于PSNR/SSIM的加载函数 (保持不变)
def load_gray_uint8(path):
    im = Image.open(path).convert("L")
    im = center_crop_keep(im, CENTER_KEEP)
    return np.array(im)


# <--- NEW (新增LPIPS加载函数)
# 用于LPIPS的加载函数 (加载RGB PIL图像)
def load_rgb_pil(path):
    im = Image.open(path).convert("RGB")
    im = center_crop_keep(im, CENTER_KEEP)
    return im


# <--- NEW (结束)

pred_files = sorted([p for p in glob.glob(os.path.join(PRED_ROOT, "**", "*.*"), recursive=True)
                     if os.path.isfile(p)])

rows = []
missing = []
for p in tqdm(pred_files, desc="PSNR/SSIM/LPIPS"):  # <--- NEW (描述更新)
    rel = os.path.relpath(p, PRED_ROOT)
    gt_path = os.path.join(GT_ROOT, rel)
    if not os.path.exists(gt_path):
        missing.append(rel);
        continue

    # 1. PSNR/SSIM 路径 (灰度, uint8, numpy)
    pred_gray = load_gray_uint8(p)
    gt_gray = load_gray_uint8(gt_path)
    if pred_gray.shape != gt_gray.shape:
        # 使用gt的shape进行resize
        gt_h, gt_w = gt_gray.shape
        pred_pil_gray = Image.fromarray(pred_gray)
        pred_pil_gray_resized = pred_pil_gray.resize((gt_w, gt_h), Image.BILINEAR)
        pred_gray = np.array(pred_pil_gray_resized)

    p_val = psnr(gt_gray, pred_gray, data_range=255)
    s_val = ssim(gt_gray, pred_gray, data_range=255)

    # 2. LPIPS 路径 (RGB, [-1, 1], torch) # <--- NEW (整个区块)
    pred_rgb_pil = load_rgb_pil(p)
    gt_rgb_pil = load_rgb_pil(gt_path)

    # 确保LPIPS的输入尺寸也一致 (使用gt_rgb_pil的尺寸)
    if pred_rgb_pil.size != gt_rgb_pil.size:
        pred_rgb_pil = pred_rgb_pil.resize(gt_rgb_pil.size, Image.BILINEAR)

    # 将PIL [0,255] 转换为 Torch Tensor [-1, 1]
    pred_t = lpips.im2tensor(np.array(pred_rgb_pil))
    gt_t = lpips.im2tensor(np.array(gt_rgb_pil))

    if USE_GPU:
        pred_t, gt_t = pred_t.cuda(), gt_t.cuda()

    with torch.no_grad():
        l_val = lpips_net(pred_t, gt_t).item()
    # --- LPIPS计算结束 --- #

    rows.append((rel, p_val, s_val, l_val))  # <--- NEW (l_val添加)

if not rows:
    print("No matched pairs. Check PRED_ROOT / GT_ROOT.");
    sys.exit(1)

arr = np.array(rows, dtype=object)
mean_psnr = float(np.mean(arr[:, 1].astype(float)))
mean_ssim = float(np.mean(arr[:, 2].astype(float)))
mean_lpips = float(np.mean(arr[:, 3].astype(float)))  # <--- NEW

# <--- NEW (打印更新)
print(f"AVG PSNR: {mean_psnr:.3f} dB  |  AVG SSIM: {mean_ssim:.4f}  |  AVG LPIPS: {mean_lpips:.4f}   (N={len(rows)})")

if missing:
    print(f"[WARN] {len(missing)} GT not found, e.g. {missing[:5]} ...")

with open(SAVE_CSV, "w") as f:
    f.write("rel_path,psnr,ssim,lpips\n")  # <--- NEW (表头更新)
    for rel, pv, sv, lv in rows:  # <--- NEW (解包更新)
        f.write(f"{rel},{pv:.6f},{sv:.6f},{lv:.6f}\n")  # <--- NEW (写入更新)
    # <--- NEW (均值行更新)
    f.write(f"__MEAN__,{mean_psnr:.6f},{mean_ssim:.6f},{mean_lpips:.6f}\n")
print(f"Saved -> {SAVE_CSV}")