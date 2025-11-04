# eval_fid_kid.py —— 按类别分别计算 FID/KID；只按预测集合匹配 GT；可选 val.txt
import os, glob, shutil, csv, json
from PIL import Image

# 减少 MKL 告警
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# ====== 按你的目录改这里 ======
PRED_ROOT = "exp/mpos256_ch96_s2025/image_samples/samples_mpos256_ch96_s2025/pred"  # 预测图（含子目录）
GT_ROOT   = "./exp/mpos/FFA"                                                # GT 根目录（与预测相对路径一致）
WORK_DIR  = "./_fidkid_32000"                                                # 临时输出目录
VAL_SPLIT = "./exp/mpos/splits/val.txt"
# 手动指定类别顺序（可留空自动识别）。写成目录名：如 ["0_Normal","1_DR","2_RVO","3_AMD","5_CSC"]
CLASS_DIRS = []  # [] = 自动从 PRED_ROOT 首层目录推断
# =================================

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
FLAT_PRED = os.path.join(WORK_DIR, "pred_flat")
FLAT_GT   = os.path.join(WORK_DIR, "gt_flat")

def to_rgb_save(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.open(src).convert("RGB").save(dst)

def load_rel_list_from_pred(root):
    files = sorted([p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                    if os.path.isfile(p) and p.lower().endswith(EXTS)])
    return [os.path.relpath(p, root).replace("\\", "/") for p in files]

def load_rel_list_from_split(split_path):
    rels = []
    with open(split_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            s = s.replace("\\", "/")
            rels.append(s if "." in os.path.basename(s) else (s + ".png"))
    return rels

def infer_class_dirs_from_pred(rel_list):
    cls = []
    for r in rel_list:
        head = r.split("/", 1)[0]
        if head not in cls:
            cls.append(head)
    return cls

def calc_fid_kid(dir_pred, dir_gt, kid_subset):
    from torch_fidelity import calculate_metrics
    metrics = calculate_metrics(
        input1=dir_pred, input2=dir_gt, cuda=True,
        isc=False, fid=True, kid=True,
        kid_subset_size=kid_subset, verbose=False
    )
    # 关键字段名（不同版本 torch-fidelity 一致）
    fid = float(metrics.get("frechet_inception_distance", float("nan")))
    kid_mean = float(metrics.get("kernel_inception_distance_mean", float("nan")))
    kid_std  = float(metrics.get("kernel_inception_distance_std", float("nan")))
    return fid, kid_mean, kid_std

def flatten_for_rel_list(rel_list, cls_dir=None):
    """
    将 rel_list 中属于 cls_dir 的样本复制到扁平目录。
    若 cls_dir=None 则复制所有（用于整体）。
    返回: (pred_dir, gt_dir, n_pred, n_gt)
    """
    # 目标扁平目录（可按类别放到子目录）
    tag = "all" if cls_dir is None else cls_dir
    pred_out = os.path.join(FLAT_PRED, tag)
    gt_out   = os.path.join(FLAT_GT, tag)
    shutil.rmtree(pred_out, ignore_errors=True)
    shutil.rmtree(gt_out,   ignore_errors=True)
    os.makedirs(pred_out, exist_ok=True)
    os.makedirs(gt_out,   exist_ok=True)

    n_pred = n_gt = 0
    missing_gt = []
    for rel in rel_list:
        if cls_dir is not None and not rel.startswith(cls_dir + "/"):
            continue
        p_src = os.path.join(PRED_ROOT, rel)
        g_src = os.path.join(GT_ROOT,   rel)
        if os.path.exists(p_src):
            to_rgb_save(p_src, os.path.join(pred_out, rel.replace("/", "_")))
            n_pred += 1
        if os.path.exists(g_src):
            to_rgb_save(g_src, os.path.join(gt_out,   rel.replace("/", "_")))
            n_gt += 1
        else:
            missing_gt.append(rel)

    return pred_out, gt_out, n_pred, n_gt, missing_gt

def pretty_name(dir_name):
    # 把 "0_Normal" -> "Normal"；"1_DR"->"DR"
    base = dir_name.split("/", 1)[0]
    if "_" in base:
        return base.split("_", 1)[1]
    return base

def main():
    # 1) 收集预测相对路径
    if VAL_SPLIT:
        rel_list = load_rel_list_from_split(VAL_SPLIT)
        pred_rels = set(load_rel_list_from_pred(PRED_ROOT))
        rel_list = [r for r in rel_list if r in pred_rels]
        print(f"Using split: {VAL_SPLIT}; matched predictions: {len(rel_list)}")
    else:
        rel_list = load_rel_list_from_pred(PRED_ROOT)
        print(f"Using predictions only: {len(rel_list)}")

    if not rel_list:
        print("❌ No files found. Check PRED_ROOT / VAL_SPLIT.")
        return

    # 2) 推断类别
    class_dirs = CLASS_DIRS or infer_class_dirs_from_pred(rel_list)
    print("Classes:", class_dirs)

    # 3) 清理临时目录
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(FLAT_PRED, exist_ok=True)
    os.makedirs(FLAT_GT,   exist_ok=True)

    rows = []
    total_pairs = 0

    # 3.1 先计算 Overall
    pred_all, gt_all, n_pred_all, n_gt_all, miss_all = flatten_for_rel_list(rel_list, cls_dir=None)
    if n_pred_all and n_gt_all:
        kid_subset = min(1000, n_pred_all, n_gt_all)
        fid, kid_m, kid_s = calc_fid_kid(pred_all, gt_all, kid_subset)
        rows.append(["OVERALL", n_pred_all, n_gt_all, kid_subset, fid, kid_m, kid_s])
        total_pairs = min(n_pred_all, n_gt_all)
        print(f"[OVERALL] pairs={total_pairs}  FID={fid:.6f}  KID={kid_m:.6f} ± {kid_s:.6f}")
        if miss_all:
            print(f"[WARN][OVERALL] Missing GT for {len(miss_all)} files (e.g. {miss_all[:5]})")
    else:
        print("[OVERALL] Not enough images after flattening.")

    # 3.2 再逐类
    for cls in class_dirs:
        pred_dir, gt_dir, n_pred, n_gt, missing_gt = flatten_for_rel_list(rel_list, cls_dir=cls)
        if n_pred == 0 or n_gt == 0:
            print(f"[{cls}] skip: n_pred={n_pred}, n_gt={n_gt}")
            rows.append([pretty_name(cls), n_pred, n_gt, 0, float('nan'), float('nan'), float('nan')])
            continue
        kid_subset = min(1000, n_pred, n_gt)
        fid, kid_m, kid_s = calc_fid_kid(pred_dir, gt_dir, kid_subset)
        rows.append([pretty_name(cls), n_pred, n_gt, kid_subset, fid, kid_m, kid_s])
        print(f"[{cls}] pairs={min(n_pred,n_gt)}  FID={fid:.6f}  KID={kid_m:.6f} ± {kid_s:.6f}")
        if missing_gt:
            print(f"[WARN][{cls}] Missing GT for {len(missing_gt)} files (e.g. {missing_gt[:5]})")

    # 4) 保存 CSV/JSON
    csv_path  = os.path.join(WORK_DIR, "metrics_fidkid_per_class.csv")
    json_path = os.path.join(WORK_DIR, "metrics_fidkid_per_class.json")
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_pred", "n_gt", "kid_subset", "FID", "KID_mean", "KID_std"])
        for r in rows:
            w.writerow(r)
    with open(json_path, "w") as f:
        json.dump(
            [{"class": r[0], "n_pred": r[1], "n_gt": r[2], "kid_subset": r[3],
              "FID": r[4], "KID_mean": r[5], "KID_std": r[6]} for r in rows],
            f, indent=2
        )
    print(f"\nSaved:\n  {csv_path}\n  {json_path}")

if __name__ == "__main__":
    main()
