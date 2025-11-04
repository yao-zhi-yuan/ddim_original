from pathlib import Path
import random
from collections import defaultdict

# ====== 配置部分 ======
root = Path("/home/yzy/ddim/exp/mpos")   # 👈 改成你的数据根目录
cfp_dir = root / "CFP"
ffa_dir = root / "FFA"
splits_dir = root / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)

# 设置随机种子（保证复现）
seed = 2025
random.seed(seed)

# 训练/验证比例（剩下的作为验证）
train_ratio = 0.85

# 支持的图片扩展名
exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ====== 收集所有匹配图像 ======
pairs = []  # [(rel_path, class_name)]
for p in cfp_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        rel = p.relative_to(cfp_dir)
        if (ffa_dir / rel).is_file():
            # 类别名取上一级文件夹名（如 1_DR/269.png -> 类别=1_DR）
            class_name = rel.parts[0]
            pairs.append((str(rel).replace("\\", "/"), class_name))

if not pairs:
    raise SystemExit("❌ 没有找到任何匹配的图像对，请检查 CFP/ 与 FFA/ 目录结构。")

print(f"发现配对图像数: {len(pairs)}")
classes = sorted({cls for _, cls in pairs})
print(f"检测到类别: {classes}")

# ====== 按类别分层随机划分 ======
by_class = defaultdict(list)
for rel, cls in pairs:
    by_class[cls].append(rel)

train_list, val_list = [], []
for cls, rels in by_class.items():
    random.shuffle(rels)
    n_total = len(rels)
    n_train = int(round(n_total * train_ratio))
    train_list.extend(rels[:n_train])
    val_list.extend(rels[n_train:])
    print(f"类 {cls}: 共 {n_total} 张 -> train {n_train}, val {n_total - n_train}")

# ====== 写入文件 ======
train_path = splits_dir / "train.txt"
val_path = splits_dir / "val.txt"

train_path.write_text("\n".join(sorted(train_list)) + "\n", encoding="utf-8")
val_path.write_text("\n".join(sorted(val_list)) + "\n", encoding="utf-8")

print(f"✅ 写入 {train_path} ({len(train_list)} 条)")
print(f"✅ 写入 {val_path} ({len(val_list)} 条)")
print(f"随机种子 seed={seed}")
