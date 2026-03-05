import numpy as np
from datacoach.training import config as _config
from datacoach.training import policy_config
from openpi.shared import download

# helper 也在 scripts/train/a1_policy.py
from scripts.train.a1_policy import make_a1_example

# 1. 选个 A1 训练 config（与训练时用的一致）
cfg = _config.get_config("localdata_a1_pi05")   
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

policy = policy_config.create_trained_policy(cfg, ckpt)

# 2. 构造输入，make_a1_example 会生成合规的空白样本
example = make_a1_example()
example["prompt"] = "pick up box"  

# 3. 运行并查看动作
out = policy.infer(example)
print("actions:", out["actions"])   # [8] 或 [T,8]，A1Outputs 会裁到前 8 维