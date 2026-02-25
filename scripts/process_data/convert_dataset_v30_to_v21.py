#!/usr/bin/env python3
"""
快速转换工具：LeRobot v3.0 → v2.1

这个脚本可以单独运行，不会影响你的主环境。

用法：
    # 方法1：直接运行（会自动创建临时环境）
    python3 convert_v30_to_v21_simple.py \\
        --input /path/to/v30/dataset \\
        --output /path/to/v21/dataset
    
    # 方法2：如果已经有lerobot 0.4.3环境
    source ~/.venv_data/bin/activate
    python3 convert_v30_to_v21_simple.py \\
        --input /path/to/v30/dataset \\
        --output /path/to/v21/dataset \\
        --skip-env-setup
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


def check_lerobot():
    """检查是否已安装lerobot 0.4.3"""
    try:
        import lerobot
        version = lerobot.__version__
        if version >= "0.4.0":
            print(f"✓ Found lerobot {version}")
            return True
        else:
            print(f"✗ Found lerobot {version}, but need >= 0.4.0")
            return False
    except ImportError:
        print("✗ lerobot not found")
        return False


def setup_environment():
    """设置Python环境"""
    env_path = Path.home() / ".venv_lerobot_v30_converter"
    
    if env_path.exists():
        print(f"Using existing environment: {env_path}")
    else:
        print(f"Creating new environment: {env_path}")
        subprocess.run([sys.executable, "-m", "venv", str(env_path)], check=True)
        
        pip = env_path / "bin" / "pip"
        print("Installing dependencies...")
        subprocess.run([str(pip), "install", "-q", "lerobot==0.4.3", "pandas", "pyarrow", "numpy"], check=True)
    
    return env_path / "bin" / "python"


def convert_dataset_direct(input_path: Path, output_path: Path):
    """
    直接转换（当前环境已有lerobot 0.4.3时使用）
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    print(f"\n📂 Loading v3.0 dataset from: {input_path}")
    dataset = LeRobotDataset(str(input_path))
    
    print(f"✓ Loaded {len(dataset)} frames")
    
    # 创建v2.1目录结构
    output_path = Path(output_path)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(parents=True, exist_ok=True)
    
    # 检测相机keys
    camera_keys = [k for k in dataset.hf_dataset.features.keys() if 'image' in k.lower()]
    for key in camera_keys:
        camera_name = key.split('.')[-1]
        (output_path / "videos" / "chunk-000" / camera_name).mkdir(parents=True, exist_ok=True)
    
    # 获取episode索引
    episode_index = dataset.episode_data_index
    num_episodes = len(episode_index["from"])
    
    print(f"\n🔄 Converting {num_episodes} episodes...")
    
    episodes_metadata = []
    
    for ep_idx in range(num_episodes):
        from_idx = int(episode_index["from"][ep_idx])
        to_idx = int(episode_index["to"][ep_idx])
        ep_length = to_idx - from_idx
        
        print(f"  Episode {ep_idx+1}/{num_episodes} ({ep_length} frames)    ", end='\r')
        
        # 收集该episode的所有数据
        episode_data = {}
        
        for frame_idx in range(from_idx, to_idx):
            frame = dataset[frame_idx]
            
            for key, value in frame.items():
                # 跳过图像数据（太大）和字符串
                if 'image' in key.lower() or isinstance(value, str):
                    continue
                
                if key not in episode_data:
                    episode_data[key] = []
                
                # 转换为numpy
                if hasattr(value, 'numpy'):
                    episode_data[key].append(value.numpy())
                else:
                    episode_data[key].append(np.array(value))
        
        # 转换为numpy数组
        for key in episode_data:
            episode_data[key] = np.stack(episode_data[key])
        
        # 保存为parquet（v2.1格式）
        df_data = {}
        for key, arr in episode_data.items():
            if arr.ndim == 1:
                df_data[key] = arr
            else:
                # 多维数组转为list
                df_data[key] = [arr[i] for i in range(len(arr))]
        
        df = pd.DataFrame(df_data)
        parquet_path = output_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, engine='pyarrow')
        
        # 记录元数据
        task = episode_index.get("task", [""])[ep_idx] if "task" in episode_index else ""
        episodes_metadata.append({
            "episode_index": ep_idx,
            "length": ep_length,
            "task": task if isinstance(task, str) else str(task),
        })
    
    print(f"\n✓ Converted {num_episodes} episodes                    ")
    
    # 保存元数据
    meta = dataset.meta
    info = {
        "codebase_version": "v2.1",
        "fps": int(meta.fps),
        "video": meta.video,
        "total_episodes": num_episodes,
        "total_frames": len(dataset),
        "robot_type": getattr(meta, 'robot_type', 'unknown'),
    }
    
    with open(output_path / "meta" / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    with open(output_path / "meta" / "episodes.jsonl", 'w') as f:
        for ep in episodes_metadata:
            f.write(json.dumps(ep) + '\n')
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Format: v2.1")
    print(f"📦 Episodes: {num_episodes}")
    print(f"🎬 Frames: {len(dataset)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 dataset to v2.1 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-setup environment and convert
  python3 convert_v30_to_v21_simple.py \\
      --input ~/datasets/my_v30_dataset \\
      --output ~/datasets/my_v21_dataset
  
  # Use existing lerobot 0.4.3 environment
  source ~/.venv_data/bin/activate
  python3 convert_v30_to_v21_simple.py \\
      --input ~/datasets/my_v30_dataset \\
      --output ~/datasets/my_v21_dataset \\
      --skip-env-setup
"""
    )
    
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Path to v3.0 dataset directory")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output path for v2.1 dataset")
    parser.add_argument("--skip-env-setup", action="store_true",
                        help="Skip environment setup (use current environment)")
    
    args = parser.parse_args()
    
    # 验证输入
    if not args.input.exists():
        print(f"❌ Error: Input path not found: {args.input}")
        sys.exit(1)
    
    # 检查是否是v3.0格式
    if not (args.input / "meta" / "episodes").exists():
        print(f"⚠️  Warning: This doesn't look like a v3.0 dataset")
        print(f"   Expected to find: {args.input / 'meta' / 'episodes'}")
        response = input("   Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 转换
    if args.skip_env_setup:
        if not check_lerobot():
            print("\n❌ lerobot >= 0.4.0 not found in current environment")
            print("   Install it with: pip install lerobot==0.4.3")
            print("   Or remove --skip-env-setup to auto-create environment")
            sys.exit(1)
        
        convert_dataset_direct(args.input, args.output)
    else:
        # 创建独立环境并运行
        python_exe = setup_environment()
        
        # 重新运行脚本，但使用新环境
        script_path = Path(__file__).absolute()
        subprocess.run([
            str(python_exe),
            str(script_path),
            "--input", str(args.input),
            "--output", str(args.output),
            "--skip-env-setup"
        ], check=True)


if __name__ == "__main__":
    main()