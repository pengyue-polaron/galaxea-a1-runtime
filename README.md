# DataCoach
> Using a Vision-Language Model as a Coach to Guide Next-Round Data Collection

**DataCoach** explores how a **Vision-Language Model (VLM)** can act as a **coach** to guide a robot in iterative data collection.  
Instead of passively collecting data, the robot receives **feedback and guidance** from the VLM after each trial, improving the quality and diversity of the next round of demonstrations.

If you have already set up the workspace before, you can directly jump to (## Teleoperation with LeRobot).


## Setup
```bash
git clone --recurse-submodules https://github.com/joliachen/DataCoach.git
# Or if you already cloned the repo:
git submodule update --init --recursive
cd DataCoach
cd third_party/lerobot
git lfs pull
cd ../..
```

###
Install A1
Follow [guidance](./ros_workspace/README.md) to configurate A1 SDK. 

### Set up pi 0.5 environment

If uv is already set up, skip below commands:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'source $HOME/.local/bin/env' >> ~/.bashrc
source ~/.bashrc
```
run the following to set up the uv environment for openpi:
```bash
# cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ./third_party/openpi # import openpi in the root uv

# If you use PyTorch for Linux x86_64 and Linux SBSA on NVIDIA 5080, 5090 Blackwell RTX GPUs, run the command below.
source /home/jolia/DataCoach/.venv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv sync

```
```bash
UV_HT。TP_TIMEOUT=120 GIT_LFS_SKIP_SMUDGE=1 uv sync

```


`Note: Since there is confliction between lerobot packages and openpi packages, we use uv to manage pacakges of openpi, which is independent from the lerobot conda environment.

Set up conda environment for datacoach project
```bash
cd lerobot
conda create -y -n datacoach python=3.10
conda activate datacoach
pip install -e .

pip install lerobot feetech-servo-sdk scipy placo pyrealsense2 hydra-core zmq scservo_sdk
conda install -c conda-forge ffmpeg opencv

conda config --set auto_activate_base false # turn off activating conda env automatically
```

## Data collection with LeRobot

```bash
conda activate datacoach
lerobot-find-port # ttyACM0
# If you have not set up index for motor, you should set up for once
lerobot-setup-motors --teleop.type=so101_leader --teleop.port='/dev/ttyACM2'

sudo chmod 777 /dev/ttyACM0 # port for lerobot
sudo chmod 777 /dev/ttyACM1 # port for A1
```

```bash
# in a new terminal, with datacoach conda environment deactivated
source /home/jolia/DataCoach/ros_workspace/scripts/setup_a1.sh
#or 
a1env # if you set alias 
roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:="/dev/ttyACM1"
```
it will print out:
```bash
[INFO] [1760946372.464759096]: Initialization Complete
[INFO] [1760946372.464957949]: Launch feedback and control thread
[INFO] [1760946372.464999897]: Bringup all
****CANNOT PASS CRC16 CHECK!!!****
****CANNOT PASS CRC16 CHECK!!!****
```

If it print out ` [ERROR] [1760946263.450528882]: Serial interface error: Driver Feedback: Serial Read/Write Fault `, it can be the port issue (might not be /dev/ttyACM1 can be /dev/ttyACM0)

```bash
# open a new terminal
a1env
roslaunch mobiman eeTrackerdemo.launch
```

# In a new terminal
```bash
a1env
cd scripts/collect_data
python run_a1_server.py # which start teleoperating A1

# In a new terminal,
conda activate datacoach
python run_data_services.py 
```

Under `configs/lerobot/collect_data.yaml`, specify your task name and demo index everytime before you start to collect a new demo.
```bash
# In a new terminal,
conda activate datacoach
python run_data_collection.py 

```


## Data preprocessing

Remember in `configs/lerobot/process_data.yaml` change your task_name and data path.

```bash
conda activate datacoach
cd scripts/process_data
python align_timestamps.py
python convert_data_to_lerobot.py
```


## Training
```bash
# register openai package into current env

cd ../train
uv run compute_norm_stats.py --config-name localdata_a1_pi05
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run train.py localdata_a1_pi05 --exp-name=my_experiment --overwrite

```




sudo -E bash -c "source /opt/ros/noetic/setup.bash && source /home/nyush_robo/DataCoach/ros_workspace/ros_ws/src/a1_sdk/install/setup.bash && python3 /home/nyush_robo/DataCoach/receiver.py