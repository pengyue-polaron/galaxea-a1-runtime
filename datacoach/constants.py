
#Lerobot communication 
SEND_IP = "127.0.0.1" 
DEFAULT_A1_USB_DEVICE = "/dev/ttyACM1"
DEFAULT_LEROBOT_USB_DEVICE = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
DEFAULT_TELEOP_ID = "my_awesome_leader_arm"


# training
DEFAULT_POLICY_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 32
DEFAULT_STEPS = 50000
DEFAULT_OPTIMIZER_LR = 1e-4
DEFAULT_WANDB_ENABLE = False

ZMQ_CMD_PORT = 5556  # commanded state port
ZMQ_STATE_PORT = 5557  # state port
ZMQ_CAM_PORT = 5558  # camera stream port
ZMQ_POLICY_ACTION_PORT = 5559  # policy action output port

# data collection
ROBOT_FPS = 50  # Hz
CAM_FPS = 20  # Hz
