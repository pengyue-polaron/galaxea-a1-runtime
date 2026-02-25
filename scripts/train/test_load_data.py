from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="OpenGalaxea/Galaxea-Open-World-Dataset",
    split="train",
)

sample = dataset[0]

print(sample.keys())
