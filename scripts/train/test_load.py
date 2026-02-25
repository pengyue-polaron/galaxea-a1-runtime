from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset = LeRobotDataset(
    repo_id="local-data",
    root="/home/jolia/DataCoach/data/formatted_data/test",
)
breakpoint()
print(dataset[700])
