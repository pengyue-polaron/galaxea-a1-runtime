FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-diagnostic-msgs \
    ros-jazzy-realsense2-camera \
    ros-jazzy-message-filters \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport-plugins \
    ros-jazzy-rosbag2-storage-mcap \
    python3-numpy python3-opencv python3-msgpack python3-yaml \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/workspace:/workspace/external/embodied-ops/src
WORKDIR /workspace
