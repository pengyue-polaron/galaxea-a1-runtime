FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    ca-certificates \
    curl \
    git \
    iproute2 \
    iputils-ping \
    less \
    libconsole-bridge0.4 \
    libeigen3-dev \
    liboctomap-dev \
    liborocos-kdl1.4 \
    python3-catkin-tools \
    python3-pip \
    python3-rosdep \
    python3-serial \
    python3-vcstool \
    rsync \
    sudo \
    tmux \
    udev \
    usbutils \
    vim \
    ros-noetic-controller-interface \
    ros-noetic-controller-manager \
    ros-noetic-joint-state-controller \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-kdl-parser \
    ros-noetic-pluginlib \
    ros-noetic-robot-state-publisher \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-rqt-tf-tree \
    ros-noetic-rviz \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-ros \
    ros-noetic-trac-ik-lib \
    ros-noetic-urdf \
    ros-noetic-urdf-parser-plugin \
    ros-noetic-xacro \
    && rm -rf /var/lib/apt/lists/*

COPY docker/a1_noetic_arm64/entrypoint.sh /usr/local/bin/a1-noetic-entrypoint.sh
RUN chmod +x /usr/local/bin/a1-noetic-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/a1-noetic-entrypoint.sh"]
CMD ["bash"]
