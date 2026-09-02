FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
    -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
    -e 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/|g' \
    /etc/apt/sources.list \
    && find /etc/apt -type f -name '*.list' -exec sed -i \
      's|http://packages.ros.org/ros/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu|g' {} +

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    ca-certificates \
    curl \
    git \
    iproute2 \
    iputils-ping \
    less \
    libconsole-bridge0.4 \
    libasio-dev \
    libeigen3-dev \
    liboctomap-dev \
    liborocos-kdl1.4 \
    libssl-dev \
    libwebsocketpp-dev \
    nlohmann-json3-dev \
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
    ros-noetic-diagnostic-msgs \
    ros-noetic-joint-state-controller \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-kdl-parser \
    ros-noetic-nodelet \
    ros-noetic-pluginlib \
    ros-noetic-resource-retriever \
    ros-noetic-robot-state-publisher \
    ros-noetic-ros-babel-fish \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-rqt-tf-tree \
    ros-noetic-rviz \
    ros-noetic-std-srvs \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-ros \
    ros-noetic-trac-ik-lib \
    ros-noetic-urdf \
    ros-noetic-urdf-parser-plugin \
    ros-noetic-xacro \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ROS 1 is EOL and its binary Foxglove package is outdated. Build the reviewed
# maintenance revision from source for both amd64 development and A1 arm64.
ARG FOXGLOVE_BRIDGE_REVISION=42c0ba4fffeec88dc6890dd98ea2590e20bd4370
RUN mkdir -p /opt/foxglove_bridge_ws/src \
    && git clone https://github.com/foxglove/ros-foxglove-bridge.git \
      /opt/foxglove_bridge_ws/src/foxglove_bridge \
    && git -C /opt/foxglove_bridge_ws/src/foxglove_bridge checkout \
      "${FOXGLOVE_BRIDGE_REVISION}" \
    && . /opt/ros/noetic/setup.sh \
    && cd /opt/foxglove_bridge_ws \
    && catkin_make install \
      -DCMAKE_BUILD_TYPE=Release \
      -DCATKIN_ENABLE_TESTING=OFF \
    && rm -rf build devel src

# Install Python 3.12 and the minimal image-encoding dependencies used by
# repository ROS sidecars. Versions match the locked main environment.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv python install 3.12 \
    && /root/.local/bin/uv venv --python 3.12 /opt/a1-python \
    && /root/.local/bin/uv pip install --python /opt/a1-python/bin/python \
      msgpack==1.2.1 \
      numpy==2.2.6 \
      opencv-python-headless==4.11.0.86
ENV PATH="/opt/a1-python/bin:/root/.local/bin:${PATH}"

# Patch rosgraph/roslogging.py for Python 3.12+ (infinite loop in findCaller)
RUN sed -i 's/while hasattr(f, "f_code"):/max_depth = 20\n        while hasattr(f, "f_code") and max_depth > 0:\n            max_depth -= 1/' \
    /opt/ros/noetic/lib/python3/dist-packages/rosgraph/roslogging.py

COPY docker/a1-noetic-arm64/entrypoint.sh /usr/local/bin/a1-noetic-entrypoint.sh
RUN chmod +x /usr/local/bin/a1-noetic-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/a1-noetic-entrypoint.sh"]
CMD ["bash"]
