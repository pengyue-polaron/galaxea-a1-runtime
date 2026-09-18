# Build the ROS 1 client libraries from pinned releases in an isolated image.
# The ROS 2 side is Jazzy, matching cameras and Foxglove (no cross-distro DDS).
FROM ros:jazzy-ros-base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config python3-venv python3-rospkg-modules \
    python3-vcstool libboost-all-dev libconsole-bridge-dev liblz4-dev libbz2-dev \
    libtinyxml2-dev libpoco-dev libgpgme-dev libssl-dev \
    ros-jazzy-diagnostic-msgs ros-jazzy-tf2-msgs ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
# ROS 1 generators require Empy 3; keep this build interpreter out of Jazzy's env.
RUN python3 -m venv --system-site-packages /opt/ros1-build-python \
    && /opt/ros1-build-python/bin/pip install --no-cache-dir empy==3.3.4
COPY docker/ros1-bridge/ros1.rosinstall /opt/ros1.rosinstall
RUN mkdir -p /opt/ros1_ws/src \
    && vcs import --input /opt/ros1.rosinstall /opt/ros1_ws/src
RUN cd /opt/ros1_ws && env -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
    /opt/ros1-build-python/bin/python src/catkin/bin/catkin_make_isolated \
    --install --install-space /opt/ros1 --cmake-args \
    -DPYTHON_EXECUTABLE=/opt/ros1-build-python/bin/python \
    -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=OFF -DROSCONSOLE_BACKEND=print \
    --make-args -j2
RUN git clone https://github.com/ros2/ros1_bridge.git /opt/bridge_ws/src/ros1_bridge \
    && git -C /opt/bridge_ws/src/ros1_bridge checkout 611755fd917285316051cbea80507e8b2f6b7ec1
RUN source /opt/ros1/setup.bash && source /opt/ros/jazzy/setup.bash \
    && cd /opt/bridge_ws && MAKEFLAGS=-j2 colcon build --executor sequential \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
COPY galaxea_a1_runtime/runtime/native/observation_bridge /opt/a1_observation_bridge
RUN source /opt/ros1/setup.bash && source /opt/ros/jazzy/setup.bash \
    && source /opt/bridge_ws/install/local_setup.bash \
    && CMAKE_PREFIX_PATH="/opt/ros/jazzy:/opt/bridge_ws/install/ros1_bridge:/opt/ros1" cmake -S /opt/a1_observation_bridge -B /opt/a1_observation_bridge/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/a1_observation_bridge/build -j2 \
    && cmake --install /opt/a1_observation_bridge/build
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros1/setup.bash && source /opt/ros/jazzy/setup.bash && source /opt/bridge_ws/install/local_setup.bash && exec /usr/local/bin/a1_observation_bridge \"$@\"", "a1-observation-bridge"]
