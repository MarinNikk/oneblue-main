#!/usr/bin/env bash
# Instalacija ROS 2 Jazzy (Ubuntu 24.04 Noble) + ovisnosti za čvor.
# Pokreni SA sudo:   sudo bash marin/ros/install_ros2_jazzy.sh
# Traje nekoliko minuta i skida ~2-3 GB (desktop uključuje RViz2).
set -e
export DEBIAN_FRONTEND=noninteractive

if [ "$(id -u)" -ne 0 ]; then
  echo "Pokreni sa sudo:  sudo bash $0"; exit 1
fi

echo "== 1/5  locale (UTF-8) =="
apt update
apt install -y locales
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "== 2/5  Universe repo + alati =="
apt install -y software-properties-common curl
add-apt-repository -y universe

echo "== 3/5  ROS 2 apt izvor (službeni ros-apt-source paket) =="
ROS_APT=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F '"tag_name"' | awk -F'"' '{print $4}')
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT}/ros2-apt-source_${ROS_APT}.${CODENAME}_all.deb"
apt install -y /tmp/ros2-apt-source.deb
apt update
apt upgrade -y

echo "== 4/5  ROS 2 Jazzy (desktop: rclpy, nav_msgs, geometry_msgs, RViz2, rosbag2) =="
apt install -y ros-jazzy-desktop

echo "== 5/5  ovisnost čvora: numpy za sistemski Python 3.12 =="
apt install -y python3-numpy

echo
echo "GOTOVO. Dalje (kao običan korisnik, NE sudo):"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  cd marin/ros && bash run_test.sh          # cache je već izrađen"
echo "  # vizualizacija:  rviz2   (Fixed Frame: map; Map=/ocean/currents, PoseArray=/ocean/current_vectors)"
