#!/usr/bin/env bash
# Jednokratni test: pokrene izdavača, snimi rosbag i zapiše sažetak.
# Preduvjet:  source /opt/ros/jazzy/setup.bash   (i napravljena cache/fields.npz)
# Pokretanje:  bash run_test.sh
set -e
: "${ROS_DISTRO:?Prvo:  source /opt/ros/jazzy/setup.bash}"
cd "$(dirname "$0")"
mkdir -p tested
rm -rf tested/adriatic_bag

pkill -f current_publisher.py 2>/dev/null || true   # ukloni eventualne zaostale izdavače
sleep 1

echo "[1/3] pokrećem izdavača (period 0.5 s)..."
python3 current_publisher.py --ros-args -p period_s:=0.5 &
PUB=$!
trap 'kill -9 $PUB 2>/dev/null; pkill -f current_publisher.py 2>/dev/null || true' EXIT
sleep 3

echo "[2/3] snimam rosbag (8 s)..."
timeout 8 ros2 bag record -o tested/adriatic_bag \
    /ocean/currents /ocean/current_vectors >/dev/null 2>&1 || true

echo "[3/3] hvatam sažetak (do 15 s)..."
timeout 15 python3 field_listener.py || true

echo
echo "GOTOVO. Artefakti u marin/ros/tested/:"
ls -la tested
[ -f tested/summary.json ] && { echo; echo "--- summary.json ---"; cat tested/summary.json; }
