
# scan-the-code (RobotX)

This repository contains the source code and documentation for a solution to the "scan the code" task for the Maritime RobotX Challenge. This solution can be easily deployed and tested in the VRX simulation environment, which supports simulation of unmanned surface vehicles in marine environments.

### Building the Workspace 

Start with a clean workspace by removing previous build files

```bash
# Remove any prior build directories
cd ~/vrx_ws
rm -rf build/ install/ log/

# Build the new workspace
cd ~/vrx_ws
colcon build --merge-install

# Source ROS2
source /opt/ros/humble/setup.bash  # or whatever ROS2 version you're using

# Source the VRX workspace
source ~/vrx_ws/install/setup.bash  # adjust path if different
```
Enable GPU Acceleration (WSL2 Only)

```bash
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```
