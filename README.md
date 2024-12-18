
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

### Hardware Acceleration (WSL2)

On a device with multiple GPUs the WSL instance will select the first enumerated GPU by default. 
To choose a specific GPU, set the environment variable below to the name of your GPU as it appears in device manager:

```bash
export MESA_D3D12_DEFAULT_ADAPTER_NAME="<NameFromDeviceManager>"
```
This will do a string match, so if you set it to "NVIDIA" it will match the first GPU that starts with "NVIDIA".



The MESA utility glxinfo can be used to determine which GPU is currently used. For example, below the NVIDIA GPU is being used.

```bash
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
glxinfo -B
```