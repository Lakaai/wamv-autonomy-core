# About

## Launch Instructions

### Onboard WAMV:

Launch the scan_code_pkg by running:

```shell
. install/setup.bash
ros2 run scan_code_pkg read_gazebo_frame
```

In a seperate terminal view the output on /code_sequence ROS topic

```shell
. install/setup.bash
ros2 topic echo /code_sequence
```


### Inside VRX:

Open terminal and run the following:

```shell
. install/setup.bash
ros2 launch vrx_gz competition.launch.py world:=scan_dock_deliver_task
```

In a seperate terminal launch the scan_code_pkg by running:

```shell
. install/setup.bash
ros2 run scan_code_pkg read_gazebo_frame
```

In a seperate terminal view the output on /code_sequence ROS topic

```shell
. install/setup.bash
ros2 topic echo /code_sequence
```


