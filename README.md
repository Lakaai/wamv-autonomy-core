
# WAMV Autonomy Core

This solution can be found by switching to the "scan-the-code" branch.

### Tests 
```bash
# Run pytest in verbose mode 
pytest -v PYTHONPATH=src
```

### Common VRX Commands

```bash
# Launch the simulation
ros2 launch vrx_gz competition.launch.py world:=sydney_regatta
```

```bash
. install/setup.bash
```

```bash
ros2 launch vrx_gz usv_joy_teleop.py
```