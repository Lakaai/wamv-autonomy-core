# About

This folder contains several python scripts that are used to solve the Scan the Code tasks in the RobotX competition. The custom models that have been trained to detect the RGB flashing display can be found [here](). These models were trained using this [dataset](https://universe.roboflow.com/matrix-uf83o/matrix-model).

## Usage

### Linux OS: 

It is highly encouraged to use any of these scripts inside a python [virtual environment](https://docs.python.org/3/library/venv.html). Virtual environments keep the dependencies and libraries you install for one project separate from those of other projects in order to avoid package conflicts. 

To setup a python virtual environment on a linux OS, first verify the venv installation by running the following command in terminal: 

```bash
sudo apt install <python-version>-venv
```

For example: 
```bash
sudo apt install python3.10-venv
```

Then create the virtual environment: 

```bash
python -m venv <venv-name>
```

For example: 
```bash 
python -m venv venv
```

Now activate the virtual environment:

```bash
source <venv-name>/bin/activate # On macOS/Linux
```

For example: 
```bash
source venv/bin/activate # On macOS/Linux
```

# Dependencies 

After creating and activating the virtual environment you may now install any of the required packages using the pip installer. 

```bash
pip install <package-name>
```
The header of each python script should provide a comprehensive list of the packages required to run that script. 

Please update the header appropriately to accomodate for any changes that you make.  

numpy==1.26.4 opencv-python ultralytics

## Spinnaker Blackfly
https://github.com/Teledyne-MV/Spinnaker-Examples

### PySpin Installation

```bash 
pip install spinnaker_python-4.0.0.116-cp310-cp310-linux_x86_64.whl
```

# YOLOv8 

YOLOv8 is the primary object detection model used in this implementation of colour detection scripts. To learn more see [YOLOv8 by Ultralytics](https://docs.ultralytics.com/models/yolov8/).

# OpenVINO 

[OpenVINO](https://github.com/openvinotoolkit/openvino) is a toolkit developed by Intel that allows developers to optimize and deploy deep learning models for various Intel hardware platforms. In order to maximise performance with OpenVINO by utilising an Intel integrated GPU the correct graphics drivers must be properly configured on the system. To install these drivers carefully follow these [instructions](https://docs.openvino.ai/2023.3/openvino_docs_install_guides_configurations_for_intel_gpu.html).

# RGB Matrix

Components:

Adafruit 64x32 RGB LED Matrix - 4mm Pitch
Adafruit LED Matrix RGB HAT RTC RASP PI
AC/DC DESKTOP ADAPTER 5V 50W

Setup Tutorial:
https://learn.adafruit.com/adafruit-rgb-matrix-plus-real-time-clock-hat-for-raspberry-pi/driving-matrices

# Troubleshooting

Several scripts rely on a older version of numpy. If there is a numpy error try downgrading pip to use numpy version 1.26.4 by running:
```bash
pip install numpy==1.26.4
```

If using a venv make sure that the venv is added to the .gitignore as any packages will take up too much space in the repo. 