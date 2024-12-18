from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='scan_code_vrx',
            executable='scan_code_visualiser',
            name='scan_code_visualiser',
            output='screen',
            emulate_tty=True,
        )
    ])
