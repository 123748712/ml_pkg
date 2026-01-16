import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare(package='turtlebot3_gazebo').find('turtlebot3_gazebo')
    pkg_tb3_ml = get_package_share_directory('turtlebot3_machine_learning')

    included_dqn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'turtlebot3_dqn_stage1.launch.py')
        )
    )


    dqn_gazebo_node = Node(
        package='turtlebot3_dqn',
        executable='dqn_gazebo',
        arguments=['1'],
        output='screen'
    )

    dqn_environment_node = Node(
        package='turtlebot3_dqn',
        executable='dqn_gazebo',
        arguments=['1'],
        output='screen'
    )

    return LaunchDescription([
       included_dqn_launch,
       dqn_gazebo_node,
       dqn_environment_node
    ])