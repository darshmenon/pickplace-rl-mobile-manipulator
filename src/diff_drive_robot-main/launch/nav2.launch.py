import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    home = os.path.expanduser('~')
    pkg_share = get_package_share_directory('diff_drive_robot')

    map_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_map = DeclareLaunchArgument(
        name='map',
        default_value=os.path.join(home, 'rosnav', 'my_map.yaml'),
        description='Full path to map yaml file')

    declare_sim_time = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation clock if true')

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map': map_file,
            'params_file': os.path.join(pkg_share, 'config', 'nav2_params.yaml'),
        }.items(),
    )

    return LaunchDescription([
        declare_map,
        declare_sim_time,
        nav2_bringup,
    ])
