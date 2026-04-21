from setuptools import find_packages, setup

package_name = 'pickplace_rl_mobile'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/display_launch.py',
            'launch/gazebo.launch.py',
            'launch/rl_train.launch.py',
            'launch/standalone_rl_training.launch.py',
            'launch/multi_world_training.launch.py',
            'launch/full_system.launch.py',
            'launch/display_mobile_ur3.launch.py',
            'launch/vla_phase1.launch.py',
            'launch/vla_full_pipeline.launch.py',
        ]),
        ('share/' + package_name + '/urdf', [
            'urdf/mobile_ur3.urdf',
            'urdf/pickplace_mobile_arm.urdf',
        ]),
        ('share/' + package_name + '/worlds', ['worlds/pickplace_world.world']),
        ('share/' + package_name + '/config', [
            'config/nav2_params.yaml',
            'config/robot_view.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Darsh Menon',
    maintainer_email='darshmenon02@gmail.com',
    description='Pick-and-place RL mobile manipulator with perception, Nav2, and safety',   
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = pickplace_rl_mobile.perception_node:main',
            'manip_rl_node = pickplace_rl_mobile.manip_rl_node:main',
            'safety_guard = pickplace_rl_mobile.safety_guard:main',
            'train_rl = pickplace_rl_mobile.train_rl:main',
            'test_policy = pickplace_rl_mobile.test_policy:main',
            'smart_pick_place = pickplace_rl_mobile.smart_pick_place:main',
            'vla_action_node = pickplace_rl_mobile.vla_action_node:main',
            'vla_vision_node = pickplace_rl_mobile.vla_vision_node:main',
            'vla_language_node = pickplace_rl_mobile.vla_language_node:main',
            'vla_coordinator_node = pickplace_rl_mobile.vla_coordinator_node:main',
            'task_planner_node   = pickplace_rl_mobile.task_planner_node:main',
            'object_memory_node   = pickplace_rl_mobile.object_memory_node:main',
            'navigate_and_pick    = pickplace_rl_mobile.navigate_and_pick_node:main',
        ],
    },
)
