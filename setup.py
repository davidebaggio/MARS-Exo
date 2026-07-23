import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'exo_head_slam'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*_launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='baggio',
    maintainer_email='baggio@todo.todo',
    description='Multi-agent RGB-D SLAM pipeline for wearable exoskeleton and head-mounted camera system.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_preprocessor = exo_head_slam.depth_preprocessor_node:main',
            'semantic_masker = exo_head_slam.semantic_masker_node:main',
            'extrinsic_solver = exo_head_slam.extrinsic_solver_node:main',
            'pointcloud_publisher = exo_head_slam.pointcloud_publisher_node:main',
            'dense_global_map = exo_head_slam.dense_global_map_node:main',
            'benchmark_evaluator = exo_head_slam.benchmark_evaluator_node:main',
        ],
    },
)
