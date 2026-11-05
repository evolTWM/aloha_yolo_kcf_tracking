from glob import glob
from setuptools import find_packages, setup

package_name = 'aloha_tracking_perception'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob(package_name + '/*.yaml')),
    ],
    package_data={package_name: ['*.yaml']},
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    extras_require={'yolo': ['torch', 'ultralytics']},
    zip_safe=False,
    maintainer='ALOHA research lab',
    maintainer_email='maintainer@example.invalid',
    description='YOLO aligned-depth detection, calibrated workspace XYZ, EMA smoothing, and visualization',
    license='License to be chosen by project maintainers',
    entry_points={'console_scripts': [
        'detector = aloha_tracking_perception.detection:main',
        'yolo_to_robot = aloha_tracking_perception.yolo_to_robot:main',
        'tracking_viewer = aloha_tracking_perception.visualization:main',
    ]},
)
