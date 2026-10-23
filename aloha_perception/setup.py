from setuptools import find_packages, setup

package_name = 'aloha_tracking_perception'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ALOHA research lab',
    maintainer_email='maintainer@example.invalid',
    description='YOLO-once KCF tracker, aligned RealSense XYZ, and separate visualization',
    license='License to be chosen by project maintainers',
    entry_points={'console_scripts': [
        'detector = aloha_tracking_perception.detection:main',
        'tracking_viewer = aloha_tracking_perception.visualization:main',
    ]},
)
