from setuptools import setup

package_name = 'my_action_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ras',
    maintainer_email='kimhg@rastech.co.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_server_node = my_action_server.action_server:main',
            'action_server_0721 = my_action_server.action_server_0713:main',
            'action_test = my_action_server.test_action:main',
        ],
    },
)
