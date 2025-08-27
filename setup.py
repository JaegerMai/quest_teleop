from setuptools import setup, find_packages

setup(
  name = 'quest_teleop',
  packages = find_packages(),
  install_requires = ['numpy', 'scipy', 'pybullet']
)