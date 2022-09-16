from setuptools import setup, find_packages

setup(
    name='tod',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['einops>=0.3', 'torch>=1.8', 'torchvision>=0.9'
                      'matplotlib>=3.3'],
)
