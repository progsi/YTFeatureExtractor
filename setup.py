from setuptools import setup, find_packages
import os
import subprocess

def parse_env_file(env_file):
    with open(env_file, 'r') as file:
        lines = file.readlines()
    dependencies = []
    for line in lines:
        if line.startswith('- '):
            dependencies.append(line[2:].strip())
    return dependencies

env_file = 'env.yml'
if os.path.exists(env_file):
    install_requires = parse_env_file(env_file)
    # Install the conda environment
    subprocess.run(['conda', 'env', 'create', '-f', env_file], check=True)
else:
    install_requires = []

setup(
    name='YTFeatureExtractor',
    version='0.1',
    packages=find_packages(),
    install_requires=install_requires,
    author='Simon Hachmeier',
    author_email='simon.hachmeier@hu-berlin.de',
    description='A feature extractor for YouTube videos',
    url='https://github.com/progsi/YTFeatureExtractor',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)