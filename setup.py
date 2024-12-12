from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='YTFeatureExtractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Simon Hachmeier',
    author_email='simon.hachmeier@hu-berlin.de',
    description='A package for extracting features from YouTube videos',
    url='https://github.com/progsi/YTFeatureExtractor',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)