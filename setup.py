from setuptools import setup, find_packages

# Read the requirements.txt file
def parse_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Define setup
setup(
    name="ytfeatextract",
    version="0.1.0",
    description="A feature extraction tool for YouTube videos.",
    author="Simon Hachmeier",  # Replace with your name
    author_email="simon.hachmeier@hu-berlin.de",  # Replace with your email
    url="https://github.com/progsi/ytfeatextract",  # Replace with your GitHub URL
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=parse_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update if using a different license
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "ytfeatextract=ytfeatextract.main:main",  # Replace with your package's entry point
        ]
    },
)
