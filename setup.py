"""Setup configuration for pysuspension package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    install_requires = [
        'numpy>=1.20.0',
        'scipy>=1.7.0',
    ]

# Read dev requirements from requirements-dev.txt
dev_requirements_path = Path(__file__).parent / "requirements-dev.txt"
if dev_requirements_path.exists():
    with open(dev_requirements_path, 'r', encoding='utf-8') as f:
        dev_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#') and not line.startswith('-r')
        ]
else:
    dev_requires = [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
    ]

setup(
    name="pysuspension",
    version="0.1.0",
    author="pysuspension contributors",
    description="A Python library for suspension geometry modeling and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jadeblaquiere/pysuspension",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
    keywords="suspension geometry modeling vehicle automotive kinematics",
    project_urls={
        "Bug Reports": "https://github.com/jadeblaquiere/pysuspension/issues",
        "Source": "https://github.com/jadeblaquiere/pysuspension",
    },
)
