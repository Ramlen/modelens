from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="modelens",
    version="0.1.2",
    author="Leonid",
    author_email="lvgajval@yandex.ru",
    description="Diagnostic lens for binary classification ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ramlen/modelens",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "scikit-learn>=1.0",
    ],
)