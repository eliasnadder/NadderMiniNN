from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NadderMiniNN",
    version="1.0.0",
    author="Elias Nadder",
    author_email="eliasnadder@example.com",
    description="A mini neural network library built from scratch for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliasnadder/NadderMiniNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "scikit-learn>=0.24.0",
            "matplotlib>=3.3.0",
        ],
    },
    keywords="neural network, deep learning, machine learning, education",
    project_urls={
        "Bug Reports": "https://github.com/eliasnadder/NadderMiniNN/issues",
        "Source": "https://github.com/eliasnadder/NadderMiniNN",
    },
)
