from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ststransformers",
    version="0.0.3",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="An easy-to-use wrapper library for the Transformers library for Semantic Textual Similrity Tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TharinduDR/STS-Transformers",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "requests",
        "tqdm",
        "regex",
        "transformers",
        "scipy",
        "scikit-learn",
        "tensorboardx",
        "pandas",
        "tokenizers",
        "wandb"
    ],
)
