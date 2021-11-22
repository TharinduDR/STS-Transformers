from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ststransformers",
    version="0.0.5",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="An easy-to-use wrapper library for using Transformers in Semantic Textual Similarity Tasks.",
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
        "tqdm>=4.47.0",
        "regex",
        "transformers>=4.6.0",
        "datasets",
        "scipy",
        "scikit-learn",
        "seqeval",
        "tensorboard",
        "tensorboardX",
        "pandas",
        "tokenizers",
        "wandb>=0.10.32",
        "sentencepiece",
    ],
)
