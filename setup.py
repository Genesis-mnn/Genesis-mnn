# setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genesis-mnn",
    version="0.3.0",
    author="荀则瑞",
    license="GPL v3",
    description="Genesis: 第四代形态神经网络 (Morphic Neural Network) 构建框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "numpy",
        "h5py",
    ],
    extras_require={
        "matplotlib": ["matplotlib"],
        "tensorboard": ["tensorboard"],
        "qt": ["PyQt5", "pyqtgraph", "PyOpenGL"],
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)