"""Setup script for XenoViewer package."""
from setuptools import setup

setup(
    name="xenoviewer",
    version="0.1.0",
    description="A base class for building interactive 3D viewers with OpenCV",
    packages=["xenoviewer"],
    package_dir={"xenoviewer": "."},
    install_requires=[
        "numpy",
        "torch",
        "opencv-python",
    ],
    python_requires=">=3.8",
)
