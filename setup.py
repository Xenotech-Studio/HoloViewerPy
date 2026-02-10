"""Setup script for HoloViewer package."""
from setuptools import setup

setup(
    name="holoviewer",
    version="0.1.0",
    description="A base class for building interactive 3D viewers with OpenCV",
    packages=["holoviewer"],
    package_dir={"holoviewer": "."},
    install_requires=[
        "numpy",
        "torch",
        "opencv-python",
        "websockets>=12.0",
        "aiortc>=1.6",
        "av>=10.0",
    ],
    python_requires=">=3.8",
)
