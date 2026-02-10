"""
HoloViewer - A base class for building interactive 3D viewers with OpenCV.

This package provides HoloViewer, an abstract base class that developers can inherit
to create their own interactive 3D viewers. It handles window management, camera controls,
input handling, and time playback, while allowing subclasses to implement custom rendering logic.
"""

from .holo_viewer import (
    HoloViewer,
    AxisSystem,
    AxisConfig,
    parse_network_args,
    is_client,
    wrap_time,
    to_uint8_bgr,
    _VK_SHIFT,
    _VK_CONTROL,
    _VK_MENU,
    _VK_SPACE,
    _VK_W,
    _VK_A,
    _VK_S,
    _VK_D,
    _VK_Q,
    _VK_E,
    _VK_TAB,
    _VK_OEM3,
    _win_key,
)

try:
    from .holo_viewer import cv2
except ImportError:
    cv2 = None

__all__ = [
    "HoloViewer",
    "AxisSystem",
    "AxisConfig",
    "parse_network_args",
    "is_client",
    "wrap_time",
    "to_uint8_bgr",
    "cv2",
    "_VK_SHIFT",
    "_VK_CONTROL",
    "_VK_MENU",
    "_VK_SPACE",
    "_VK_W",
    "_VK_A",
    "_VK_S",
    "_VK_D",
    "_VK_Q",
    "_VK_E",
    "_VK_TAB",
    "_VK_OEM3",
    "_win_key",
]

__version__ = "0.1.0"
