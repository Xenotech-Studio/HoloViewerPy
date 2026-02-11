import argparse
import asyncio
import concurrent.futures
import fractions
import logging
import math
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

logger = logging.getLogger("holoviewer")


def _ensure_console_logging() -> None:
    """若 holoviewer 尚未配置 handler，则添加控制台 INFO 输出，便于看到关键步骤与错误。"""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s [holoviewer] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

import numpy as np
import torch

FramePacket = Tuple[np.ndarray, float, Optional[float]]
CameraCommand = Tuple[np.ndarray, float, float, Optional[float]]

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# Socket 模式与 WebRTC 模式均需 pip install holoviewer[stream] (websockets, aiortc, av)
_WS_AVAILABLE = False
try:
    import websockets  # type: ignore
    _WS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore

_STREAM_AVAILABLE = False
try:
    import asyncio
    import av  # type: ignore
    from aiortc import RTCConfiguration, RTCIceCandidate, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaBlackhole
    from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
    _STREAM_AVAILABLE = True
except ImportError:
    asyncio = None  # type: ignore
    av = None
    RTCConfiguration = None  # type: ignore
    RTCIceCandidate = None  # type: ignore
    RTCIceServer = None  # type: ignore
    RTCPeerConnection = None  # type: ignore
    RTCSessionDescription = None  # type: ignore
    MediaBlackhole = None
    candidate_from_sdp = None  # type: ignore
    candidate_to_sdp = None  # type: ignore


def _default_rtc_configuration() -> Any:
    """WebRTC ICE 配置：使用公共 STUN，便于跨 NAT（如经 SSH 隧道信令、媒体直连）时建立连接。"""
    if not _STREAM_AVAILABLE or RTCConfiguration is None or RTCIceServer is None:
        return None
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
            RTCIceServer(urls="stun:stun1.l.google.com:19302"),
        ]
    )

_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12
_VK_SPACE = 0x20
_VK_W = 0x57
_VK_A = 0x41
_VK_S = 0x53
_VK_D = 0x44
_VK_Q = 0x51
_VK_E = 0x45
_VK_TAB = 0x09
_VK_OEM3 = 0xC0
_VK_OEM_PLUS = 0xBB   # + =
_VK_OEM_MINUS = 0xBD  # - _
_VK_OEM_4 = 0xDB     # [
_VK_OEM_6 = 0xDD     # ]
_VK_LEFT = 0x25
_VK_UP = 0x26
_VK_RIGHT = 0x27
_VK_DOWN = 0x28
_VK_RETURN = 0x0D
_win_key = None
if os.name == "nt":  # pragma: no cover - platform specific
    try:
        import ctypes  # type: ignore

        _win_key = ctypes.windll.user32  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _win_key = None

# macOS: 使用 Quartz 查询按键状态，实现与 Windows GetAsyncKeyState 类似的“按住连续移动”
_mac_key: Optional[Callable[[int], bool]] = None
if sys.platform == "darwin":  # pragma: no cover - platform specific
    try:
        import ctypes as _ctypes
        from ctypes import util as _ctypes_util

        _cg = _ctypes.CDLL(_ctypes_util.find_library("CoreGraphics"))
        # CGEventSourceKeyState(CGEventSourceStateID state, CGKeyCode key) -> Boolean
        _cg.CGEventSourceKeyState.argtypes = [_ctypes.c_uint32, _ctypes.c_uint16]
        _cg.CGEventSourceKeyState.restype = _ctypes.c_uint8
        # kCGEventSourceStateCombinedSessionState = 0
        _kCGEventSourceStateCombinedSessionState = 0
        # Carbon Events.h 物理键码 (ANSI)
        _kVK_ANSI_W = 0x0D
        _kVK_ANSI_A = 0x00
        _kVK_ANSI_S = 0x01
        _kVK_ANSI_D = 0x02
        _kVK_ANSI_Q = 0x0C
        _kVK_ANSI_E = 0x0E
        _kVK_Space = 0x31
        _kVK_Tab = 0x30
        _kVK_ANSI_Grave = 0x32
        _kVK_ANSI_LeftBracket = 0x21
        _kVK_ANSI_RightBracket = 0x1E
        _kVK_ANSI_Minus = 0x1B
        _kVK_ANSI_Equal = 0x18
        _kVK_Shift = 0x38
        _kVK_Option = 0x3A
        _kVK_LeftArrow = 0x7B
        _kVK_RightArrow = 0x7C
        _kVK_UpArrow = 0x7E
        _kVK_DownArrow = 0x7D
        _kVK_Return = 0x24

        def _mac_key_state(vk: int) -> bool:
            return bool(_cg.CGEventSourceKeyState(_kCGEventSourceStateCombinedSessionState, vk))

        _mac_key_map = {
            "w": _kVK_ANSI_W,
            "a": _kVK_ANSI_A,
            "s": _kVK_ANSI_S,
            "d": _kVK_ANSI_D,
            "q": _kVK_ANSI_Q,
            "e": _kVK_ANSI_E,
            "space": _kVK_Space,
            "tab": _kVK_Tab,
            "grave": _kVK_ANSI_Grave,
            "bracket_left": _kVK_ANSI_LeftBracket,
            "bracket_right": _kVK_ANSI_RightBracket,
            "minus": _kVK_ANSI_Minus,
            "plus": _kVK_ANSI_Equal,
            "shift": _kVK_Shift,
            "option": _kVK_Option,
            "left": _kVK_LeftArrow,
            "right": _kVK_RightArrow,
            "up": _kVK_UpArrow,
            "down": _kVK_DownArrow,
            "return": _kVK_Return,
        }

        def _mac_key(vk_name: str) -> bool:
            if vk_name not in _mac_key_map:
                return False
            return _mac_key_state(_mac_key_map[vk_name])

        _mac_key = _mac_key
    except Exception:  # pragma: no cover
        _mac_key = None

# Key name -> (vk, mac_name, cv2_keys). 内置映射，subclass 可用元组自定义。
# vk: Windows GetAsyncKeyState; mac_name: _mac_key 参数; cv2_keys: cv2.waitKey 返回值
_KEY_MAP: Dict[str, Tuple[Optional[int], Optional[str], List[int]]] = {
    "escape": (0x1B, None, [27]),
    "tab": (_VK_TAB, "tab", [9]),
    "space": (_VK_SPACE, "space", [32]),
    "grave": (_VK_OEM3, "grave", [96, 126]),
    "w": (_VK_W, "w", [ord("w"), ord("W")]),
    "a": (_VK_A, "a", [ord("a"), ord("A")]),
    "s": (_VK_S, "s", [ord("s"), ord("S")]),
    "d": (_VK_D, "d", [ord("d"), ord("D")]),
    "q": (_VK_Q, "q", [ord("q"), ord("Q")]),
    "e": (_VK_E, "e", [ord("e"), ord("E")]),
    "plus": (_VK_OEM_PLUS, "plus", [ord("+"), ord("=")]),
    "minus": (_VK_OEM_MINUS, "minus", [ord("-"), ord("_")]),
    "bracket_left": (_VK_OEM_4, "bracket_left", [ord("[")]),
    "bracket_right": (_VK_OEM_6, "bracket_right", [ord("]")]),
    "left": (_VK_LEFT, "left", []),
    "right": (_VK_RIGHT, "right", []),
    "up": (_VK_UP, "up", []),
    "down": (_VK_DOWN, "down", []),
}

# 类型：key_spec 可为 int(VK)、str(名称)、或 (vk, mac_name, cv2_keys) 元组
KeySpec = Union[int, str, Tuple[Optional[int], Optional[str], List[int]]]


def _key_spec_hashable(key_spec: KeySpec) -> Union[int, str, Tuple[Optional[int], Optional[str], Tuple[int, ...]]]:
    """转为可 hash 的 key，用于 _key_handler_prev 等 dict。"""
    if isinstance(key_spec, tuple) and len(key_spec) >= 3:
        return (key_spec[0], key_spec[1], tuple(key_spec[2]))
    return key_spec  # type: ignore


def _resolve_key_spec(key_spec: KeySpec) -> Tuple[Optional[int], Optional[str], List[int]]:
    """将 key_spec 解析为 (vk, mac_name, cv2_keys)。支持 int、str、或 (vk, mac_name, cv2_keys) 元组。"""
    if isinstance(key_spec, tuple) and len(key_spec) >= 3:
        return (key_spec[0], key_spec[1], key_spec[2])
    if isinstance(key_spec, int):
        return (key_spec, None, [])
    name = key_spec.lower() if isinstance(key_spec, str) else ""
    if name in _KEY_MAP:
        return _KEY_MAP[name]
    if len(name) == 1:
        c = name[0]
        vk = (0x41 + (ord(c.upper()) - ord("A"))) if c.isalpha() else None
        mac_name = c if c.isalpha() else None
        cv2_keys = [ord(c), ord(c.upper())] if c.isalpha() else [ord(c)]
        return (vk, mac_name, cv2_keys)
    return (None, None, [])


def _is_key_pressed(key_spec: KeySpec, key_from_cv2: Optional[int]) -> bool:
    """检查按键是否按下。key_spec: VK(int) 或 名称(str)。key_from_cv2: cv2.waitKey 返回值。"""
    vk, mac_name, cv2_keys = _resolve_key_spec(key_spec)
    if _win_key is not None and vk is not None:
        return bool(_win_key.GetAsyncKeyState(vk) & 0x8000)
    if _mac_key is not None and mac_name is not None:
        return _mac_key(mac_name)
    if key_from_cv2 is not None and cv2_keys and key_from_cv2 in cv2_keys:
        return True
    return False


class AxisConfig(NamedTuple):
    world_up: np.ndarray
    forward: np.ndarray
    right: np.ndarray


class AxisSystem:
    def __init__(
        self,
        mode: str,
        canonical_initial: np.ndarray,
        initial_overrides: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.mode = mode
        self._canonical_initial = canonical_initial.astype(np.float32)
        self._overrides: Dict[str, np.ndarray] = (
            initial_overrides.copy() if initial_overrides is not None else {}
        )
        self._config = self._compute_axis_alignment(mode)
        basis_matrix = np.stack(
            [self._config.forward, self._config.right, self._config.world_up], axis=1
        ).astype(np.float32)
        self._basis_matrix = basis_matrix
        self._world_to_canonical = np.linalg.inv(basis_matrix).astype(np.float32)

    def _compute_axis_alignment(self, mode: str) -> AxisConfig:
        axis_up_vectors = {
            "Z_UP": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "Y_UP": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "mY_UP": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        }
        world_up = axis_up_vectors.get(mode, axis_up_vectors["Z_UP"]).astype(np.float32)
        norm = np.linalg.norm(world_up)
        if norm <= 0.0:
            world_up = axis_up_vectors["Z_UP"]
            norm = np.linalg.norm(world_up)
        world_up = world_up / (norm + 1e-9)

        candidate_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(candidate_forward, world_up))) > 0.9:
            candidate_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        basis_right = np.cross(world_up, candidate_forward)
        if np.linalg.norm(basis_right) <= 1e-6:
            candidate_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            basis_right = np.cross(world_up, candidate_forward)
        basis_right = basis_right / (np.linalg.norm(basis_right) + 1e-9)

        basis_forward = np.cross(basis_right, world_up)
        basis_forward = basis_forward / (np.linalg.norm(basis_forward) + 1e-9)

        return AxisConfig(world_up=world_up, forward=basis_forward, right=basis_right)

    @property
    def world_up(self) -> np.ndarray:
        return self._config.world_up

    def canonical_to_world(self, vec: np.ndarray) -> np.ndarray:
        return (self._basis_matrix @ np.asarray(vec, dtype=np.float32)).astype(
            np.float32
        )

    def world_to_canonical(self, vec: np.ndarray) -> np.ndarray:
        return (self._world_to_canonical @ np.asarray(vec, dtype=np.float32)).astype(
            np.float32
        )

    def compute_camera_axes(
        self, yaw: float, pitch: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        forward_c = np.array([cy * cp, sy * cp, sp], dtype=np.float32)
        forward = self.canonical_to_world(forward_c)
        forward = forward / (np.linalg.norm(forward) + 1e-9)

        right = np.cross(self.world_up, forward)
        right = right / (np.linalg.norm(right) + 1e-9)

        up = np.cross(forward, right)
        up = up / (np.linalg.norm(up) + 1e-9)

        return forward, right, up

    def look_dir_to_angles(self, look_dir: np.ndarray) -> Tuple[float, float]:
        """将视线方向 look_dir 转为 (yaw, pitch)。"""
        dir_c = self.world_to_canonical(look_dir)
        dir_c = dir_c / (np.linalg.norm(dir_c) + 1e-9)
        yaw = math.atan2(float(dir_c[1]), float(dir_c[0]))
        pitch = math.asin(float(dir_c[2]))
        yaw += math.pi
        pitch = -pitch
        return yaw, pitch

    def initial_position(self) -> np.ndarray:
        if self.mode in self._overrides:
            return self._overrides[self.mode].astype(np.float32)
        return self.canonical_to_world(self._canonical_initial)


def wrap_time(t: float, tmin: float, tmax: float) -> float:
    width = float(tmax - tmin)
    if width <= 0.0:
        return tmin
    return tmin + ((t - tmin) % width)


def to_uint8_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    img = torch.clamp(image_tensor, 0.0, 1.0).detach().cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img[:, :, ::-1]


def parse_network_args(argv: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[str], bool, bool]:
    """Parse --expose-port、--scribe、--subscribe、--headless、--webrtc from argv. Returns (expose_port, scribe_addr, headless, use_webrtc)."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--expose-port", type=int, default=None, metavar="PORT", help="Start WebSocket server on PORT for streaming (default: socket mode; use --webrtc for WebRTC)")
    parser.add_argument("--scribe", type=str, default=None, metavar="HOST:PORT", help="Connect to remote HOST:PORT and display stream (no local render)")
    parser.add_argument("--subscribe", type=str, default=None, metavar="HOST:PORT or PORT", help="Same as --scribe; if only PORT, use 127.0.0.1:PORT")
    parser.add_argument("--headless", action="store_true", help="No cv2 window; use with --expose-port to run server and render pipeline only.")
    parser.add_argument("--webrtc", action="store_true", help="Use WebRTC for streaming (LAN only without TURN); default is WebSocket/socket mode for cross-network (e.g. SSH tunnel)")
    args, _ = parser.parse_known_args(argv)
    expose_port = args.expose_port
    scribe_addr = args.scribe or args.subscribe
    if scribe_addr and ":" not in scribe_addr:
        scribe_addr = "127.0.0.1:" + scribe_addr
    return expose_port, scribe_addr, getattr(args, "headless", False), getattr(args, "webrtc", False)


def is_client(argv: Optional[List[str]] = None) -> bool:
    """True 表示当前为订阅端（--scribe/--subscribe），仅收流不渲染，可据此延迟导入仅服务端需要的包。"""
    _, scribe_addr, _, _ = parse_network_args(argv)
    return scribe_addr is not None


def _draw_error_image(width: int, height: int, message: str, addr: str = "") -> np.ndarray:
    """在灰色背景上绘制错误/状态信息，用于订阅端连接失败或断开时在窗口内显示。"""
    if cv2 is None:
        return np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
    img = np.full((height, width, 3), 48, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (220, 220, 220)
    y = 40
    if addr:
        cv2.putText(img, f"Target: {addr}", (20, y), font, font_scale, color, thickness)
        y += 36
    # 简单换行：按字符数折行
    chars_per_line = max(20, width // 14)
    for i in range(0, len(message), chars_per_line):
        line = message[i : i + chars_per_line]
        cv2.putText(img, line, (20, y), font, font_scale, color, thickness)
        y += 36
    cv2.putText(img, "Press ESC or close window to exit.", (20, height - 24), font, 0.5, (160, 160, 160), 1)
    return img


# --- WebRTC stream helpers (only used when _STREAM_AVAILABLE) ---
def _run_expose_server(
    port: int,
    frame_queue: "Queue[FramePacket]",
    stop_ev: threading.Event,
    camera_command_queue: "Queue[CameraCommand]",
    viewer: "HoloViewer",
) -> None:
    if not _STREAM_AVAILABLE:
        return
    from aiortc.mediastreams import VideoStreamTrack  # type: ignore

    class QueueVideoTrack(VideoStreamTrack):
        kind = "video"

        def __init__(self, q: "Queue[FramePacket]", stop: threading.Event) -> None:
            super().__init__()
            self._queue = q
            self._stop = stop
            self._pts = 0

        async def recv(self) -> Any:
            while not self._stop.is_set():
                try:
                    frame_bgr, _, _ = self._queue.get(timeout=0.5)
                except Empty:
                    continue
                self._pts += 1
                av_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                av_frame.pts = self._pts
                av_frame.time_base = __import__("fractions").Fraction(1, 30)
                return av_frame
            raise Exception("track stopped")

    async def _serve(
        ws_port: int,
        q: "Queue[FramePacket]",
        stop: threading.Event,
        cam_queue: "Queue[CameraCommand]",
        v: "HoloViewer",
    ) -> None:
        frame_track: Optional[QueueVideoTrack] = None
        pc: Optional[Any] = None
        remote_pos: Optional[np.ndarray] = None
        remote_yaw: Optional[float] = None
        remote_pitch: Optional[float] = None

        def init_remote() -> None:
            nonlocal remote_pos, remote_yaw, remote_pitch
            if remote_pos is not None:
                return
            pos, forward = v.get_initial_pose()
            pos = pos.astype(np.float64)
            forward = forward / (np.linalg.norm(forward) + 1e-9)
            remote_yaw, remote_pitch = v.axis_system.look_dir_to_angles(forward)
            remote_pos = pos

        async def handler(websocket: Any) -> None:
            nonlocal frame_track, pc, remote_pos, remote_yaw, remote_pitch
            peer = getattr(websocket, "remote_address", None) or "unknown"
            logger.info("Client connected (WebSocket), peer=%s", peer)
            try:
                logger.info("Waiting for SDP offer from client ...")
                msg = await websocket.recv()
                data = __import__("json").loads(msg)
                msg_type = data.get("type")
                logger.info("Received message from client: type=%s", msg_type)
                if msg_type == "offer":
                    logger.info("Creating RTCPeerConnection and video track ...")
                    pc = RTCPeerConnection(_default_rtc_configuration())
                    frame_track = QueueVideoTrack(q, stop)
                    pc.addTrack(frame_track)
                    @pc.on("datachannel")
                    def on_datachannel(channel: Any) -> None:
                        if getattr(channel, "label", "") == "input":
                            @channel.on("message")
                            def on_message(message: Any) -> None:
                                nonlocal remote_pos, remote_yaw, remote_pitch
                                try:
                                    recv_data = __import__("json").loads(message) if isinstance(message, str) else __import__("json").loads(message.decode("utf-8"))
                                    init_remote()
                                    pos, yaw, pitch = v._apply_input(remote_pos, remote_yaw, remote_pitch, recv_data)
                                    remote_pos, remote_yaw, remote_pitch = pos, yaw, pitch
                                    try:
                                        cam_queue.put_nowait((pos, yaw, pitch, None))
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                    logger.info("Setting remote description (offer) ...")
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type=data["type"]))
                    logger.info("Creating WebRTC answer ...")
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    answer_msg = __import__("json").dumps({"type": pc.localDescription.type, "sdp": pc.localDescription.sdp})
                    await websocket.send(answer_msg)
                    logger.info("WebRTC answer sent (%d bytes), streaming video to client; input via Data Channel", len(answer_msg))
                    @pc.on("icecandidate")
                    def on_ice(candidate: Any) -> None:
                        if candidate:
                            cand_str = "candidate:" + candidate_to_sdp(candidate)
                            asyncio.ensure_future(websocket.send(__import__("json").dumps({
                                "type": "candidate",
                                "candidate": cand_str,
                                "sdpMid": getattr(candidate, "sdpMid", None),
                                "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None),
                            })))
                    while not stop.is_set():
                        try:
                            recv_msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            recv_data = __import__("json").loads(recv_msg)
                            if recv_data.get("type") == "candidate" and recv_data.get("candidate"):
                                raw = recv_data["candidate"]
                                sdp_part = raw.split(":", 1)[1] if isinstance(raw, str) and ":" in raw else raw
                                c = candidate_from_sdp(sdp_part)
                                c.sdpMid = recv_data.get("sdpMid")
                                c.sdpMLineIndex = recv_data.get("sdpMLineIndex")
                                await pc.addIceCandidate(c)
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break
            except Exception as e:
                logger.warning("Client handler error: %s", e)
                logger.exception("Full traceback:")
            finally:
                if pc:
                    await pc.close()
                logger.info("Client disconnected (peer=%s)", peer)

        logger.info("WebSocket server binding to 0.0.0.0:%s ...", ws_port)
        async with websockets.serve(handler, "0.0.0.0", ws_port, ping_interval=None, ping_timeout=None, close_timeout=1):  # type: ignore
            logger.info("Server ready. Subscribe side use: --subscribe 127.0.0.1:%s (this machine) or --subscribe <this-ip>:%s (LAN)", ws_port, ws_port)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, stop.wait)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_serve(port, frame_queue, stop_ev, camera_command_queue, viewer))
    except Exception:
        logger.exception("Expose server failed")
    finally:
        loop.close()
        logger.info("WebSocket server stopped")


def _run_expose_server_socket(
    port: int,
    frame_queue: "Queue[FramePacket]",
    stop_ev: threading.Event,
    camera_command_queue: "Queue[CameraCommand]",
    viewer: "HoloViewer",
) -> None:
    """WebSocket 纯 socket 模式：服务端经同一条连接发 H.264 帧、收 JSON 操控，无需 WebRTC/ICE。
    低延迟：只发最新帧、周期性 I 帧、rc_lookahead=0。自适应画质：根据客户端 stats 反馈调节码率与分辨率。"""
    if not _WS_AVAILABLE or websockets is None:
        return
    if not _STREAM_AVAILABLE or av is None:
        logger.error("Socket mode (H.264) requires: pip install holoviewer[stream] (websockets, aiortc, av)")
        return
    json_mod = __import__("json")

    async def _serve_socket(
        ws_port: int,
        q: "Queue[FramePacket]",
        stop: threading.Event,
        cam_queue: "Queue[CameraCommand]",
        v: "HoloViewer",
    ) -> None:
        remote_pos: Optional[np.ndarray] = None
        remote_yaw: Optional[float] = None
        remote_pitch: Optional[float] = None

        def init_remote() -> None:
            nonlocal remote_pos, remote_yaw, remote_pitch
            if remote_pos is not None:
                return
            pos, forward = v.get_initial_pose()
            pos = pos.astype(np.float64)
            forward = forward / (np.linalg.norm(forward) + 1e-9)
            remote_yaw, remote_pitch = v.axis_system.look_dir_to_angles(forward)
            remote_pos = pos

        async def handler(websocket: Any) -> None:
            nonlocal remote_pos, remote_yaw, remote_pitch
            peer = getattr(websocket, "remote_address", None) or "unknown"
            logger.info("Client connected (WebSocket socket mode), peer=%s", peer)
            encode_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                await websocket.send(json_mod.dumps({"type": "tunnel", "version": 1, "codec": "h264"}))
                loop = asyncio.get_running_loop()

                # 自适应画质（类似 WebRTC REMB/反馈）：根据客户端 stats 调节码率与分辨率
                _BITRATE_STEPS = [300_000, 500_000, 700_000, 1_000_000, 1_500_000]
                _SCALE_STEPS = [0.5, 0.75, 1.0]
                adaptive = {"bitrate_idx": 4, "scale_idx": 2}  # 初始最高质量

                # 低延迟：只编码并发送最新一帧，丢弃积压的旧帧（last-frame-wins）
                def _get_latest_frame() -> Optional[FramePacket]:
                    latest: Optional[FramePacket] = None
                    try:
                        latest = q.get(timeout=0.5)
                    except Empty:
                        return None
                    while True:
                        try:
                            latest = q.get_nowait()
                        except Empty:
                            break
                    return latest

                # 周期性关键帧（每 KEYFRAME_INTERVAL 帧一个 I 帧，便于恢复与限时延）
                _KEYFRAME_INTERVAL = 30  # 约 1 秒 @ 30fps
                codec: Optional[Any] = None
                pts = 0
                time_base = fractions.Fraction(1, 30)

                # 将编码移到专用线程，避免阻塞 asyncio 事件循环收控制消息。
                def _encode_latest_frame(
                    item: FramePacket,
                    bitrate_idx: int,
                    scale_idx: int,
                ) -> bytes:
                    nonlocal codec, pts
                    frame_bgr, _, _ = item
                    h, w = frame_bgr.shape[:2]
                    scale = _SCALE_STEPS[scale_idx]
                    encode_w = max(64, int(w * scale))
                    encode_h = max(64, int(h * scale))
                    if scale < 1.0 and cv2 is not None:
                        frame_bgr = cv2.resize(frame_bgr, (encode_w, encode_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        encode_w, encode_h = w, h
                    bitrate = _BITRATE_STEPS[bitrate_idx]
                    if codec is None or codec.width != encode_w or codec.height != encode_h:
                        codec = av.CodecContext.create("libx264", "w")
                        codec.width = encode_w
                        codec.height = encode_h
                        codec.pix_fmt = "yuv420p"
                        codec.time_base = time_base
                        codec.framerate = fractions.Fraction(30, 1)
                        codec.bit_rate = bitrate
                        codec.options = {
                            "tune": "zerolatency",
                            "level": "31",
                            "rc_lookahead": "0",  # 禁用 lookahead，降低编码延迟
                        }
                        codec.profile = "baseline"
                    if codec.bit_rate != bitrate:
                        codec.bit_rate = bitrate
                    pts += 1
                    av_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                    av_frame.pts = pts
                    av_frame.time_base = time_base
                    if pts == 1 or (pts % _KEYFRAME_INTERVAL) == 1:
                        av_frame.pict_type = av.video.frame.PictureType.I
                    else:
                        av_frame.pict_type = av.video.frame.PictureType.NONE
                    data = b""
                    for packet in codec.encode(av_frame):
                        data += bytes(packet)
                    return data

                async def send_frames() -> None:
                    while not stop.is_set():
                        try:
                            item = await loop.run_in_executor(None, _get_latest_frame)
                            if item is None:
                                continue
                            data = await loop.run_in_executor(
                                encode_executor,
                                _encode_latest_frame,
                                item,
                                adaptive["bitrate_idx"],
                                adaptive["scale_idx"],
                            )
                            if item[2] is not None:
                                await websocket.send(json_mod.dumps({"type": "frame_meta", "input_ts": item[2]}))
                            if data:
                                await websocket.send(data)
                        except Exception:
                            break

                async def recv_control() -> None:
                    nonlocal remote_pos, remote_yaw, remote_pitch
                    while not stop.is_set():
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            if isinstance(msg, str):
                                recv_data = json_mod.loads(msg)
                                msg_type = recv_data.get("type")
                                if msg_type == "stats":
                                    # 客户端上报丢帧/收帧，用于自适应降画质（类似 WebRTC 的 REMB/反馈）
                                    dropped = recv_data.get("dropped", 0)
                                    received = max(recv_data.get("received", 0), 1)
                                    ratio = dropped / received
                                    if ratio > 0.15:
                                        if adaptive["bitrate_idx"] > 0:
                                            adaptive["bitrate_idx"] -= 1
                                            logger.debug("Adaptive: lower bitrate (dropped=%d received=%d)", dropped, received)
                                        elif adaptive["scale_idx"] > 0:
                                            adaptive["scale_idx"] -= 1
                                            logger.debug("Adaptive: lower resolution (dropped=%d received=%d)", dropped, received)
                                    elif ratio < 0.05 and received > 15:
                                        if adaptive["scale_idx"] < len(_SCALE_STEPS) - 1:
                                            adaptive["scale_idx"] += 1
                                            logger.debug("Adaptive: raise resolution (dropped=%d received=%d)", dropped, received)
                                        elif adaptive["bitrate_idx"] < len(_BITRATE_STEPS) - 1:
                                            adaptive["bitrate_idx"] += 1
                                            logger.debug("Adaptive: raise bitrate (dropped=%d received=%d)", dropped, received)
                                    continue
                                init_remote()
                                pos, yaw, pitch = v._apply_input(remote_pos, remote_yaw, remote_pitch, recv_data)
                                remote_pos, remote_yaw, remote_pitch = pos, yaw, pitch
                                input_ts_raw = recv_data.get("__input_ts")
                                input_ts = float(input_ts_raw) if isinstance(input_ts_raw, (int, float)) else None
                                try:
                                    while True:
                                        cam_queue.get_nowait()
                                except Empty:
                                    pass
                                try:
                                    cam_queue.put_nowait((pos, yaw, pitch, input_ts))
                                except Exception:
                                    pass
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break

                await asyncio.gather(send_frames(), recv_control())
            except Exception as e:
                logger.warning("Socket handler error: %s", e)
            finally:
                encode_executor.shutdown(wait=False)
                logger.info("Client disconnected (peer=%s)", peer)

        logger.info("WebSocket (socket) server binding to 0.0.0.0:%s ...", ws_port)
        async with websockets.serve(
            handler,
            "0.0.0.0",
            ws_port,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=1,
            compression=None,
            max_queue=1,
        ):  # type: ignore
            logger.info("Socket mode ready. Subscribe: --subscribe 127.0.0.1:%s or <this-ip>:%s", ws_port, ws_port)
            await asyncio.get_event_loop().run_in_executor(None, stop.wait)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_serve_socket(port, frame_queue, stop_ev, camera_command_queue, viewer))
    except Exception:
        logger.exception("Socket expose server failed")
    finally:
        loop.close()
        logger.info("WebSocket (socket) server stopped")


def _run_scribe_client_webrtc(
    addr: str,
    frame_queue: "Queue[Any]",  # 可放入 np.ndarray | ("error", str) | None
    stop_ev: threading.Event,
    camera_send_queue: Optional["Queue[Dict[str, Any]]"] = None,
) -> None:
    if not _STREAM_AVAILABLE:
        return
    host, port_s = addr.rsplit(":", 1)
    port = int(port_s)

    def put_error(msg: str) -> None:
        logger.error("Subscribe error: %s", msg)
        try:
            frame_queue.put_nowait(("error", msg))
        except Exception:
            pass

    uri = f"ws://{host}:{port}"
    connect_timeout = 10
    logger.info("Connecting to %s (connect_timeout=%ss) ...", uri, connect_timeout)

    async def _send_camera_loop(channel: Any, cam_queue: "Queue[Dict[str, Any]]", stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                inp = cam_queue.get_nowait()
                payload = __import__("json").dumps(inp)
                if hasattr(channel, "send") and callable(getattr(channel, "send")):
                    send_fn = channel.send
                    if asyncio.iscoroutinefunction(send_fn):
                        await send_fn(payload)
                    else:
                        send_fn(payload)
            except Empty:
                await asyncio.sleep(0.02)
            except Exception:
                break

    async def _client(
        ws_host: str,
        ws_port: int,
        q: "Queue[Any]",
        stop: threading.Event,
        cam_send_q: Optional["Queue[Dict[str, Any]]"] = None,
    ) -> None:
        pc = RTCPeerConnection(_default_rtc_configuration())
        pc.addTransceiver("video", direction="recvonly")
        input_channel: Optional[Any] = None
        if cam_send_q is not None:
            input_channel = pc.createDataChannel("input")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        uri_inner = f"ws://{ws_host}:{ws_port}"
        track_received: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        channel_open_ev: Optional[asyncio.Event] = asyncio.Event() if input_channel else None
        if input_channel is not None and channel_open_ev is not None:
            @input_channel.on("open")
            def _on_input_channel_open() -> None:
                channel_open_ev.set()

        @pc.on("track")
        def on_track(track: Any) -> None:
            if not track_received.done():
                track_received.set_result(track)

        try:
            async with websockets.connect(uri_inner, ping_interval=None, ping_timeout=None, close_timeout=2, open_timeout=connect_timeout) as ws:  # type: ignore
                logger.info("WebSocket connected to %s, sending SDP offer ...", uri_inner)
                send_task: Optional[asyncio.Task[None]] = None
                recv_ice_task: Optional[asyncio.Task[None]] = None
                await ws.send(__import__("json").dumps({"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}))
                @pc.on("icecandidate")
                def on_ice(candidate: Any) -> None:
                    if candidate:
                        cand_str = "candidate:" + candidate_to_sdp(candidate)
                        asyncio.get_event_loop().create_task(ws.send(__import__("json").dumps({
                            "type": "candidate",
                            "candidate": cand_str,
                            "sdpMid": getattr(candidate, "sdpMid", None),
                            "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None),
                        })))
                logger.info("Waiting for SDP answer from server (timeout 15s) ...")
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                except asyncio.TimeoutError:
                    put_error("Timeout waiting for SDP answer from server (15s). Check sender logs for errors.")
                    return
                answer_data = __import__("json").loads(msg)
                logger.info("SDP answer received, setting remote description ...")
                await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"]))

                async def _recv_ice_loop() -> None:
                    """持续接收服务端发来的 ICE 候选并加入 PC，否则跨网时 ICE 无法建立。"""
                    while not stop.is_set():
                        try:
                            recv_msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            recv_data = __import__("json").loads(recv_msg)
                            if recv_data.get("type") == "candidate":
                                cand = recv_data.get("candidate")
                                if cand:
                                    sdp_part = cand.split(":", 1)[1] if isinstance(cand, str) and ":" in cand else cand
                                    c = candidate_from_sdp(sdp_part)
                                    c.sdpMid = recv_data.get("sdpMid")
                                    c.sdpMLineIndex = recv_data.get("sdpMLineIndex")
                                    await pc.addIceCandidate(c)
                                else:
                                    await pc.addIceCandidate(None)
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break
                recv_ice_task = asyncio.create_task(_recv_ice_loop())

                logger.info("Waiting for video track (timeout 10s) ...")
                remote_track = await asyncio.wait_for(track_received, timeout=10.0)
                logger.info("Video track received, streaming from %s", uri_inner)

                ice_connected_timeout = 15.0
                t0 = time.monotonic()
                while (time.monotonic() - t0) < ice_connected_timeout and getattr(pc, "connectionState", "") not in ("connected",):
                    await asyncio.sleep(0.1)
                if getattr(pc, "connectionState", "") != "connected":
                    logger.warning("ICE/connection did not reach 'connected' within %.0fs; attempting stream anyway", ice_connected_timeout)

                if input_channel is not None and channel_open_ev is not None and cam_send_q is not None:
                    try:
                        await asyncio.wait_for(channel_open_ev.wait(), timeout=5.0)
                        logger.info("Input Data Channel open, sending camera control")
                        send_task = asyncio.create_task(_send_camera_loop(input_channel, cam_send_q, stop))
                    except asyncio.TimeoutError:
                        logger.warning("Input Data Channel did not open within 5s, camera control may not work")
                try:
                    while not stop.is_set():
                        try:
                            frame = await asyncio.wait_for(remote_track.recv(), timeout=1.0)
                            img = frame.to_ndarray(format="bgr24")
                            try:
                                q.put_nowait(img)
                            except Exception:
                                pass
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            put_error(str(e))
                            break
                finally:
                    if recv_ice_task is not None:
                        recv_ice_task.cancel()
                        try:
                            await recv_ice_task
                        except asyncio.CancelledError:
                            pass
                    if send_task is not None:
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
        except Exception as e:
            err = str(e)
            put_error(err)
            if "timed out" in err.lower() or "timeout" in err.lower():
                logger.info("Tip: ensure sender is running with --expose-port %s and address %s is correct (same machine: 127.0.0.1)", port, uri_inner)
            elif "refused" in err.lower() or "connect" in err.lower():
                logger.info("Tip: is the sender running? Start it with: python ... --expose-port %s", port)
        finally:
            await pc.close()
            try:
                q.put_nowait(None)
            except Exception:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_client(host, port, frame_queue, stop_ev, camera_send_queue))
    except Exception as e:
        logger.exception("Subscribe client failed")
        put_error(str(e))
        try:
            frame_queue.put_nowait(None)
        except Exception:
            pass
    finally:
        loop.close()


def _run_scribe_client_socket(
    addr: str,
    frame_queue: "Queue[Any]",
    stop_ev: threading.Event,
    camera_send_queue: Optional["Queue[Dict[str, Any]]"] = None,
) -> None:
    """WebSocket 纯 socket 模式：客户端经同一条连接收 H.264 帧、发 JSON 操控。低延迟：只保留最新帧。自适应：定期上报 stats 供服务端降画质。"""
    if not _WS_AVAILABLE or websockets is None:
        try:
            frame_queue.put_nowait(("error", "Socket mode requires: pip install websockets"))
        except Exception:
            pass
        return
    if not _STREAM_AVAILABLE or av is None:
        try:
            frame_queue.put_nowait(("error", "Socket mode (H.264) requires: pip install holoviewer[stream] (websockets, aiortc, av)"))
        except Exception:
            pass
        return
    host, port_s = addr.rsplit(":", 1)
    port = int(port_s)
    uri = f"ws://{host}:{port}"
    json_mod = __import__("json")
    connect_timeout = 10

    def put_error(msg: str) -> None:
        logger.error("Subscribe (socket) error: %s", msg)
        try:
            frame_queue.put_nowait(("error", msg))
        except Exception:
            pass

    async def _client() -> None:
        try:
            async with websockets.connect(
                uri,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=2,
                open_timeout=connect_timeout,
                compression=None,
                max_queue=1,
            ) as ws:  # type: ignore
                first = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(first, str):
                    data = json_mod.loads(first)
                    if data.get("type") != "tunnel":
                        put_error("Expected tunnel greeting from server")
                        return
                else:
                    put_error("Expected tunnel greeting (string), got binary")
                    return

                decoder = av.CodecContext.create("h264", "r")
                time_base = fractions.Fraction(1, 30)
                # 用于自适应画质：统计丢帧/收帧，定期上报服务端（类似 WebRTC REMB 反馈）
                stats = [0, 0]  # [dropped, received]

                pending_input_ts: Optional[float] = None

                def decode_and_put(msg: bytes, input_ts: Optional[float]) -> bool:
                    if not msg:
                        return False
                    produced = False
                    try:
                        packet = av.Packet(msg)
                        packet.pts = 0
                        packet.time_base = time_base
                        for frame in decoder.decode(packet):
                            produced = True
                            arr = frame.to_ndarray(format="bgr24")
                            try:
                                # 低延迟：只保留最新一帧，丢弃积压的旧帧
                                dropped = 0
                                while not frame_queue.empty():
                                    try:
                                        frame_queue.get_nowait()
                                        dropped += 1
                                    except Exception:
                                        break
                                frame_queue.put_nowait(("frame", arr, input_ts))
                                stats[0] += dropped
                                stats[1] += 1
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return produced

                async def recv_frames() -> None:
                    nonlocal pending_input_ts
                    while not stop_ev.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            if isinstance(msg, bytes):
                                if decode_and_put(msg, pending_input_ts):
                                    pending_input_ts = None
                            elif isinstance(msg, str):
                                try:
                                    data = json_mod.loads(msg)
                                    if data.get("type") == "frame_meta":
                                        raw_ts = data.get("input_ts")
                                        pending_input_ts = float(raw_ts) if isinstance(raw_ts, (int, float)) else None
                                except Exception:
                                    pass
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break

                async def send_stats() -> None:
                    """定期上报丢帧/收帧，供服务端做自适应码率与分辨率。"""
                    while not stop_ev.is_set():
                        await asyncio.sleep(1.0)
                        if stop_ev.is_set():
                            break
                        try:
                            await ws.send(json_mod.dumps({"type": "stats", "dropped": stats[0], "received": stats[1]}))
                            stats[0], stats[1] = 0, 0
                        except Exception:
                            break

                async def send_control() -> None:
                    while not stop_ev.is_set() and camera_send_queue is not None:
                        try:
                            inp = camera_send_queue.get_nowait()
                            await ws.send(json_mod.dumps(inp))
                        except Empty:
                            await asyncio.sleep(0.02)
                        except Exception:
                            break

                if camera_send_queue is not None:
                    await asyncio.gather(recv_frames(), send_control(), send_stats())
                else:
                    await asyncio.gather(recv_frames(), send_stats())
        except Exception as e:
            put_error(str(e))
        finally:
            try:
                frame_queue.put_nowait(None)
            except Exception:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_client())
    except Exception as e:
        logger.exception("Socket subscribe client failed")
        put_error(str(e))
        try:
            frame_queue.put_nowait(None)
        except Exception:
            pass
    finally:
        loop.close()


class HoloViewer(ABC):
    def __init__(
        self,
        window_name: str,
        width: int = 800,
        height: int = 600,
        fov_y_deg: float = 50.0,
        axis_mode: str = "Z_UP",
        canonical_initial: Optional[np.ndarray] = None,
        axis_initial_overrides: Optional[Dict[str, np.ndarray]] = None,
        pan_sensitivity: float = 0.01,
        move_speed: float = 0.05,
        arrow_rotate_speed: float = 0.02,
        show_axes: bool = True,
        expose_port: Optional[int] = None,
        scribe_addr: Optional[str] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) is required. Install via pip install opencv-python."
            )
        canonical = (
            canonical_initial.astype(np.float32)
            if canonical_initial is not None
            else np.array([2.0, 2.0, 2.0], dtype=np.float32)
        )
        self.axis_system = AxisSystem(
            axis_mode, canonical, initial_overrides=axis_initial_overrides
        )
        self.window_name = window_name
        self.width = width
        self.height = height
        self.fov_y_deg = fov_y_deg
        self.pan_sensitivity = pan_sensitivity
        self.move_speed = move_speed
        self.arrow_rotate_speed = arrow_rotate_speed
        self.show_axes = show_axes
        self.play_rate = 1.0
        self.time_min = 0.0
        self.time_max = 1.0
        self.time_step = 0.02
        self.sim_time = 0.0
        self.playing = True
        self._dragging = False
        self._pan_dragging = False
        self._last_mouse = (0, 0)
        self._prev_tab = False
        self._prev_oem3 = False
        self._prev_q = False
        self._prev_e = False
        self._show_axes_flag = show_axes
        self._locked = False
        # 左键拖拽 = 旋转（与右键一致）；Ctrl+左键点击 = 锁定
        self._left_dragging = False
        self._ctrl_click_pos: Optional[Tuple[int, int]] = None
        self._expose_port = expose_port
        self._scribe_addr = scribe_addr
        # Subclass extensibility: keyboard callbacks, FPS overlay, window title suffix
        self._key_handlers: List[Tuple[KeySpec, Callable[["HoloViewer"], None]]] = []
        self._key_handler_prev: Dict[Any, bool] = {}
        self.show_fps = False
        self.show_window_resize_log = False

    def register_key_handler(
        self,
        key: KeySpec,
        callback: Callable[["HoloViewer"], None],
    ) -> None:
        """Register a keyboard callback. Called on key press (edge-triggered).
        key: VK (int), name (str), or custom tuple (vk, mac_name, cv2_keys).
        Example: register_key_handler((_VK_W, "w", [ord("w"), ord("W")]), cb)
        """
        self._key_handlers.append((key, callback))

    def on_frame_camera_update(
        self, pos: np.ndarray, yaw: float, pitch: float, dt: float
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        """Override to modify camera before rendering. Return (pos, yaw, pitch) to override, or None to use default."""
        return None

    def on_frame_draw(self, img_bgr: np.ndarray) -> None:
        """Override to draw overlay on image after axes."""
        pass

    def get_window_title_suffix(self) -> str:
        """Override to return dynamic suffix for window title."""
        return ""

    def should_draw_axes(self) -> bool:
        """Override to control axes overlay. Default: self._show_axes_flag."""
        return self._show_axes_flag

    def _get_network_mode(self) -> Tuple[Optional[int], Optional[str], bool, bool]:
        """Return (expose_port, scribe_addr, headless, use_webrtc). If not set in __init__, parse from sys.argv."""
        expose = self._expose_port
        scribe = self._scribe_addr
        if expose is None and scribe is None:
            expose, scribe, headless, use_webrtc = parse_network_args()
        else:
            _, _, headless, use_webrtc = parse_network_args()
        return expose, scribe, headless, use_webrtc

    def run(self) -> None:
        _ensure_console_logging()
        expose_port, scribe_addr, headless, use_webrtc = self._get_network_mode()
        if scribe_addr:
            logger.info("Subscribe mode: connecting to %s (no local render)%s", scribe_addr, " [WebRTC]" if use_webrtc else " [WebSocket]")
            if use_webrtc:
                self._run_scribe_webrtc(scribe_addr)
            else:
                self._run_scribe_socket(scribe_addr)
            return
        if expose_port is not None:
            logger.info("Expose mode: WebSocket server on port %s%s%s", expose_port, " (headless)" if headless else "", " [WebRTC]" if use_webrtc else " [WebSocket]")
            if use_webrtc:
                self._run_expose_webrtc(expose_port, headless=headless)
            else:
                self._run_expose_socket(expose_port, headless=headless)
            return
        logger.info("Local mode: running render pipeline")
        self._run_local()

    def _run_scribe_webrtc(self, addr: str) -> None:
        """连接远端 WebSocket，通过 WebRTC 接收画面并显示（需 --webrtc；适合局域网）。"""
        if not _STREAM_AVAILABLE:
            raise RuntimeError("Scribe mode (WebRTC) requires: pip install holoviewer[stream] (websockets, aiortc, av)")
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required.")
        frame_queue: "Queue[Any]" = Queue(maxsize=2)
        camera_send_queue: "Queue[Dict[str, Any]]" = Queue(maxsize=8)
        stop_ev = threading.Event()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        thread = threading.Thread(
            target=_run_scribe_client_webrtc,
            args=(addr, frame_queue, stop_ev, camera_send_queue),
            daemon=True,
        )
        thread.start()
        self._scribe_display_loop(frame_queue, stop_ev, addr, thread, camera_send_queue)

    def _scribe_display_loop(
        self,
        frame_queue: "Queue[Any]",
        stop_ev: threading.Event,
        addr: str,
        thread: threading.Thread,
        camera_send_queue: "Queue[Dict[str, Any]]",
    ) -> None:
        """订阅端共用的显示与按键循环：从 frame_queue 取图显示，采集按键/鼠标送 camera_send_queue。"""
        last_img: Optional[np.ndarray] = None
        error_msg: Optional[str] = None
        disconnected = False
        latest_m2p_ms: Optional[float] = None
        smoothed_m2p_ms: Optional[float] = None
        displayed_m2p_ms: Optional[float] = None
        m2p_ema_alpha = 0.08
        m2p_display_update_interval_s = 0.20
        last_m2p_display_update_ts = 0.0
        self._dragging = False
        self._pan_dragging = False
        self._left_dragging = False
        mouse_delta: List[float] = [0.0, 0.0]

        def on_mouse_scribe(event: int, x: int, y: int, flags: int, param: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._left_dragging = True
                self._last_mouse = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self._left_dragging = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._dragging = True
                self._last_mouse = (x, y)
            elif event == cv2.EVENT_RBUTTONUP:
                self._dragging = False
            elif event == cv2.EVENT_MBUTTONDOWN:
                self._pan_dragging = True
                self._last_mouse = (x, y)
            elif event == cv2.EVENT_MBUTTONUP:
                self._pan_dragging = False
            elif event == cv2.EVENT_MOUSEMOVE:
                dx = x - self._last_mouse[0]
                dy = y - self._last_mouse[1]
                self._last_mouse = (x, y)
                mouse_delta[0] += dx
                mouse_delta[1] += dy

        cv2.setMouseCallback(self.window_name, on_mouse_scribe)
        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) <= 0:
                        break
                except Exception:
                    break
                if key == 27:
                    break
                w = a = s = d = space = shift = alt = 0
                left = right = up = down = 0
                tab = plus = minus = bracket_left = bracket_right = q = e = toggle_axes = 0
                if _win_key is not None:
                    w = 1 if (_win_key.GetAsyncKeyState(_VK_W) & 0x8000) else 0
                    a = 1 if (_win_key.GetAsyncKeyState(_VK_A) & 0x8000) else 0
                    s = 1 if (_win_key.GetAsyncKeyState(_VK_S) & 0x8000) else 0
                    d = 1 if (_win_key.GetAsyncKeyState(_VK_D) & 0x8000) else 0
                    space = 1 if (_win_key.GetAsyncKeyState(_VK_SPACE) & 0x8000) else 0
                    shift = 1 if (_win_key.GetAsyncKeyState(_VK_SHIFT) & 0x8000) else 0
                    alt = 1 if (_win_key.GetAsyncKeyState(_VK_MENU) & 0x8000) else 0
                    left = 1 if (_win_key.GetAsyncKeyState(_VK_LEFT) & 0x8000) else 0
                    right = 1 if (_win_key.GetAsyncKeyState(_VK_RIGHT) & 0x8000) else 0
                    up = 1 if (_win_key.GetAsyncKeyState(_VK_UP) & 0x8000) else 0
                    down = 1 if (_win_key.GetAsyncKeyState(_VK_DOWN) & 0x8000) else 0
                    tab = 1 if (_win_key.GetAsyncKeyState(_VK_TAB) & 0x8000) else 0
                    plus = 1 if (_win_key.GetAsyncKeyState(_VK_OEM_PLUS) & 0x8000) else 0
                    minus = 1 if (_win_key.GetAsyncKeyState(_VK_OEM_MINUS) & 0x8000) else 0
                    bracket_left = 1 if (_win_key.GetAsyncKeyState(_VK_OEM_4) & 0x8000) else 0
                    bracket_right = 1 if (_win_key.GetAsyncKeyState(_VK_OEM_6) & 0x8000) else 0
                    q = 1 if (_win_key.GetAsyncKeyState(_VK_Q) & 0x8000) else 0
                    e = 1 if (_win_key.GetAsyncKeyState(_VK_E) & 0x8000) else 0
                    toggle_axes = 1 if (_win_key.GetAsyncKeyState(_VK_OEM3) & 0x8000) else 0
                elif _mac_key is not None:
                    w = 1 if _mac_key("w") else 0
                    a = 1 if _mac_key("a") else 0
                    s = 1 if _mac_key("s") else 0
                    d = 1 if _mac_key("d") else 0
                    space = 1 if _mac_key("space") else 0
                    shift = 1 if _mac_key("shift") else 0
                    alt = 1 if _mac_key("option") else 0
                    left = 1 if _mac_key("left") else 0
                    right = 1 if _mac_key("right") else 0
                    up = 1 if _mac_key("up") else 0
                    down = 1 if _mac_key("down") else 0
                    tab = 1 if _mac_key("tab") else 0
                    plus = 1 if _mac_key("plus") else 0
                    minus = 1 if _mac_key("minus") else 0
                    bracket_left = 1 if _mac_key("bracket_left") else 0
                    bracket_right = 1 if _mac_key("bracket_right") else 0
                    q = 1 if _mac_key("q") else 0
                    e = 1 if _mac_key("e") else 0
                    toggle_axes = 1 if _mac_key("grave") else 0
                dx, dy = mouse_delta[0], mouse_delta[1]
                mouse_delta[0], mouse_delta[1] = 0.0, 0.0
                key_cv2 = key if key != 255 else None
                inp = {
                    "w": w, "a": a, "s": s, "d": d,
                    "space": space, "shift": shift, "alt": alt,
                    "left": left, "right": right, "up": up, "down": down,
                    "mouse_dx": dx, "mouse_dy": dy,
                    "is_rotate": bool(self._dragging or self._left_dragging),
                    "is_pan": bool(self._pan_dragging),
                    "tab": tab, "plus": plus, "minus": minus,
                    "bracket_left": bracket_left, "bracket_right": bracket_right,
                    "q": q, "e": e, "toggle_axes": toggle_axes,
                    "__input_ts": time.perf_counter(),
                }
                for i, (key_spec, _) in enumerate(self._key_handlers):
                    inp[f"custom_{i}"] = 1 if _is_key_pressed(key_spec, key_cv2) else 0
                try:
                    while True:
                        camera_send_queue.get_nowait()
                except Empty:
                    pass
                try:
                    camera_send_queue.put_nowait(inp)
                except Exception:
                    pass

                got_item = False
                try:
                    item = frame_queue.get(timeout=0.05)
                    got_item = True
                except Empty:
                    item = None
                if got_item and item is None:
                    disconnected = True
                if got_item and item is not None:
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
                        error_msg = item[1]
                        stop_ev.set()
                    elif isinstance(item, tuple) and len(item) == 3 and item[0] == "frame":
                        if isinstance(item[1], np.ndarray):
                            last_img = item[1]
                            input_ts = item[2]
                            if isinstance(input_ts, (int, float)):
                                latest_m2p_ms = max(0.0, (time.perf_counter() - float(input_ts)) * 1000.0)
                                if smoothed_m2p_ms is None:
                                    smoothed_m2p_ms = latest_m2p_ms
                                else:
                                    smoothed_m2p_ms = (
                                        (1.0 - m2p_ema_alpha) * smoothed_m2p_ms
                                        + m2p_ema_alpha * latest_m2p_ms
                                    )
                    elif isinstance(item, np.ndarray):
                        last_img = item
                    else:
                        disconnected = True
                if not thread.is_alive() and error_msg is None and last_img is None:
                    disconnected = True
                if error_msg is not None:
                    show_img = _draw_error_image(self.width, self.height, error_msg, addr)
                    cv2.imshow(self.window_name, show_img)
                    continue
                if disconnected:
                    logger.info("Stream disconnected from %s", addr)
                    show_img = _draw_error_image(self.width, self.height, "连接已断开", addr)
                    cv2.imshow(self.window_name, show_img)
                    continue
                if last_img is not None:
                    show_img = last_img
                    # Keep display resolution in sync with current window size.
                    # Draw overlays after resize so text size stays visually stable.
                    try:
                        _x, _y, win_w, win_h = cv2.getWindowImageRect(self.window_name)
                        if win_w > 0 and win_h > 0:
                            self.width, self.height = int(win_w), int(win_h)
                    except Exception:
                        pass
                    if getattr(show_img, "ndim", 0) >= 2:
                        src_h, src_w = int(show_img.shape[0]), int(show_img.shape[1])
                        dst_w, dst_h = int(self.width), int(self.height)
                        if src_w > 0 and src_h > 0 and dst_w > 0 and dst_h > 0 and (src_w != dst_w or src_h != dst_h):
                            interp = cv2.INTER_AREA if (src_w > dst_w or src_h > dst_h) else cv2.INTER_LINEAR
                            show_img = cv2.resize(show_img, (dst_w, dst_h), interpolation=interp)
                    if latest_m2p_ms is not None:
                        show_img = show_img.copy()
                        now_ts = time.perf_counter()
                        shown_m2p_ms = smoothed_m2p_ms if smoothed_m2p_ms is not None else latest_m2p_ms
                        if (
                            displayed_m2p_ms is None
                            or (now_ts - last_m2p_display_update_ts) >= m2p_display_update_interval_s
                        ):
                            displayed_m2p_ms = shown_m2p_ms
                            last_m2p_display_update_ts = now_ts
                        label = f"M2P: {displayed_m2p_ms:.1f} ms"
                        img_h = int(show_img.shape[0]) if getattr(show_img, "ndim", 0) >= 2 else self.height
                        cv2.putText(
                            show_img,
                            label,
                            (10, max(20, img_h - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (180, 255, 180),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.imshow(self.window_name, show_img)
        finally:
            stop_ev.set()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _run_scribe_socket(self, addr: str) -> None:
        """连接远端 WebSocket（默认 socket 模式），经同一条连接收 H.264 帧与发操控，适合 SSH 隧道等跨网。"""
        if not _WS_AVAILABLE or not _STREAM_AVAILABLE or av is None:
            raise RuntimeError("Socket mode (H.264) requires: pip install holoviewer[stream] (websockets, aiortc, av)")
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required.")
        frame_queue: "Queue[Any]" = Queue(maxsize=2)
        camera_send_queue: "Queue[Dict[str, Any]]" = Queue(maxsize=8)
        stop_ev = threading.Event()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        thread = threading.Thread(
            target=_run_scribe_client_socket,
            args=(addr, frame_queue, stop_ev, camera_send_queue),
            daemon=True,
        )
        thread.start()
        self._scribe_display_loop(frame_queue, stop_ev, addr, thread, camera_send_queue)

    def _run_expose_webrtc(self, port: int, headless: bool = False) -> None:
        """本地渲染 + WebRTC 服务（--webrtc；适合局域网）。"""
        if not _STREAM_AVAILABLE:
            raise RuntimeError("Expose mode requires: pip install holoviewer[stream] (websockets, aiortc, av)")
        frame_queue: "Queue[FramePacket]" = Queue(maxsize=2)
        camera_command_queue: "Queue[CameraCommand]" = Queue(maxsize=8)
        stop_ev = threading.Event()
        server_thread = threading.Thread(
            target=_run_expose_server,
            args=(port, frame_queue, stop_ev, camera_command_queue, self),
            daemon=True,
        )
        server_thread.start()
        logger.info("WebSocket server thread started (port=%s); main thread running %s", port, "headless render loop" if headless else "render loop")
        try:
            if headless:
                import signal
                signal.signal(signal.SIGINT, lambda *a: stop_ev.set())
                self._run_local_headless(
                    frame_queue_for_stream=frame_queue,
                    camera_command_queue=camera_command_queue,
                    stop_ev=stop_ev,
                )
            else:
                self._run_local(
                    frame_queue_for_stream=frame_queue,
                    yield_to_ws_thread=True,
                    camera_command_queue=camera_command_queue,
                )
        finally:
            stop_ev.set()

    def _run_expose_socket(self, port: int, headless: bool = False) -> None:
        """本地渲染 + WebSocket 推流（默认 socket 模式，H.264），视频与操控同一条连接，适合 SSH 隧道等。"""
        if not _WS_AVAILABLE or not _STREAM_AVAILABLE or av is None:
            raise RuntimeError("Socket mode (H.264) requires: pip install holoviewer[stream] (websockets, aiortc, av)")
        if cv2 is None and not headless:
            raise RuntimeError("OpenCV (cv2) is required for non-headless.")
        frame_queue: "Queue[FramePacket]" = Queue(maxsize=2)
        camera_command_queue: "Queue[CameraCommand]" = Queue(maxsize=8)
        stop_ev = threading.Event()
        server_thread = threading.Thread(
            target=_run_expose_server_socket,
            args=(port, frame_queue, stop_ev, camera_command_queue, self),
            daemon=True,
        )
        server_thread.start()
        logger.info("WebSocket (socket) server thread started (port=%s); main thread running %s", port, "headless render loop" if headless else "render loop")
        try:
            if headless:
                import signal
                signal.signal(signal.SIGINT, lambda *a: stop_ev.set())
                self._run_local_headless(
                    frame_queue_for_stream=frame_queue,
                    camera_command_queue=camera_command_queue,
                    stop_ev=stop_ev,
                )
            else:
                self._run_local(
                    frame_queue_for_stream=frame_queue,
                    yield_to_ws_thread=True,
                    camera_command_queue=camera_command_queue,
                )
        finally:
            stop_ev.set()

    def _run_local_headless(
        self,
        frame_queue_for_stream: "Queue[FramePacket]",
        camera_command_queue: "Queue[CameraCommand]",
        stop_ev: threading.Event,
    ) -> None:
        """无窗口渲染循环：仅跑渲染管线并向 frame_queue 推帧，相机由 camera_command_queue 驱动（订阅端控制）；无 cv2。"""
        self.load_assets()
        pos, initial_forward = self.get_initial_pose()
        pos = pos.astype(np.float64)
        initial_forward = initial_forward.astype(np.float64)
        if np.linalg.norm(initial_forward) <= 1e-9:
            initial_forward = -pos / (np.linalg.norm(pos) + 1e-9)
        initial_forward = initial_forward / (np.linalg.norm(initial_forward) + 1e-9)
        yaw, pitch = self.axis_system.look_dir_to_angles(initial_forward)
        last = time.time()
        self._show_axes_flag = self.show_axes
        self.playing = True
        last_frame_input_ts: Optional[float] = None
        try:
            while not stop_ev.is_set():
                remote_cam: Optional[CameraCommand] = None
                try:
                    while True:
                        remote_cam = camera_command_queue.get_nowait()
                except Empty:
                    pass
                frame_input_ts: Optional[float] = None
                if remote_cam is not None:
                    pos, yaw, pitch = remote_cam[0].astype(np.float64).copy(), remote_cam[1], remote_cam[2]
                    frame_input_ts = remote_cam[3]
                    if isinstance(frame_input_ts, (int, float)):
                        last_frame_input_ts = float(frame_input_ts)
                if frame_input_ts is None:
                    frame_input_ts = last_frame_input_ts
                now = time.time()
                dt = now - last
                last = now
                override = self.on_frame_camera_update(pos, yaw, pitch, dt)
                if override is not None:
                    pos, yaw, pitch = override[0], override[1], override[2]
                forward, right, up = self.axis_system.compute_camera_axes(yaw, pitch)
                if self.playing:
                    time_delta = self._compute_time_delta(dt)
                    self.sim_time = wrap_time(
                        self.sim_time + time_delta, self.time_min, self.time_max
                    )
                (
                    view,
                    proj,
                    fov_x,
                    fov_y_rad,
                    znear,
                    zfar,
                ) = self._build_fps_camera(pos, forward, right, up)
                cam, full = self.build_camera(
                    view, proj, fov_x, fov_y_rad, znear, zfar, self.sim_time
                )
                image_tensor = self.render_frame(cam, self.sim_time)
                if image_tensor.shape[0] > 3:
                    image_tensor = image_tensor[:3]
                img_bgr = to_uint8_bgr(image_tensor)
                img_bgr = np.ascontiguousarray(img_bgr)
                if self._show_axes_flag:
                    self.draw_world_axes(img_bgr, full)
                try:
                    frame_queue_for_stream.put_nowait((img_bgr.copy(), self.sim_time, frame_input_ts))
                except Exception:
                    pass
                time.sleep(0.001)
        except Exception:
            logger.exception("Headless render loop error")

    def _run_local(
        self,
        frame_queue_for_stream: Optional["Queue[FramePacket]"] = None,
        yield_to_ws_thread: bool = False,
        camera_command_queue: Optional["Queue[CameraCommand]"] = None,
    ) -> None:
        """主循环：本地渲染、相机控制、显示。若提供 frame_queue_for_stream 则每帧推入队列（expose 模式）。
        yield_to_ws_thread=True 时每帧 sleep(0.001) 以让出 CPU 给 WebSocket 服务线程（macOS 上 OpenCV 须在主线程）。
        camera_command_queue 非空时，每帧取最新远端相机 (pos,yaw,pitch) 用于渲染（订阅端控制视角）。"""
        self.load_assets()

        pos, initial_forward = self.get_initial_pose()
        pos = pos.astype(np.float64)
        initial_forward = initial_forward.astype(np.float64)
        if np.linalg.norm(initial_forward) <= 1e-9:
            initial_forward = -pos / (np.linalg.norm(pos) + 1e-9)
        initial_forward = initial_forward / (np.linalg.norm(initial_forward) + 1e-9)
        forward_dir = initial_forward
        yaw, pitch = self.axis_system.look_dir_to_angles(forward_dir)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        def on_mouse(event, x, y, flags, param):
            nonlocal pos, yaw, pitch
            ctrl_pressed = False
            if _win_key is not None:
                ctrl_pressed = bool(_win_key.GetAsyncKeyState(_VK_CONTROL) & 0x8000)
            else:
                ctrl_pressed = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

            # 左键按下：与 Windows 一致，左键拖拽 = 旋转；若带 Ctrl 且未拖拽则松开时切换锁定
            if event == cv2.EVENT_LBUTTONDOWN:
                self._left_dragging = True
                self._last_mouse = (x, y)
                self._ctrl_click_pos = (x, y) if ctrl_pressed else None
                return
            if event == cv2.EVENT_LBUTTONUP and self._left_dragging:
                if self._ctrl_click_pos is not None:
                    cx, cy = self._ctrl_click_pos
                    if (x - cx) ** 2 + (y - cy) ** 2 < 25:
                        self._locked = not self._locked
                        if not self._locked:
                            self._dragging = False
                            self._pan_dragging = False
                self._left_dragging = False
                self._ctrl_click_pos = None
                return

            if self._locked:
                return

            if event == cv2.EVENT_RBUTTONDOWN:
                self._dragging = True
                self._last_mouse = (x, y)
            elif event == cv2.EVENT_RBUTTONUP:
                self._dragging = False
            elif event == cv2.EVENT_MBUTTONDOWN:
                self._pan_dragging = True
                self._last_mouse = (x, y)
            elif event == cv2.EVENT_MBUTTONUP:
                self._pan_dragging = False
            elif event == cv2.EVENT_MOUSEMOVE:
                dx = x - self._last_mouse[0]
                dy = y - self._last_mouse[1]
                self._last_mouse = (x, y)
                # 右键或左键拖拽 = 旋转视角（与 Windows 设计一致）
                if self._dragging or self._left_dragging:
                    yaw, pitch = self._update_yaw_pitch(yaw, pitch, dx, dy)
                elif self._pan_dragging:
                    _, right_vec, up_vec = self.axis_system.compute_camera_axes(
                        yaw, pitch
                    )
                    pos -= right_vec * dx * self.pan_sensitivity
                    pos += up_vec * dy * self.pan_sensitivity

        cv2.setMouseCallback(self.window_name, on_mouse)

        last = time.time()
        self._show_axes_flag = self.show_axes
        self.playing = True
        self._fps_frame_count = 0
        self._fps_time_accum = 0.0
        self._fps_value = 0.0
        last_frame_input_ts: Optional[float] = None

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                key_cv2 = key if key != 0xFF else None  # 0xFF = no key
                try:
                    vis = cv2.getWindowProperty(
                        self.window_name, cv2.WND_PROP_VISIBLE
                    )
                    if vis <= 0:
                        cv2.destroyAllWindows()
                        return
                except Exception:
                    cv2.destroyAllWindows()
                    return
                if key == 27:
                    break

                # Invoke registered key handlers (edge-triggered)
                for key_spec, callback in self._key_handlers:
                    hk = _key_spec_hashable(key_spec)
                    prev = self._key_handler_prev.get(hk, False)
                    curr = _is_key_pressed(key_spec, key_cv2)
                    self._key_handler_prev[hk] = curr
                    if curr and not prev:
                        try:
                            callback()
                        except Exception:
                            logger.exception("Key handler callback error")

                now = time.time()
                dt = now - last
                last = now
                if self.playing:
                    time_delta = self._compute_time_delta(dt)
                    self.sim_time = wrap_time(
                        self.sim_time + time_delta, self.time_min, self.time_max
                    )

                use_remote_camera = False
                frame_input_ts: Optional[float] = None
                if camera_command_queue is not None:
                    remote_cam: Optional[CameraCommand] = None
                    try:
                        while True:
                            remote_cam = camera_command_queue.get_nowait()
                    except Empty:
                        pass
                    if remote_cam is not None:
                        pos, yaw, pitch = remote_cam[0].astype(np.float64).copy(), remote_cam[1], remote_cam[2]
                        frame_input_ts = remote_cam[3]
                        if isinstance(frame_input_ts, (int, float)):
                            last_frame_input_ts = float(frame_input_ts)
                        use_remote_camera = True
                if frame_input_ts is None:
                    frame_input_ts = last_frame_input_ts

                move = np.zeros(3, dtype=np.float64)
                speed = self.move_speed
                forward, right, up = self.axis_system.compute_camera_axes(yaw, pitch)
                world_up = self.axis_system.world_up

                if not use_remote_camera:
                    w_down = a_down = s_down = d_down = False
                    space_down = shift_down = alt_down = False
                    tab_down = oem3_down = False
                    q_down = e_down = False
                    left_down = right_down = up_down = down_down = False

                    if _win_key is not None:
                        w_down = bool(_win_key.GetAsyncKeyState(_VK_W) & 0x8000)
                        a_down = bool(_win_key.GetAsyncKeyState(_VK_A) & 0x8000)
                        s_down = bool(_win_key.GetAsyncKeyState(_VK_S) & 0x8000)
                        d_down = bool(_win_key.GetAsyncKeyState(_VK_D) & 0x8000)
                        q_down = bool(_win_key.GetAsyncKeyState(_VK_Q) & 0x8000)
                        e_down = bool(_win_key.GetAsyncKeyState(_VK_E) & 0x8000)
                        space_down = bool(_win_key.GetAsyncKeyState(_VK_SPACE) & 0x8000)
                        shift_down = bool(_win_key.GetAsyncKeyState(_VK_SHIFT) & 0x8000)
                        alt_down = bool(_win_key.GetAsyncKeyState(_VK_MENU) & 0x8000)
                        tab_down = bool(_win_key.GetAsyncKeyState(_VK_TAB) & 0x8000)
                        oem3_down = bool(_win_key.GetAsyncKeyState(_VK_OEM3) & 0x8000)
                        left_down = bool(_win_key.GetAsyncKeyState(_VK_LEFT) & 0x8000)
                        right_down = bool(_win_key.GetAsyncKeyState(_VK_RIGHT) & 0x8000)
                        up_down = bool(_win_key.GetAsyncKeyState(_VK_UP) & 0x8000)
                        down_down = bool(_win_key.GetAsyncKeyState(_VK_DOWN) & 0x8000)
                    elif _mac_key is not None:
                        w_down = _mac_key("w")
                        a_down = _mac_key("a")
                        s_down = _mac_key("s")
                        d_down = _mac_key("d")
                        q_down = _mac_key("q")
                        e_down = _mac_key("e")
                        space_down = _mac_key("space")
                        shift_down = _mac_key("shift")
                        alt_down = _mac_key("option")
                        tab_down = _mac_key("tab")
                        oem3_down = _mac_key("grave")
                        left_down = _mac_key("left")
                        right_down = _mac_key("right")
                        up_down = _mac_key("up")
                        down_down = _mac_key("down")
                    else:
                        q_down = key == ord("q") or key == ord("Q")
                        e_down = key == ord("e") or key == ord("E")

                    # 如果视口已锁定，禁用大多数键盘移动和交互（但保留帧切换）
                    if not self._locked:
                        # 方向键旋转视野（左/右改 yaw，上/下改 pitch，与鼠标拖拽一致）
                        if _win_key is not None or _mac_key is not None:
                            if left_down:
                                yaw += self.arrow_rotate_speed
                            if right_down:
                                yaw -= self.arrow_rotate_speed
                            if up_down:
                                pitch -= self.arrow_rotate_speed
                                pitch = max(-math.pi / 2 + 1e-3, min(math.pi / 2 - 1e-3, pitch))
                            if down_down:
                                pitch += self.arrow_rotate_speed
                                pitch = max(-math.pi / 2 + 1e-3, min(math.pi / 2 - 1e-3, pitch))
                            if left_down or right_down or up_down or down_down:
                                forward, right, up = self.axis_system.compute_camera_axes(
                                    yaw, pitch
                                )
                        if _win_key is not None or _mac_key is not None:
                            if w_down:
                                move -= forward * speed
                            if s_down:
                                move += forward * speed
                            if a_down:
                                move -= right * speed
                            if d_down:
                                move += right * speed
                            if space_down:
                                move += world_up * speed
                            if shift_down or alt_down:
                                move -= world_up * speed

                            if tab_down and not self._prev_tab:
                                self.playing = not self.playing
                            self._prev_tab = tab_down

                            if oem3_down and not self._prev_oem3:
                                self._show_axes_flag = not self._show_axes_flag
                            self._prev_oem3 = oem3_down
                        else:
                            if key == ord("w") or key == ord("W"):
                                move -= forward * speed
                            if key == ord("s") or key == ord("S"):
                                move += forward * speed
                            if key == ord("a") or key == ord("A"):
                                move -= right * speed
                            if key == ord("d") or key == ord("D"):
                                move += right * speed
                            if key == ord(" "):
                                move += world_up * speed
                            if key == ord("`"):
                                self._show_axes_flag = not self._show_axes_flag
                            if key == 9:
                                self.playing = not self.playing

                        if key == ord("+") or key == ord("="):
                            self.play_rate = min(64.0, self.play_rate * 2.0)
                        if key == ord("-") or key == ord("_"):
                            self.play_rate = max(0.001, self.play_rate * 0.5)
                        if key == ord("9"):
                            self.fov_y_deg = max(5.0, min(120.0, self.fov_y_deg * 0.9))
                        if key == ord("0"):
                            self.fov_y_deg = max(5.0, min(120.0, self.fov_y_deg * 1.1))
                        if key == ord("["):
                            self.adjust_time(-self.time_step)
                        if key == ord("]"):
                            self.adjust_time(self.time_step)

                    if _win_key is not None or _mac_key is not None:
                        if q_down and not self._prev_q:
                            self.adjust_time(-self.time_step)
                        if e_down and not self._prev_e:
                            self.adjust_time(self.time_step)
                        self._prev_q = q_down
                        self._prev_e = e_down
                    else:
                        if q_down:
                            self.adjust_time(-self.time_step)
                        if e_down:
                            self.adjust_time(self.time_step)

                    move = self._transform_move_vector(
                        move, forward, right, up, world_up
                    )
                    pos += move

                # Update width/height from actual window size (e.g. after user resize).
                try:
                    _x, _y, w, h = cv2.getWindowImageRect(self.window_name)
                    if w > 0 and h > 0:
                        if w != self.width or h != self.height:
                            if self.show_window_resize_log:
                                print(f"🔄 Window resized: {w}x{h}")
                            self.width, self.height = w, h
                except Exception:
                    pass

                # Subclass hook: override camera (e.g. auto camera movement)
                override = self.on_frame_camera_update(pos, yaw, pitch, dt)
                if override is not None:
                    pos, yaw, pitch = override[0], override[1], override[2]
                    forward, right, up = self.axis_system.compute_camera_axes(yaw, pitch)

                (
                    view,
                    proj,
                    fov_x,
                    fov_y_rad,
                    znear,
                    zfar,
                ) = self._build_fps_camera(pos, forward, right, up)

                cam, full = self.build_camera(
                    view, proj, fov_x, fov_y_rad, znear, zfar, self.sim_time
                )
                image_tensor = self.render_frame(cam, self.sim_time)
                if image_tensor.shape[0] > 3:
                    image_tensor = image_tensor[:3]
                img_bgr = to_uint8_bgr(image_tensor)
                img_bgr = np.ascontiguousarray(img_bgr)

                if self.should_draw_axes():
                    self.draw_world_axes(img_bgr, full)

                self.on_frame_draw(img_bgr)

                if self.show_fps:
                    self._fps_frame_count += 1
                    self._fps_time_accum += dt
                    if self._fps_time_accum >= 1.0:
                        self._fps_value = self._fps_frame_count / self._fps_time_accum
                        self._fps_frame_count = 0
                        self._fps_time_accum = 0.0
                    fps_text = f"FPS: {self._fps_value:.1f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    color = (0, 255, 0)
                    (tw, th), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
                    tx, ty = self.width - tw - 10, self.height - 10
                    cv2.rectangle(img_bgr, (tx - 5, ty - th - 5), (tx + tw + 5, ty + baseline + 5), (0, 0, 0), -1)
                    cv2.putText(img_bgr, fps_text, (tx, ty), font, font_scale, color, thickness)

                suffix = self.get_window_title_suffix()
                if suffix:
                    cv2.setWindowTitle(self.window_name, self.window_name + suffix)

                if frame_queue_for_stream is not None:
                    try:
                        frame_queue_for_stream.put_nowait((img_bgr.copy(), self.sim_time, frame_input_ts))
                    except Exception:
                        pass
                cv2.imshow(self.window_name, img_bgr)
                if yield_to_ws_thread:
                    time.sleep(0.001)
        finally:
            cv2.destroyAllWindows()

    def _update_yaw_pitch(
        self, yaw: float, pitch: float, dx: float, dy: float
    ) -> Tuple[float, float]:
        yaw -= dx * 0.005
        pitch += dy * 0.005
        pitch = max(-math.pi / 2 + 1e-3, min(math.pi / 2 - 1e-3, pitch))
        return yaw, pitch

    def _apply_input(
        self,
        pos: np.ndarray,
        yaw: float,
        pitch: float,
        input_dict: Dict[str, Any],
    ) -> Tuple[np.ndarray, float, float]:
        """根据 input_dict（来自本地或远端）更新相机 (pos, yaw, pitch)。用于本地主循环与远端控制。"""
        pos = pos.astype(np.float64).copy()
        forward, right, up = self.axis_system.compute_camera_axes(yaw, pitch)
        world_up = self.axis_system.world_up
        speed = self.move_speed

        if input_dict.get("left"):
            yaw += self.arrow_rotate_speed
        if input_dict.get("right"):
            yaw -= self.arrow_rotate_speed
        if input_dict.get("up"):
            pitch -= self.arrow_rotate_speed
            pitch = max(-math.pi / 2 + 1e-3, min(math.pi / 2 - 1e-3, pitch))
        if input_dict.get("down"):
            pitch += self.arrow_rotate_speed
            pitch = max(-math.pi / 2 + 1e-3, min(math.pi / 2 - 1e-3, pitch))
        if input_dict.get("left") or input_dict.get("right") or input_dict.get("up") or input_dict.get("down"):
            forward, right, up = self.axis_system.compute_camera_axes(yaw, pitch)

        move = np.zeros(3, dtype=np.float64)
        if input_dict.get("w"):
            move -= forward * speed
        if input_dict.get("s"):
            move += forward * speed
        if input_dict.get("a"):
            move -= right * speed
        if input_dict.get("d"):
            move += right * speed
        if input_dict.get("space"):
            move += world_up * speed
        if input_dict.get("shift") or input_dict.get("alt"):
            move -= world_up * speed

        move = self._transform_move_vector(move, forward, right, up, world_up)
        pos += move

        if input_dict.get("is_rotate") and (input_dict.get("mouse_dx") is not None or input_dict.get("mouse_dy") is not None):
            yaw, pitch = self._update_yaw_pitch(
                yaw, pitch,
                float(input_dict.get("mouse_dx", 0)),
                float(input_dict.get("mouse_dy", 0)),
            )
        if input_dict.get("is_pan") and (input_dict.get("mouse_dx") is not None or input_dict.get("mouse_dy") is not None):
            _, right, up = self.axis_system.compute_camera_axes(yaw, pitch)
            pos -= right * float(input_dict.get("mouse_dx", 0)) * self.pan_sensitivity
            pos += up * float(input_dict.get("mouse_dy", 0)) * self.pan_sensitivity

        # 订阅端同步：播放/时间/速度/坐标轴（边沿检测，避免每帧重复）
        tab = input_dict.get("tab", 0)
        if tab and not getattr(self, "_remote_prev_tab", False):
            self.playing = not self.playing
        self._remote_prev_tab = bool(tab)

        plus = input_dict.get("plus", 0)
        if plus and not getattr(self, "_remote_prev_plus", False):
            self.play_rate = min(64.0, self.play_rate * 2.0)
        self._remote_prev_plus = bool(plus)

        minus = input_dict.get("minus", 0)
        if minus and not getattr(self, "_remote_prev_minus", False):
            self.play_rate = max(0.001, self.play_rate * 0.5)
        self._remote_prev_minus = bool(minus)

        bl = input_dict.get("bracket_left", 0)
        if bl and not getattr(self, "_remote_prev_bracket_left", False):
            self.adjust_time(-self.time_step)
        self._remote_prev_bracket_left = bool(bl)

        br = input_dict.get("bracket_right", 0)
        if br and not getattr(self, "_remote_prev_bracket_right", False):
            self.adjust_time(self.time_step)
        self._remote_prev_bracket_right = bool(br)

        qk = input_dict.get("q", 0)
        if qk and not getattr(self, "_remote_prev_q", False):
            self.adjust_time(-self.time_step)
        self._remote_prev_q = bool(qk)

        ek = input_dict.get("e", 0)
        if ek and not getattr(self, "_remote_prev_e", False):
            self.adjust_time(self.time_step)
        self._remote_prev_e = bool(ek)

        togg = input_dict.get("toggle_axes", 0)
        if togg and not getattr(self, "_remote_prev_toggle_axes", False):
            self._show_axes_flag = not self._show_axes_flag
        self._remote_prev_toggle_axes = bool(togg)

        # 注册的按键回调（边沿触发，与本地一致）
        prev_custom = getattr(self, "_remote_prev_custom", None)
        if prev_custom is None:
            self._remote_prev_custom = {}
        for i, (key_spec, callback) in enumerate(self._key_handlers):
            curr = bool(input_dict.get(f"custom_{i}", 0))
            prev = self._remote_prev_custom.get(i, False)
            self._remote_prev_custom[i] = curr
            if curr and not prev:
                try:
                    callback()
                except Exception:
                    logger.exception("Key handler callback error (remote)")

        return pos, yaw, pitch

    def _transform_move_vector(
        self,
        move: np.ndarray,
        forward: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        world_up: np.ndarray,
    ) -> np.ndarray:
        return move

    def _build_fps_camera(
        self,
        pos: np.ndarray,
        forward: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        view = np.eye(4, dtype=np.float64)
        basis = np.stack([right.astype(np.float64), -up.astype(np.float64), -forward.astype(np.float64)], axis=0)
        view[:3, :3] = basis
        view[:3, 3] = (-basis @ pos.reshape(3, 1)).flatten()

        fov_y = math.radians(self.fov_y_deg)
        aspect = max(1e-6, self.width / float(self.height))
        f = 1.0 / math.tan(0.5 * fov_y)
        znear, zfar = 0.2, 200.0
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = zfar / (zfar - znear)
        proj[2, 3] = 1.0
        proj[3, 2] = -(zfar * znear) / (zfar - znear)

        fov_x = 2.0 * math.atan(math.tan(0.5 * fov_y) * aspect)
        return view, proj, fov_x, fov_y, znear, zfar

    def draw_world_axes(
        self,
        img_bgr: np.ndarray,
        full: torch.Tensor,
        axis_len: float = 1.0,
    ) -> None:
        origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        origin_ndc = self.project_points_ndc(origin, full)[0]
        if not (
            np.isfinite(origin_ndc).all()
            and -1.0 <= origin_ndc[0] <= 1.0
            and -1.0 <= origin_ndc[1] <= 1.0
            and -1.0 <= origin_ndc[2] <= 1.0
        ):
            return

        axes = np.stack(
            [
                origin[0],
                origin[0] + np.array([axis_len, 0.0, 0.0], dtype=np.float32),
                origin[0] + np.array([0.0, axis_len, 0.0], dtype=np.float32),
                origin[0] + np.array([0.0, 0.0, axis_len], dtype=np.float32),
            ],
            axis=0,
        )
        ndc_pts = self.project_points_ndc(axes, full)

        def _draw(axis_ndc: np.ndarray, color: Tuple[int, int, int]) -> None:
            if not np.isfinite(axis_ndc).all():
                return
            ok, c0, c1 = self._clip_line_to_ndc(origin_ndc[:2], axis_ndc[:2])
            if not ok:
                return
            scr = self._ndc_to_screen(np.stack([c0, c1], axis=0))
            p0 = (int(scr[0, 0]), int(scr[0, 1]))
            p1 = (int(scr[1, 0]), int(scr[1, 1]))
            cv2.line(img_bgr, p0, p1, color, 2)

        _draw(ndc_pts[1], (0, 0, 255))
        _draw(ndc_pts[2], (0, 255, 0))
        _draw(ndc_pts[3], (255, 0, 0))

    def _ndc_to_screen(self, ndc_xy: np.ndarray) -> np.ndarray:
        x = (ndc_xy[:, 0] * 0.5 + 0.5) * self.width
        y = (ndc_xy[:, 1] * 0.5 + 0.5) * self.height
        return np.stack([x, y], axis=1)

    @staticmethod
    def _clip_line_to_ndc(
        p0: np.ndarray, p1: np.ndarray
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 + 1.0), (dx, 1.0 - x0), (-dy, y0 + 1.0), (dy, 1.0 - y0)):
            if p == 0.0:
                if q < 0.0:
                    return False, p0, p1
            else:
                r = q / p
                if p < 0.0:
                    if r > t1:
                        return False, p0, p1
                    if r > t0:
                        t0 = r
                else:
                    if r < t0:
                        return False, p0, p1
                    if r < t1:
                        t1 = r
        c0 = np.array([x0 + t0 * dx, y0 + t0 * dy], dtype=np.float32)
        c1 = np.array([x0 + t1 * dx, y0 + t1 * dy], dtype=np.float32)
        return True, c0, c1

    def get_initial_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (相机位置, 视线方向 forward_dir)，单位向量，默认朝向原点。"""
        pos = self.axis_system.initial_position()
        forward_dir = -pos / (np.linalg.norm(pos) + 1e-9)
        return pos.astype(np.float32), forward_dir.astype(np.float32)

    def adjust_time(self, delta: float) -> None:
        self.sim_time = wrap_time(
            self.sim_time + delta, self.time_min, self.time_max
        )
    
    def _compute_time_delta(self, dt: float) -> float:
        """
        Compute the time delta for sim_time update.
        Subclasses can override this to apply frame rate scaling.
        
        Args:
            dt: Real time delta in seconds
            
        Returns:
            Time delta to add to sim_time
        """
        return dt * self.play_rate

    @abstractmethod
    def load_assets(self) -> None:
        ...

    def build_camera(
        self,
        view: np.ndarray,
        proj: np.ndarray,
        fov_x: float,
        fov_y: float,
        znear: float,
        zfar: float,
        sim_time: float,
    ) -> Tuple[object, torch.Tensor]:
        """Default: return (cam_dict, full_tensor) with view/proj/full/width/height. Override if you need custom cam data for render_frame or axes."""
        full = (proj @ view).astype(np.float32)
        cam = {"view": view, "proj": proj, "full": full, "width": self.width, "height": self.height}
        return cam, torch.from_numpy(full)

    @abstractmethod
    def render_frame(self, cam: object, sim_time: float) -> torch.Tensor:
        ...

    def project_points_ndc(
        self, points_ws: np.ndarray, full: torch.Tensor
    ) -> np.ndarray:
        """World-space points to NDC (e.g. for axes/overlays). Override if you need a different NDC convention."""
        fn = full.cpu().numpy()
        n = points_ws.shape[0]
        pts = np.hstack([np.asarray(points_ws, dtype=np.float64), np.ones((n, 1))])
        clip = pts @ fn.T
        w = np.where(np.abs(clip[:, 3]) < 1e-8, 1e-8, clip[:, 3])
        return (clip[:, :3] / w.reshape(-1, 1)).astype(np.float32)
