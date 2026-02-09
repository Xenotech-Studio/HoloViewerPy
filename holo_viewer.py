import math
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

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
        _kVK_Shift = 0x38
        _kVK_Option = 0x3A

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
            "shift": _kVK_Shift,
            "option": _kVK_Option,
        }

        def _mac_key(vk_name: str) -> bool:
            if vk_name not in _mac_key_map:
                return False
            return _mac_key_state(_mac_key_map[vk_name])

        _mac_key = _mac_key
    except Exception:  # pragma: no cover
        _mac_key = None


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


class HoloViewer(ABC):
    def __init__(
        self,
        window_name: str,
        width: int,
        height: int,
        fov_y_deg: float,
        axis_mode: str = "Z_UP",
        canonical_initial: Optional[np.ndarray] = None,
        axis_initial_overrides: Optional[Dict[str, np.ndarray]] = None,
        pan_sensitivity: float = 0.01,
        move_speed: float = 0.05,
        show_axes: bool = True,
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

    def run(self) -> None:
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

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
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

                now = time.time()
                dt = now - last
                last = now
                if self.playing:
                    time_delta = self._compute_time_delta(dt)
                    self.sim_time = wrap_time(
                        self.sim_time + time_delta, self.time_min, self.time_max
                    )

                move = np.zeros(3, dtype=np.float64)
                speed = self.move_speed
                forward, right, up = self.axis_system.compute_camera_axes(yaw, pitch)
                world_up = self.axis_system.world_up

                w_down = a_down = s_down = d_down = False
                space_down = shift_down = alt_down = False
                tab_down = oem3_down = False
                q_down = e_down = False

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
                else:
                    q_down = key == ord("q") or key == ord("Q")
                    e_down = key == ord("e") or key == ord("E")

                # 如果视口已锁定，禁用大多数键盘移动和交互（但保留帧切换）
                if not self._locked:
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

                cv2.imshow(self.window_name, img_bgr)
        finally:
            cv2.destroyAllWindows()

    def _update_yaw_pitch(
        self, yaw: float, pitch: float, dx: float, dy: float
    ) -> Tuple[float, float]:
        yaw -= dx * 0.005
        pitch += dy * 0.005
        pitch = max(-math.pi / 2 + 1e-3, min(math.pi / 2 - 1e-3, pitch))
        return yaw, pitch

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

    @abstractmethod
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
        ...

    @abstractmethod
    def render_frame(self, cam: object, sim_time: float) -> torch.Tensor:
        ...

    @abstractmethod
    def project_points_ndc(
        self, points_ws: np.ndarray, full: torch.Tensor
    ) -> np.ndarray:
        ...
