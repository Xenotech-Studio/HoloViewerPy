#!/usr/bin/env python3

# This is a sample for using HoloViewer: subclass HoloViewer, plug in your own
# 3D/time renderer (anything with render(view_matrix, sim_time) -> frame tensor),
# and implement load_assets and render_frame. build_camera and project_points_ndc
# are optional (defaults: pass view/proj/full through; identity-like NDC).
from typing import Any, Optional
import numpy as np
import torch
try:
    import pyvista as pv
except ImportError as e:
    raise ImportError("Need PyVista: pip install pyvista") from e
from holoviewer import HoloViewer


# --- Your 3D/time rendering backend. You can simply ignore this code block, you will have your own.
#     Only requirement: render(view_matrix, sim_time) returns (3, H, W) float [0,1] RGB. 
class SampleRenderer:
    def _cammatrix_to_pvcam(self, full, width, height):
        full = np.asarray(full, dtype=np.float64)
        znear, zfar = 0.2, 200.0 # whatever this value is doesn't affect final render result.
        n = np.linalg.norm(full[1, :3])
        fov_y = 2 * np.arctan(1.0 / n) if n > 1e-8 else np.radians(50.0)
        aspect = max(1e-6, (width or self._last_w) / float(height or self._last_h))
        f = 1.0 / np.tan(0.5 * fov_y)
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0], proj[1, 1] = f / aspect, f
        proj[2, 2], proj[2, 3] = zfar / (zfar - znear), 1.0
        proj[3, 2] = -(zfar * znear) / (zfar - znear)
        view = np.linalg.inv(proj) @ full
        pos = np.linalg.inv(view)[:3, 3].astype(np.float64)
        foc = pos + view[2, :3].astype(np.float64)
        up = (-view[1, :3]).astype(np.float64)
        view_angle_deg = np.degrees(fov_y)
        return pos, foc, up, view_angle_deg

    def __init__(self):
        self._last_w, self._last_h = 800, 600
        pl = self._plotter = pv.Plotter(off_screen=True, window_size=(self._last_w, self._last_h))
        pl.set_background((0.6, 0.62, 0.68))
        s = pv.Sphere(radius=400, center=(0,0,0), theta_resolution=64, phi_resolution=48)
        s.points = -np.array(s.points)
        s.compute_normals(inplace=True)
        t = np.clip((s.points[:, 2]/400 + 0.18) / 0.36, 0, 1)
        t = t * t * (3 - 2 * t)
        s["sky_rgb"] = np.clip((1-t)[:, None]*np.array((0.6, 0.62, 0.68)) + t[:, None]*np.array((0.35, 0.55, 0.85)), 0, 1).astype(np.float32)
        a = pl.add_mesh(s, scalars="sky_rgb", rgb=True, lighting=False, smooth_shading=True)
        try: a.GetProperty().SetDepthWrite(0)
        except Exception: pass
        n = 21
        x, y = np.mgrid[-10:10:n*1j, -10:10:n*1j]
        pts = np.column_stack([x.ravel(), y.ravel(), np.zeros(n*n)]).astype(np.float64)
        c = np.array([c for j in range(n) for i in range(n-1) for c in [2, j*n+i, j*n+i+1]] + [c for i in range(n) for j in range(n-1) for c in [2, j*n+i, (j+1)*n+i]], dtype=np.int64)
        g = pl.add_mesh(pv.PolyData(pts, lines=c), color=(0.5, 0.52, 0.55), line_width=1, lighting=False)
        try: g.GetProperty().SetRenderLinesAsTubes(False)
        except Exception: pass
        self._body_actors = []
        self._body_pos = np.array([[-0.766, 0.026, -0.34], [0.684, -0.424, 0.31], [0.034, 0.376, 0.01]], dtype=np.float64).copy()
        self._body_vel = np.array([[-0.034, 0.432, -0.058], [-0.347, -0.358, 0.123], [0.358, -0.047, -0.065]], dtype=np.float64).copy()
        self._body_masses = np.array([0.94, 0.999, 1.06], dtype=np.float64)
        self._last_sim_time = 0
        r, C = np.array([0.135, 0.138, 0.14], dtype=np.float64), ["#ffffff", "#d8d8d8", "#b0b0b0"]
        for i in range(3): self._body_actors.append(pl.add_mesh(pv.Sphere(center=(0,0,0), radius=r[i], theta_resolution=32, phi_resolution=24), color=C[i], specular=0.12, specular_power=6, smooth_shading=True))

    def render(self, full, sim_time=0, width=None, height=None):
        if width is not None and height is not None and (width, height) != (self._last_w, self._last_h):
            self._last_w, self._last_h = width, height
            self._plotter.window_size = (width, height)
        pos, foc, up, view_angle_deg = self._cammatrix_to_pvcam(full, width, height)
        def accel(pos):
            d = pos[None, :, :] - pos[:, None, :]
            d2 = (d * d).sum(axis=2) + 0.08**2
            np.fill_diagonal(d2, 1)
            f = 0.5 * self._body_masses[None, :] / (np.sqrt(d2) * d2)
            np.fill_diagonal(f, 0)
            return (f[:, :, None] * d).sum(axis=1)
        t, te = self._last_sim_time, self._last_sim_time + (sim_time - self._last_sim_time) * 0.5
        while t < te:
            dt = min(0.012, te - t)
            p, v = self._body_pos, self._body_vel
            k1p, k1v = v.copy(), accel(p)
            k2p, k2v = v + 0.5*dt*k1v, accel(p + 0.5*dt*k1p)
            k3p, k3v = v + 0.5*dt*k2v, accel(p + 0.5*dt*k2p)
            k4p, k4v = v + dt*k3v, accel(p + dt*k3p)
            self._body_pos += (dt/6) * (k1p + 2*k2p + 2*k3p + k4p)
            self._body_vel += (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
            t += dt
        self._last_sim_time = sim_time
        for i in range(3): self._body_actors[i].SetPosition(*self._body_pos[i])
        cam = self._plotter.camera
        cam.position, cam.focal_point, cam.up, cam.view_angle = pos, foc, up, view_angle_deg
        self._plotter.render()
        return torch.from_numpy(np.transpose(self._plotter.screenshot(return_img=True).astype(np.float32)/255, (2,0,1)))


# --- HoloViewer subclass: wire your renderer here. ---
class CubeSampleViewer(HoloViewer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            window_name="HoloViewer Cube Sample",
            axis_mode="Z_UP",
            show_axes=True,
            **kwargs,
        )
        self._renderer: Optional[SampleRenderer] = None

    # Create your renderer (called once when viewer starts).
    def load_assets(self) -> None:
        # this sample renderer dont need data loading, just init it.
        self._renderer = SampleRenderer()

    # Main rendering loop: this is where your own code does the work. 
    # Called each frame with the current view and sim_time. 
    # Return (3, H, W) float [0,1].
    def render_frame(self, cam: Any, sim_time: float) -> torch.Tensor:
        # "cam" object has: "view" (4x4), "proj" (4x4), "full" (4x4 = proj@view), "width", "height".
        # Override build_camera in a subclass to pass a custom cam dict if you need more fields.
        return self._renderer.render(cam["full"], sim_time, cam["width"], cam["height"])
    


if __name__ == "__main__":
    CubeSampleViewer().run()
