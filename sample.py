#!/usr/bin/env python3

# Sample for HoloViewer: subclass HoloViewer, plug in your own 3D/time renderer,
# implement load_assets and render_frame. build_camera and project_points_ndc are optional.
#
# This sample uses only numpy + cv2 — no OpenGL context, no system libs.
# The contract is: render_frame(cam, sim_time) -> (3, H, W) float [0,1] RGB torch tensor.
from typing import Any, Optional
from holoviewer import HoloViewer, is_client

if not is_client():

    # --- Your 3D/time rendering backend. You can simply ignore this code block, you will have your own.
    #     Only requirement: render(cam, sim_time) returns (3, H, W) float [0,1] RGB.
    import torch
    import numpy as np
    import cv2

    class SampleRenderer:
        def __init__(self, world_up=(0.0, 0.0, 1.0)):
            wu = np.asarray(world_up, dtype=np.float64)
            self._world_up = wu / max(float(np.linalg.norm(wu)), 1e-9)
            self._last_w, self._last_h = 800, 600
            self._sky_nxny = None  # cached (W2, H2, nx_unit, ny_unit, lut) for sky vectorization
            self._frame_buf = None  # (3, H, W) torch.float32 output buffer, reused across frames
            # three-body initial state (positions / velocities / masses)
            self._body_pos = np.array([[-0.766, 0.026, -0.34], [0.684, -0.424, 0.31], [0.034, 0.376, 0.01]], dtype=np.float64).copy()
            self._body_vel = np.array([[-0.034, 0.432, -0.058], [-0.347, -0.358, 0.123], [0.358, -0.047, -0.065]], dtype=np.float64).copy()
            self._body_masses = np.array([0.94, 0.999, 1.06], dtype=np.float64)
            self._body_radii = np.array([0.135, 0.138, 0.14], dtype=np.float64)
            self._body_colors_bgr = np.array([[255, 255, 255], [216, 216, 216], [176, 176, 176]], dtype=np.uint8)
            self._last_sim_time = 0.0
            # precompute one lit-sphere BGRA template per body color (fake headlight + soft specular)
            self._disc_templates = [self._make_lit_disc(c) for c in self._body_colors_bgr]

        @staticmethod
        def _make_lit_disc(base_bgr, size=256):
            # Fake-3D lit sphere on a flat disc:
            #   - ambient base + Lambertian diffuse from a key light (sun, upper-left)
            #   - Blinn-Phong specular highlight (half-vector against view = +z)
            #   - Fresnel rim (Schlick) to brighten silhouette
            # Light is fixed in disc-template space, so the highlight pose doesn't track world rotation —
            # acceptable because for a sphere the lit appearance is rotation-invariant by symmetry.
            y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
            r2 = x*x + y*y
            z = np.sqrt(np.clip(1 - r2, 0, 1))

            Lx, Ly, Lz = -0.40, -0.40, 0.825
            ln = (Lx*Lx + Ly*Ly + Lz*Lz) ** 0.5
            Lx, Ly, Lz = Lx/ln, Ly/ln, Lz/ln
            n_dot_l = np.clip(x*Lx + y*Ly + z*Lz, 0, 1)

            # Blinn-Phong half-vector: H = normalize(L + V), V = +z
            Hx, Hy, Hz = Lx, Ly, Lz + 1.0
            hn = (Hx*Hx + Hy*Hy + Hz*Hz) ** 0.5
            Hx, Hy, Hz = Hx/hn, Hy/hn, Hz/hn
            spec = np.clip(x*Hx + y*Hy + z*Hz, 0, 1) ** 80

            fresnel = (1.0 - z) ** 4

            sun = np.array([240, 245, 255], dtype=np.float32)  # warm white (BGR)
            base = base_bgr.astype(np.float32)
            base_norm = base / 255.0

            col = (0.16 * base
                   + 0.74 * n_dot_l[..., None] * sun * base_norm
                   + 0.95 * spec[..., None] * sun
                   + 0.28 * fresnel[..., None] * sun)
            rgb = col.clip(0, 255).astype(np.uint8)
            alpha = (r2 < 1).astype(np.uint8) * 255
            return np.dstack([rgb, alpha])

        def _make_sky(self, W, H, view, proj):
            # View-tracking gradient: each pixel's ray (in world) is dotted with world_up.
            # +1 → blue zenith, 0 → horizon, -1 → grey ground (looking straight down).
            # Sky is smooth so we render at 1/4 resolution + bilinear upscale.
            # The color sample is a 256-entry uint8 LUT, kept around between frames.
            W2, H2 = max(W // 4, 16), max(H // 4, 16)
            right_w = view[0, :3]
            up_w = -view[1, :3]
            forward_w = -view[2, :3]
            wu = self._world_up
            rdot = float(right_w @ wu)
            udot = float(up_w @ wu)
            fdot = float(forward_w @ wu)
            tx = 1.0 / float(proj[0, 0])  # = tan(fov_x / 2)
            ty = 1.0 / float(proj[1, 1])  # = tan(fov_y / 2)
            if self._sky_nxny is None or self._sky_nxny[0] != (W2, H2):
                nx0 = np.arange(W2, dtype=np.float32) / max(W2 - 1, 1) * 2 - 1
                ny0 = np.arange(H2, dtype=np.float32) / max(H2 - 1, 1) * 2 - 1
                top = np.array([217, 140, 89], dtype=np.float32)   # blue (looking up)
                bot = np.array([173, 158, 153], dtype=np.float32)  # ground (looking down)
                ti = np.arange(256, dtype=np.float32) / 255.0
                lut = (top[None, :] * ti[:, None] + bot[None, :] * (1 - ti[:, None])).clip(0, 255).astype(np.uint8)
                self._sky_nxny = ((W2, H2), nx0, ny0, lut)
            _, nx0, ny0, lut = self._sky_nxny
            nx = nx0 * tx
            ny = ny0 * ty
            vert = rdot * nx[None, :] - udot * ny[:, None] - fdot
            nrm = np.sqrt(nx[None, :] ** 2 + ny[:, None] ** 2 + 1.0)
            t = ((vert / nrm).clip(-1.0, 1.0) + 1.0) * 0.5  # [0, 1]
            t = t * t * (3.0 - 2.0 * t)
            idx = (t * 255).astype(np.uint8)
            small = lut[idx]  # (H2, W2, 3) uint8 BGR
            return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

        @staticmethod
        def _project(pts_world, view2clip):
            pts = np.hstack([pts_world.astype(np.float64), np.ones((len(pts_world), 1))])
            clip = pts @ view2clip
            w = np.where(np.abs(clip[:, 3]) < 1e-8, 1e-8, clip[:, 3])
            return clip[:, :3] / w[:, None], clip[:, 3]

        @staticmethod
        def _project_segments(seg_a, seg_b, view2clip, eps=0.01):
            """World-space line segments → NDC pairs, with near-plane clipping in homogeneous space.
            Returns (ndc_a, ndc_b, mask) where mask[i]=True means segment i is at least partially visible.
            Endpoints that fall behind the near plane (clip.w <= eps) are interpolated onto w = eps."""
            pa = np.hstack([seg_a.astype(np.float64), np.ones((len(seg_a), 1))])
            pb = np.hstack([seg_b.astype(np.float64), np.ones((len(seg_b), 1))])
            ca = pa @ view2clip
            cb = pb @ view2clip
            wa, wb = ca[:, 3], cb[:, 3]
            mask = (wa > eps) | (wb > eps)
            # for segments crossing the near plane, interpolate the behind endpoint to w = eps
            cross_a = mask & (wa <= eps)  # a is behind, b is in front
            cross_b = mask & (wb <= eps)  # b is behind, a is in front
            if np.any(cross_a):
                t = (eps - wa[cross_a]) / (wb[cross_a] - wa[cross_a])
                ca[cross_a] = ca[cross_a] + t[:, None] * (cb[cross_a] - ca[cross_a])
            if np.any(cross_b):
                t = (eps - wb[cross_b]) / (wa[cross_b] - wb[cross_b])
                cb[cross_b] = cb[cross_b] + t[:, None] * (ca[cross_b] - cb[cross_b])
            ndc_a = ca[:, :3] / np.where(np.abs(ca[:, 3]) < 1e-8, 1e-8, ca[:, 3])[:, None]
            ndc_b = cb[:, :3] / np.where(np.abs(cb[:, 3]) < 1e-8, 1e-8, cb[:, 3])[:, None]
            return ndc_a, ndc_b, mask

        @staticmethod
        def _to_px(ndc, W, H, clamp=100.0):
            nx = max(-clamp, min(clamp, float(ndc[0])))
            ny = max(-clamp, min(clamp, float(ndc[1])))
            return (int((nx * 0.5 + 0.5) * W), int((ny * 0.5 + 0.5) * H))

        @staticmethod
        def _clip_line_ndc(p0, p1, lo=-1.0, hi=1.0):
            """Liang-Barsky clip of a 2D segment against the [lo, hi]^2 NDC box. Returns clipped
            endpoints or None if fully outside. Clipping in NDC (before pixel conversion) preserves
            the true line slope when an endpoint is far off-screen (e.g. just past the near plane)."""
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            dx, dy = x1 - x0, y1 - y0
            t0, t1 = 0.0, 1.0
            for p, q in ((-dx, x0 - lo), (dx, hi - x0), (-dy, y0 - lo), (dy, hi - y0)):
                if p == 0:
                    if q < 0: return None
                else:
                    r = q / p
                    if p < 0:
                        if r > t1: return None
                        if r > t0: t0 = r
                    else:
                        if r < t0: return None
                        if r < t1: t1 = r
            return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)

        def _step_sim(self, sim_time):
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

        @staticmethod
        def _blit_disc(img, template, cx, cy, r):
            d = 2 * r
            if d < 2: return
            tile = cv2.resize(template, (d, d), interpolation=cv2.INTER_AREA if d < template.shape[0] else cv2.INTER_LINEAR)
            H, W = img.shape[:2]
            x0, y0 = int(cx - r), int(cy - r)
            sx0, sy0 = max(0, -x0), max(0, -y0)
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(W, x0 + d), min(H, y0 + d)
            if dx1 <= dx0 or dy1 <= dy0: return
            sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
            tile_rgb = tile[sy0:sy1, sx0:sx1, :3].astype(np.float32)
            tile_a = tile[sy0:sy1, sx0:sx1, 3:4].astype(np.float32) * (1.0 / 255.0)
            dst = img[dy0:dy1, dx0:dx1].astype(np.float32)
            img[dy0:dy1, dx0:dx1] = (tile_rgb * tile_a + dst * (1 - tile_a)).astype(np.uint8)

        def render(self, cam, sim_time=0):
            view = np.asarray(cam["view"], dtype=np.float64)
            proj = np.asarray(cam["proj"], dtype=np.float64)
            W, H = int(cam["width"]), int(cam["height"])
            self._last_w, self._last_h = W, H
            # Row-vector world-to-clip transform. HoloViewer's cam["full"] uses a mixed convention that
            # produces out-of-range NDC, so we bypass it and compute the proper transform ourselves.
            view2clip = view.T @ proj

            self._step_sim(sim_time)
            img = self._make_sky(W, H, view, proj)

            # ground grid (20x20 lines on the z=0 plane)
            n = 21
            g = np.linspace(-10, 10, n)
            seg_a = np.concatenate([np.stack([g, np.full(n, -10.0), np.zeros(n)], 1),
                                    np.stack([np.full(n, -10.0), g, np.zeros(n)], 1)])
            seg_b = np.concatenate([np.stack([g, np.full(n, +10.0), np.zeros(n)], 1),
                                    np.stack([np.full(n, +10.0), g, np.zeros(n)], 1)])
            nda, ndb, mask = self._project_segments(seg_a, seg_b, view2clip)
            for i in range(len(seg_a)):
                if not mask[i]: continue
                clipped = self._clip_line_ndc(nda[i], ndb[i])
                if clipped is None: continue
                (cx0, cy0), (cx1, cy1) = clipped
                p0 = (int((cx0 * 0.5 + 0.5) * W), int((cy0 * 0.5 + 0.5) * H))
                p1 = (int((cx1 * 0.5 + 0.5) * W), int((cy1 * 0.5 + 0.5) * H))
                cv2.line(img, p0, p1, (140, 132, 128), 1, cv2.LINE_AA)

            # bodies — back-to-front blit so closer ones occlude further ones
            focal_y_px = 0.5 * H * float(proj[1, 1])  # = 0.5 * H / tan(fov_y / 2)
            ndc_c, w_c = self._project(self._body_pos, view2clip)
            depth = view[2, :3] @ self._body_pos.T + view[2, 3]  # camera-space z, +ve when in front
            for i in np.argsort(-depth):
                if w_c[i] <= 0 or depth[i] < 0.01: continue
                if not (ndc_c[i, 2] <= 1.0): continue  # past far plane
                cx = (ndc_c[i, 0] * 0.5 + 0.5) * W
                cy = (ndc_c[i, 1] * 0.5 + 0.5) * H
                r_px = max(2, int(self._body_radii[i] * focal_y_px / depth[i]))
                self._blit_disc(img, self._disc_templates[i], cx, cy, r_px)

            # (H, W, 3) uint8 BGR → (3, H, W) float32 RGB [0, 1], minimizing temporaries.
            # Order matters: do the channel-reverse + transpose on uint8 first (a single 6 MB copy),
            # then multiply-cast directly into a reused float buffer (one 25 MB write).
            chw_rgb = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
            if self._frame_buf is None or self._frame_buf.shape != (3, H, W):
                self._frame_buf = torch.empty((3, H, W), dtype=torch.float32)
            np.multiply(chw_rgb, np.float32(1.0 / 255.0), out=self._frame_buf.numpy(), casting="unsafe")
            return self._frame_buf


# --- HoloViewer subclass: wire your renderer here. ---
class CubeSampleViewer(HoloViewer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            window_name="HoloViewer Three-Body Sample",
            axis_mode="Z_UP",
            show_axes=True,
            **kwargs,
        )
        self._renderer: Optional[SampleRenderer] = None

    # Create your renderer (called once when viewer starts).
    def load_assets(self) -> None:
        # this sample renderer doesn't need data loading, just init it.
        # pass world_up so the sky gradient knows which direction is "up" in the world.
        self._renderer = SampleRenderer(world_up=self.axis_system.world_up)

    # Main rendering loop: called each frame with the current camera and sim_time.
    # Return (3, H, W) float [0,1] RGB.
    def render_frame(self, cam: Any, sim_time: float):
        # "cam" dict has: "view" (4x4), "proj" (4x4), "full" (4x4 = proj @ view), "width", "height".
        # Override build_camera in a subclass to pass a custom cam dict if you need more fields.
        return self._renderer.render(cam, sim_time)


if __name__ == "__main__":
    CubeSampleViewer().run()
