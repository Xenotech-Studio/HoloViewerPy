#!/usr/bin/env python3
# HoloViewer sample — three-body N-body rendered with numpy + cv2 only (no GL context).
# Contract: render_frame(cam, sim_time) -> (3, H, W) float32 RGB in [0, 1].
from holoviewer import HoloViewer, is_client

if not is_client():
    import torch, numpy as np, cv2, platform

    def _detect_hw():
        if torch.cuda.is_available(): return f"GPU: {torch.cuda.get_device_name(0)}"
        try: return "CPU: " + next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        except Exception: return f"CPU: {platform.processor() or platform.machine()}"

    class SampleRenderer:
        def __init__(self, world_up=(0, 0, 1)):
            wu = np.asarray(world_up, dtype=np.float64); self._world_up = wu / (np.linalg.norm(wu) + 1e-9)
            self._body_pos = np.array([[-0.766, 0.026, -0.34], [0.684, -0.424, 0.31], [0.034, 0.376, 0.01]])
            self._body_vel = np.array([[-0.034, 0.432, -0.058], [-0.347, -0.358, 0.123], [0.358, -0.047, -0.065]])
            self._body_mass = np.array([0.94, 0.999, 1.06])
            self._body_radius = np.array([0.135, 0.138, 0.14])
            self._t_last = 0.0
            self._disc = [self._make_disc(c) for c in np.array([[255]*3, [216]*3, [176]*3], dtype=np.uint8)]
            self._sky_cache = None; self._frame_buf = None
            self._hw_label = _detect_hw()

        @staticmethod
        def _make_disc(base_bgr, n=256):
            # Lit sphere on a flat disc: ambient + Lambert + Blinn-Phong + Fresnel rim from fixed key light.
            y, x = np.mgrid[-1:1:n*1j, -1:1:n*1j]
            r2 = x*x + y*y; z = np.sqrt(np.clip(1 - r2, 0, 1))
            L = np.array([-0.4, -0.4, 0.825]); L /= np.linalg.norm(L)
            H = L + np.array([0, 0, 1]); H /= np.linalg.norm(H)
            ndl = np.clip(x*L[0] + y*L[1] + z*L[2], 0, 1)
            spec = np.clip(x*H[0] + y*H[1] + z*H[2], 0, 1) ** 80
            fres = (1 - z) ** 4
            sun = np.array([240, 245, 255], dtype=np.float32); base = base_bgr.astype(np.float32)
            col = (0.16 * base + 0.74 * ndl[..., None] * sun * (base / 255.)
                   + 0.95 * spec[..., None] * sun + 0.28 * fres[..., None] * sun).clip(0, 255).astype(np.uint8)
            return np.dstack([col, (r2 < 1).astype(np.uint8) * 255])

        def _sky(self, W, H, view, proj):
            # View-tracking gradient: per-pixel ray·world_up → LUT, at 1/4 res + bilinear upscale.
            W2, H2 = max(W // 4, 16), max(H // 4, 16)
            wu = self._world_up
            rd, ud, fd = view[0, :3] @ wu, -view[1, :3] @ wu, -view[2, :3] @ wu
            tx, ty = 1 / float(proj[0, 0]), 1 / float(proj[1, 1])
            if self._sky_cache is None or self._sky_cache[0] != (W2, H2):
                nx = np.arange(W2, dtype=np.float32) / max(W2 - 1, 1) * 2 - 1
                ny = np.arange(H2, dtype=np.float32) / max(H2 - 1, 1) * 2 - 1
                top = np.array([217, 140, 89], dtype=np.float32); bot = np.array([173, 158, 153], dtype=np.float32)
                ti = np.arange(256, dtype=np.float32) / 255.
                lut = (top * ti[:, None] + bot * (1 - ti[:, None])).clip(0, 255).astype(np.uint8)
                self._sky_cache = ((W2, H2), nx, ny, lut)
            _, nx, ny, lut = self._sky_cache
            xs, ys = nx * tx, ny * ty
            vert = rd * xs[None, :] - ud * ys[:, None] - fd
            nrm = np.sqrt(xs[None, :]**2 + ys[:, None]**2 + 1)
            t = ((vert / nrm).clip(-1, 1) + 1) * 0.5; t = t * t * (3 - 2 * t)
            return cv2.resize(lut[(t * 255).astype(np.uint8)], (W, H), interpolation=cv2.INTER_LINEAR)

        def _step(self, sim_time):
            # RK4: softened mutual gravity + harmonic well (-k·p binds system) + contact spring (F/m_i, Newton 3rd).
            def accel(p):
                d = p[None] - p[:, None]
                r2 = (d * d).sum(2) + 0.08**2; np.fill_diagonal(r2, 1)
                r = np.sqrt(r2)
                f = 0.5 * self._body_mass[None] / (r * r2)
                gap = np.maximum(0.40 - r, 0); f -= 3000.0 * gap * gap / (r * self._body_mass[:, None])
                np.fill_diagonal(f, 0)
                return (f[..., None] * d).sum(1) - 0.15 * p
            t, te = self._t_last, self._t_last + (sim_time - self._t_last) * 0.5
            while t < te:
                dt = min(0.012, te - t)
                p, v = self._body_pos, self._body_vel
                k1p, k1v = v, accel(p)
                k2p, k2v = v + 0.5*dt*k1v, accel(p + 0.5*dt*k1p)
                k3p, k3v = v + 0.5*dt*k2v, accel(p + 0.5*dt*k2p)
                k4p, k4v = v + dt*k3v, accel(p + dt*k3p)
                self._body_pos = p + (dt/6) * (k1p + 2*k2p + 2*k3p + k4p)
                self._body_vel = v + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
                t += dt
            self._t_last = sim_time

        @staticmethod
        def _project_segments(a, b, v2c, eps=0.01):
            # World-space segments → NDC pairs with near-plane (clip.w > eps) clipping in homogeneous space.
            pa = np.hstack([a.astype(np.float64), np.ones((len(a), 1))]) @ v2c
            pb = np.hstack([b.astype(np.float64), np.ones((len(b), 1))]) @ v2c
            mask = (pa[:, 3] > eps) | (pb[:, 3] > eps)
            ca, cb = mask & (pa[:, 3] <= eps), mask & (pb[:, 3] <= eps)
            if ca.any():
                t = ((eps - pa[ca, 3]) / (pb[ca, 3] - pa[ca, 3]))[:, None]; pa[ca] += t * (pb[ca] - pa[ca])
            if cb.any():
                t = ((eps - pb[cb, 3]) / (pa[cb, 3] - pb[cb, 3]))[:, None]; pb[cb] += t * (pa[cb] - pb[cb])
            return pa[:, :3] / pa[:, 3:4], pb[:, :3] / pb[:, 3:4], mask

        @staticmethod
        def _blit(img, tpl, cx, cy, r):
            d = 2 * r
            if d < 2: return
            tile = cv2.resize(tpl, (d, d), interpolation=cv2.INTER_AREA if d < tpl.shape[0] else cv2.INTER_LINEAR)
            H, W = img.shape[:2]
            ix, iy = int(cx - r), int(cy - r)
            x0, y0, x1, y1 = max(0, ix), max(0, iy), min(W, ix + d), min(H, iy + d)
            if x1 <= x0 or y1 <= y0: return
            src = tile[y0-iy:y1-iy, x0-ix:x1-ix]
            a = src[..., 3:4].astype(np.float32) / 255.
            img[y0:y1, x0:x1] = (src[..., :3].astype(np.float32) * a
                                 + img[y0:y1, x0:x1].astype(np.float32) * (1 - a)).astype(np.uint8)

        def render(self, cam, sim_time=0):
            view = np.asarray(cam["view"], dtype=np.float64); proj = np.asarray(cam["proj"], dtype=np.float64)
            W, H = int(cam["width"]), int(cam["height"])
            v2c = view.T @ proj  # row-vector world-to-clip; bypasses HoloViewer's mixed-convention cam["full"]
            self._step(sim_time)
            img = self._sky(W, H, view, proj)

            # ground grid: 21 × 21 lines on z=0; cv2.line clips internally so out-of-screen endpoints are fine.
            g = np.linspace(-10, 10, 21)
            z0, lo, hi = np.zeros(21), np.full(21, -10.), np.full(21, 10.)
            nda, ndb, mask = self._project_segments(np.r_[np.c_[g, lo, z0], np.c_[lo, g, z0]],
                                                    np.r_[np.c_[g, hi, z0], np.c_[hi, g, z0]], v2c)
            for i in range(len(mask)):
                if not mask[i]: continue
                p0 = (int((nda[i, 0]*0.5+0.5)*W), int((nda[i, 1]*0.5+0.5)*H))
                p1 = (int((ndb[i, 0]*0.5+0.5)*W), int((ndb[i, 1]*0.5+0.5)*H))
                cv2.line(img, p0, p1, (140, 132, 128), 1, cv2.LINE_AA)

            # bodies: back-to-front blit so closer ones occlude further ones.
            fy = 0.5 * H * float(proj[1, 1])
            clip = np.hstack([self._body_pos, np.ones((len(self._body_pos), 1))]) @ v2c
            ndc = clip[:, :3] / np.where(np.abs(clip[:, 3]) < 1e-8, 1e-8, clip[:, 3])[:, None]
            depth = view[2, :3] @ self._body_pos.T + view[2, 3]
            for i in np.argsort(-depth):
                if clip[i, 3] <= 0 or depth[i] < 0.01 or ndc[i, 2] > 1.0: continue
                self._blit(img, self._disc[i], (ndc[i, 0]*0.5+0.5)*W, (ndc[i, 1]*0.5+0.5)*H,
                           max(2, int(self._body_radius[i] * fy / depth[i])))

            # hardware label, bottom-right with 1px black shadow for legibility on any background
            tw = cv2.getTextSize(self._hw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            for dx, dy, c in ((1, 1, (0, 0, 0)), (0, 0, (210, 210, 210))):
                cv2.putText(img, self._hw_label, (W - tw - 10 + dx, H - 10 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

            # uint8 BGR HWC → float32 RGB CHW [0, 1] into a reused buffer (one copy, one multiply-cast).
            chw = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
            if self._frame_buf is None or self._frame_buf.shape != (3, H, W):
                self._frame_buf = torch.empty((3, H, W), dtype=torch.float32)
            np.multiply(chw, np.float32(1/255), out=self._frame_buf.numpy(), casting="unsafe")
            return self._frame_buf


class CubeSampleViewer(HoloViewer):
    def __init__(self, **kw):
        super().__init__(window_name="HoloViewer Three-Body Sample", axis_mode="Z_UP", show_axes=True, **kw)
        self._r = None

    def load_assets(self):
        self._r = SampleRenderer(world_up=self.axis_system.world_up)

    def render_frame(self, cam, sim_time):
        return self._r.render(cam, sim_time)


if __name__ == "__main__":
    CubeSampleViewer().run()
