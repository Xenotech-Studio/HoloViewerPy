# HoloViewer

一个用于构建交互式 3D 查看器的基类库，基于 OpenCV。

## 安装

使用可编辑模式安装（推荐用于开发）：

```bash
cd Utils/HoloViewerPy
pip install -e .
```

或者从项目根目录安装：

```bash
pip install -e Utils/HoloViewerPy
```

## 示例

用 PyVista（VTK）渲染一个立方体，支持多 GPU 后端（macOS Metal、Linux Vulkan、Windows DX12 等）：

```bash
cd packages/HoloViewerPy
pip install -e ".[sample]"   # 或 pip install -e . && pip install pyvista
python sample.py
```

- **WASD** 移动，**空格/Shift/Alt** 上下，**鼠标左键或右键拖拽** 旋转视角，**中键** 平移
- **Q/E** 上一帧/下一帧，**Tab** 播放/暂停，**`** 切换坐标轴，**Ctrl+左键点击** 切换锁定
- Windows 与 macOS 交互一致（macOS 通过 Quartz 支持按键按住连续移动）

### 网络推流 / 拉流（可选）

安装流扩展后，子类无需改代码即可通过命令行切换模式：

```bash
pip install -e ".[sample,stream]"   # stream 含 websockets + aiortc/av（socket 与 WebRTC 均需）
# 默认：WebSocket socket 模式（视频+操控同一条连接，适合 SSH 隧道等跨网）
python sample.py --expose-port 1145
python sample.py --subscribe 192.168.1.100:1145
# 可选：WebRTC 模式（适合局域网；跨网需自建 TURN）
python sample.py --expose-port 1145 --webrtc
python sample.py --subscribe 192.168.1.100:1145 --webrtc
```

- **默认 socket 模式**：`--expose-port` 在指定端口启动 WebSocket 服务，经同一条连接推送 H.264 帧并接收 JSON 操控。`--scribe HOST:PORT` 连接后只收帧、不本地渲染。无需 TURN，可经 SSH 反向隧道等跨网使用。
- **`--webrtc`**：改用 WebRTC 推流（信令 WebSocket + 媒体 ICE）。适合局域网；跨网需公网 TURN。
- **`--headless`**：与 `--expose-port` 同用时不创建 cv2 窗口，仅跑渲染与推流。

## 使用方法

安装后，可以直接导入使用：

```python
from holoviewer import HoloViewer, wrap_time, to_uint8_bgr
import numpy as np
import torch

class MyViewer(HoloViewer):
    def load_assets(self):
        # 加载你的资源
        pass
    
    def build_camera(self, view, proj, fov_x, fov_y, znear, zfar, sim_time):
        # 构建相机参数
        cam = ...  # 你的相机对象
        full = ...  # 你的完整变换矩阵
        return cam, full
    
    def render_frame(self, cam, sim_time):
        # 渲染单帧，返回 [C, H, W] 格式的 torch.Tensor
        return torch.zeros(3, 480, 640)
    
    def project_points_ndc(self, points_ws, full):
        # 将世界坐标投影到 NDC 空间
        return np.zeros((len(points_ws), 3))

# 创建并运行查看器
viewer = MyViewer(
    window_name="My Viewer",
    width=1280,
    height=720,
    fov_y_deg=60.0,
    axis_mode="Z_UP"
)
viewer.run()
```

## 功能特性

- **完整的查看器框架**：窗口管理、输入处理、相机控制
- **FPS 风格控制**：WASD 移动、鼠标拖拽视角
- **时间播放控制**：播放/暂停、速度调节、帧切换
- **坐标轴系统**：支持 Z_UP、Y_UP、mY_UP 等模式
- **世界坐标轴可视化**：可选的坐标轴显示

## 依赖

- Python >= 3.8
- numpy
- torch
- opencv-python

## 许可证

本项目采用 [MIT License](LICENSE)。
