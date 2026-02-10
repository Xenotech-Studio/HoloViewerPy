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
pip install -e ".[sample,stream]"   # stream 含 websockets, aiortc, av
# 本地渲染并开启 WebSocket+WebRTC 服务，等待客户端连接后推流
python sample.py --expose-port 1145
# 连接远端主机，仅接收 WebRTC 画面、不执行本地渲染
python sample.py --scribe 192.168.1.100:1145
```

- **`--expose-port PORT`**：在指定端口启动 WebSocket 服务器，准备好 WebRTC 推流；有客户端连接时进行 SDP/ICE 交换并推送当前渲染画面。
- **`--scribe HOST:PORT`**：连接 `HOST:PORT` 的 WebSocket，获取信令后建立 WebRTC 连接，只显示远端画面，跳过子类内部的 `load_assets`/`render_frame` 等渲染逻辑。

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
