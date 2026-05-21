# Tiny1-C Thermal Scope PC Simulator

这是一个用于毕业设计展示和调试的 PC 端热成像瞄具实机仿真工具。Tiny1-C 通过 USB-UVC 被电脑识别为摄像头后，本工具用 PySide6 + OpenCV 读取热图，并在电脑端模拟后续 ESP32-S3 + FPGA 系统的显示效果。

当前工具的目标是“显示效果仿真”，不是实物 AMOLED 实测。

## 功能定位

- 输入：Tiny1-C USB 摄像头、Mock 热图场景，或普通可见光摄像头演示。
- 内部数据：统一 14-bit `raw14`（`uint16`, 0..16383）。
- 显示：横向热成像瞄具屏幕，白热/黑热增强和纯 Outline 模式。
- 增强：AGC/DDE-like 热图增强，突出目标细节和弱纹理。
- 双 Profile：`Thermal Tiny1-C` 使用热图 raw/Y16 主链路；`Visible Demo` 仅做普通摄像头黑底亮边演示。
- 叠加：中心十字分划、倍率、增强等级、菜单、热点标记。
- 候选目标：基于 candidate mask、轮廓和几何规则进行人物候选高亮。
- 操作：两枚虚拟物理按键模拟真实瞄具交互。
- 输出：截图保存最终瞄具屏幕画面。

## 重要说明

- 白热/黑热与 Outline 效果均为 PC 端软件仿真，不代表厂商专有算法细节公开。
- 可见光摄像头没有温差信息，`Visible Demo` 不能代表真实热成像增强效果，只用于调试和演示轮廓风格。
- 人物候选为规则辅助分类，不是训练完成的 AI 人体识别。
- 不显示真实摄氏温度；界面使用的是相对热强度和显示增强结果。
- 当前工具用于模拟后续 ESP32-S3 显示层效果，可与 FPGA 输出 `thumb + candidate mask + metadata` 的毕业设计方案对应。

## 运行

```powershell
cd F:\final_design\thermal_sim_v3
pip install -r requirements.txt
python main.py
```

macOS 本地虚拟环境示例：

```bash
cd "/Users/stephengu/Desktop/毕业设计/Thermal_sim"
source .venv/bin/activate
python main.py
```

默认使用 Mock Person 场景，方便直接验证人物候选轮廓高亮。

## 界面操作

顶部工具条：

- `Source`：选择 Mock 或 UVC 摄像头。
- `Scene`：选择 Mock 场景。
- `Profile`：
  - `Thermal Tiny1-C`：Tiny1-C/Mock 热图主模式，优先使用 Y16/raw14 数据。
  - `Visible Demo`：普通可见光摄像头轮廓演示，不做热目标分类。
- `Refresh`：刷新摄像头列表。
- `Source` 默认显示可打开的 `UVC x`；若自动探测不到，会提供 `UVC 0..12 (manual)` 兜底项。
- `Start/Stop`：开始或停止采集。
- `Screenshot`：保存当前瞄具屏幕画面。

虚拟物理按键：

- 左键 `MENU / NEXT` 短按：打开菜单或切换菜单项。
- 右键 `OK / ADJUST` 短按：调整当前菜单项；菜单关闭时循环倍率。
- 左键长按：退出菜单。
- 右键长按：截图。

键盘快捷键：

- `A` 或 `Left`：左键短按。
- `D` 或 `Right`：右键短按。
- `Shift + A`：左键长按。
- `Shift + D`：右键长按。
- `Space`：冻结/恢复画面。
- `S`：截图。

菜单项：

1. `ENH`：增强等级 1-5。
2. `ZOOM`：倍率 1x/2x/4x/8x。
3. `WHOT/BHOT`：白热/黑热。
4. `OUTLINE`：纯轮廓模式开关（`ON` 时仅保留高频边缘并清零低频背景）。
5. `FREEZE`：冻结画面。

## 技术路线

显示增强分为三条关键链路：

```text
USB/Mock raw14
→ (UVC: Y16优先, 8bit回退升维) / (Mock: 原生14bit热场)
→ WHOT/BHOT链路:
   percentile AGC
   → Gaussian base/detail decomposition
   → detail gain
   → white-hot/black-hot mapping
→ Thermal Tiny1-C OUTLINE链路:
   bad-pixel suppression
   → temporal EMA
   → light spatial denoise
   → thermal target gate
   → Sobel5x5 + Scharr 混合梯度
   → gradient magnitude + direction
   → NMS(1像素压缩)
   → hysteresis edge linking
   → edge density cap
   → 1-2像素智能补边
   → strength-weighted outline
→ Visible Demo OUTLINE链路:
   visible grayscale
   → bilateral/Gaussian denoise
   → mild contrast stretch
   → Sobel/Scharr + Canny-like linking
   → small-component cleanup
   → edge density cap
→ candidate mask
→ contour extraction
→ rule-based person/object candidate classification
→ scope HUD rendering
```

其中：

- `OUTLINE=ON` 时底图改为纯边缘图，不叠加人物/物体分类描边。
- `Visible Demo` 下不运行热目标分类和热点标记，HUD 会显示 `VISIBLE DEMO`。
- 低频分量在 outline 输出中直接置零，仅保留高频边缘。
- 文档 `docs/holosun_outline_reverse_engineering.md` 记录了参数、依据和偏差说明。

## 工程结构

```text
thermal_sim_v3/
├── main.py
├── requirements.txt
├── README.md
├── core/
│   ├── camera_capture.py       UVC/Mock 采集
│   ├── scope_enhancement.py    AGC/DDE-like 热图增强
│   ├── outline_processing.py   Thermal/Visible 双 Profile Outline 链路
│   ├── scope_renderer.py       瞄具屏幕 HUD 渲染
│   ├── hotspot_detector.py     热点和 candidate mask
│   ├── contour_overlay.py      外轮廓和几何特征
│   ├── target_classifier.py    规则辅助人物/物品候选
│   ├── thermal_processing.py   灰度输入工具函数
│   └── frame_recorder.py       截图/录像保存
├── ui/
│   ├── main_window.py          PySide6 主界面和两按键状态机
│   └── video_widget.py         预渲染画面显示
└── output/
    ├── screenshots/
    └── recordings/
```

## 验证

```powershell
python -m compileall -q main.py core ui
```

验收建议：

1. `python main.py` 能启动。
2. Mock Person 场景显示横向瞄具画面。
3. 中央有简洁十字分划。
4. `OUTLINE=ON` 时画面低频背景接近全黑，主轮廓连续，杂边不过量。
5. 两个虚拟按键能切换菜单、倍率、增强等级和截图。
6. 截图保存的是最终瞄具屏幕画面。
7. 切换到 `Visible Demo` 时，HUD/状态栏明确显示可见光演示模式。
