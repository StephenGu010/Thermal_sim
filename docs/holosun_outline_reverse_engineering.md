# HOLOSUN 风格 Outline 模式逆向拆解（thermal_sim_v3）

## 1. 目标与边界

本文档描述 `thermal_sim_v3` 的 Outline 仿真实现，目标是**行为级贴近**实机观感，而不是复现厂商未公开的专有源码。

- 目标：`OUTLINE=ON` 时低频背景清零，仅保留高频边缘，边缘尽量保持单像素连续。
- 非目标：声明“完全等价 HOLOSUN 内部算法”。

## 2. 公开资料依据（调研结果）

以下信息是公开可验证的“硬事实”：

- HOLOSUN DRS-TH 对外提供 White Hot / Black Hot / Outline / Highlight 模式，并使用 256×192 传感器。  
  来源：<https://www.holosun.com/products/drs-th>
- Canny 流程包含梯度计算、非极大值抑制（NMS）和滞后阈值连通。  
  来源：<https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html>
- OpenCV 支持 Scharr 与 Sobel 导数算子，可用于梯度计算。  
  来源：<https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html>
- 14/16-bit 热像原始数据通常需要映射到 8-bit 才适合显示；预 AGC 数据常用于后处理。  
  来源：<https://flir.custhelp.com/app/answers/detail/a_id/5986/~/flir-oem---16-bit-or-14-bit-pre-agc-data-display-and-conversion->
- UVC 场景可通过 `Y16` + `CAP_PROP_CONVERT_RGB=0` 获取 16-bit 热流（设备支持时）。  
  来源：<https://flir.custhelp.com/app/answers/detail/a_id/3387/~/flir-oem---boson-video-and-image-capture-using-opencv-16-bit-y16>
- Canny 参考设计中的 edge linking 是“强边缘带弱边缘”的连通策略。  
  来源：<https://www.intel.com/content/www/us/en/docs/programmable/683433/current/edge-linking.html>

## 3. 逆向假设（Inference）

基于以上公开资料与实机常见观感，本项目采用以下推断：

- Outline 模式核心是“高频边缘可视化”，而不是温度灰度底图显示。
- 先做梯度、再做 NMS、再做滞后连通是合理主链路。
- 小断裂修复（1-2像素）能更接近“连续轮廓”观感。

> 以上为工程推断，不代表厂商确认。

## 4. v3 具体实现

### 4.1 输入统一为 14-bit

- 内部标准数据：`raw14` (`uint16`, 0..16383)。
- Mock：直接生成 14-bit 热场（背景热梯度、目标热源、固定纹理噪声、时域扰动）。
- UVC：优先尝试 Y16；若不可用则使用 8-bit 灰度并线性升维到 14-bit。

### 4.2 Outline 核心管线

实现文件：`core/outline_processing.py`

```text
raw14
→ Gaussian denoise (5x5)
→ Sobel(5x5) + Scharr 混合梯度
→ 幅值/方向计算
→ NMS (0/45/90/135)
→ 双阈值滞后连通
→ 基于方向的 1-2 像素补边
→ 输出纯边缘图（低频=0，高频=255）
```

### 4.3 渲染语义

- `OUTLINE=OFF`：使用 WHOT/BHOT 主链路。
- `OUTLINE=ON`：底图切到纯边缘图，不叠加人物/物体分类描边。
- HUD 保留（倍率、菜单、准星、冻结状态等）。
- 左下角输入源字样（`PC SIM`/`USB CAM`）移除。

## 5. 参数说明（首版）

首版关键参数（`OutlineConfig`）：

- `gaussian_ksize=5`
- `sobel_weight=0.58`, `scharr_weight=0.42`
- `high_percentile=92.0`, `low_ratio=0.44`
- `bridge_max_gap=2`, `bridge_strength_ratio=0.55`
- `glow_gain=0.22`, `glow_sigma=0.9`

增强等级 `ENH 1..5` 会联动阈值与 glow 强度，达到“弱边缘可见性”和“噪声抑制”平衡。

## 6. 已知偏差与后续建议

- 由于缺少官方算法与实机标定图，本实现属于“视觉贴近”而非“参数同构”。
- 若后续获得实机截图，建议做二次拟合：
  1. 边缘密度（每帧边缘像素占比）
  2. 边缘连续度（断裂长度分布）
  3. 背景残留亮度（应接近 0）
  4. 目标边缘主观对齐（人体轮廓、热物体轮廓）

