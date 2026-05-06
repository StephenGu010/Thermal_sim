# thermal_sim 上位机代码讲解与后续增强提示词

本文档梳理 `F:\final_design\thermal_sim` 当前 PySide6 热成像上位机的实现效果、核心代码原理，以及下一步实现“热目标描边”和“区分人像/物品”的开发提示词。

## 1. 当前工程定位

当前 `thermal_sim` 是一个 PC 端热成像预览工具，主要用于把 Tiny1-C 通过 USB-UVC 当作普通摄像头接入电脑后，在电脑端完成热图预览、伪彩色增强、热点检测、候选热区框选、截图和录像。

当前阶段不依赖 FPGA、ESP32-S3 和 AMOLED，也没有接入真实神经网络模型。它更适合作为“电脑端热成像效果仿真/预览工具”，用于先把毕业设计中的最终显示风格做出来。

## 2. 当前已经实现的功能

### 2.1 USB 摄像头与 Mock 输入

相关文件：

- `main.py`
- `core/camera_capture.py`
- `ui/main_window.py`

当前支持两种输入源：

1. `UVC` 摄像头模式  
   使用 `cv2.VideoCapture(index, cv2.CAP_DSHOW)` 打开 Windows 下的 USB 摄像头。如果 Tiny1-C 被电脑识别为标准摄像头，可以通过 UVC index 读取画面。

2. `Mock` 模拟模式  
   在没有热成像摄像头时，程序会生成一帧 256×192 的模拟热图，包含渐变背景、一个较大的移动热源、一个小热点和随机噪声。

实现原理：

- `CaptureThread` 继承 `QThread`，采集线程与 UI 线程分离，避免摄像头读取阻塞主界面。
- `frame_ready` 信号把灰度帧、原始 BGR 帧、frame_id 和 FPS 发送给主窗口。
- Mock 模式用高斯函数生成热源亮斑，用噪声模拟热成像传感器纹理。

### 2.2 灰度化与图像增强

相关文件：

- `core/thermal_processing.py`
- `ui/main_window.py`

当前处理链路为：

```text
摄像头/Mock frame
→ to_gray
→ min-max 自动归一化
→ 可选高斯降噪
→ 可选 CLAHE 局部对比度增强
→ 可选锐化
→ 输出 processed_gray
```

已经实现的增强功能：

- 自动归一化：把当前帧的灰度范围拉伸到 0-255。
- CLAHE：增强局部对比度，适合热图中温差不明显的场景。
- 高斯降噪：降低随机噪声带来的候选区域误检。
- Unsharp 锐化：增强边缘和局部变化。

实现原理：

- `to_gray()` 把任意输入图像转成 `uint8` 灰度图。
- `normalize_minmax()` 计算当前帧 `gray_min` 和 `gray_max`，并进行线性拉伸。
- `apply_clahe()` 使用 OpenCV 的 `cv2.createCLAHE()`。
- `denoise()` 使用 `cv2.GaussianBlur()`。
- `sharpen()` 使用原图与模糊图加权相减实现锐化。

注意：

当前显示的是相对热强度，不是真实摄氏温度。没有温度标定数据时，不应在论文或界面中写成“36.5℃人体温度识别”。

### 2.3 伪彩色显示

相关文件：

- `core/color_map.py`
- `ui/main_window.py`

当前支持的色带：

- Gray
- Iron
- Inferno
- Jet
- Turbo
- Hot
- Bone

实现原理：

- 灰度图仍然作为算法处理基础。
- 色带只影响最终显示效果。
- OpenCV 内置色带通过 `cv2.applyColorMap()` 实现。
- `Iron` 色带由代码自定义 256 级 LUT，从黑色、紫色、红色、橙色、黄色过渡到白色，比较接近常见热成像观察仪的视觉风格。

当前截图 `output/screenshots/thermal_20260505_171537.png` 显示的是铁红/高亮风格的热源亮斑，说明伪彩色链路已经生效。

### 2.4 热点检测

相关文件：

- `core/hotspot_detector.py`
- `ui/video_widget.py`

当前热点检测逻辑：

```text
processed_gray
→ cv2.minMaxLoc
→ 取最大灰度点作为 hotspot
→ 在 UI 中用青色十字准星标记
```

输出字段：

- `hotspot_x`
- `hotspot_y`
- `hotspot_value`

实现原理：

- 热成像图中灰度值越高，代表相对热强度越高。
- 当前没有温度标定，因此热点表示“当前画面中最亮的相对热源点”，不是绝对温度最高的摄氏温度点。

### 2.5 候选热区 mask 与矩形框

相关文件：

- `core/hotspot_detector.py`
- `ui/video_widget.py`

当前候选区域检测流程：

```text
processed_gray
→ 百分位阈值，例如 P95
→ 生成二值 mask
→ 形态学开运算
→ 形态学闭运算
→ 连通域分析
→ 面积过滤
→ 输出候选区域 bbox、centroid、area
```

实现原理：

- 百分位阈值用于筛选当前帧中最亮的一部分区域。
- 开运算可以去掉孤立噪声点。
- 闭运算可以填补候选区域内部的小孔洞。
- `cv2.connectedComponentsWithStats()` 用于提取每个候选热区的位置、面积和质心。
- UI 中对每个候选区域绘制黄色矩形框，并可叠加半透明红色 mask。

当前效果：

- 已经能框出亮度较高的热源区域。
- 已经能显示候选热区数量和候选区域质心。
- 目前是“候选热区框选”，不是严格的人体检测。

### 2.6 UI 叠加层

相关文件：

- `ui/video_widget.py`

当前 UI 叠加内容：

- 热点十字准星。
- 候选区域黄色矩形框。
- 半透明 mask。
- `PC Preview` 水印。
- 当前色带名称。
- 当前 FPS。

实现原理：

- `VideoWidget` 先把 BGR 热图转换为 `QImage`。
- 然后用 `QPainter` 在画面上叠加 mask、矩形框、准星和文字。
- 支持原始比例和 AMOLED 294×126 预览比例。

重要问题：

当前 UI 中能看到叠加层，但截图和录像保存的是 `_last_bgr`，也就是伪彩色图像本身，不包含 `QPainter` 绘制的候选框、热点准星和 mask。  
如果后续要把效果图放进论文，建议优先修改截图/录像逻辑，让保存文件也包含最终叠加效果。

## 3. 当前还没有实现的功能

### 3.1 没有真正的目标轮廓描边

当前已经实现的是：

- 候选热区 mask。
- 候选区域矩形框。
- 半透明 mask 叠加。

当前还没有实现的是：

- 沿目标外轮廓的连续描边。
- 不同目标类型使用不同颜色描边。
- 目标边缘与伪彩热图融合后的视觉增强。

如果要实现“描边效果”，应在 mask 上使用 `cv2.findContours()` 提取轮廓，然后用 `cv2.drawContours()` 或 `QPainterPath` 绘制轮廓线。

### 3.2 没有区分人像和物品

当前候选区域只根据热强度筛选，逻辑是“亮的区域可能是热目标”。它不能判断这个热目标是人、杯子、电脑、灯泡还是其他发热物体。

当前没有实现：

- 人体轮廓特征判断。
- 人像/物品分类标签。
- 人像置信度。
- 基于机器学习或深度学习的分类模型。

如果短期内不训练模型，可以先做低风险规则分类：

- 目标面积是否足够大。
- 目标宽高比是否接近站立/坐姿人体。
- 目标是否具有“头部 + 躯干”的上下结构。
- 热区是否连续。
- 目标中心是否稳定。
- 是否排除过小、过圆、过亮的点状热源。

这种方式只能写成“规则辅助的人体候选判别”，不能写成真实 AI 人体识别准确率。

## 4. 建议下一步优先级

### 优先级 1：让截图包含叠加效果

当前论文最需要的是最终效果图。建议先把保存逻辑改成：

```text
processed_gray
→ pseudo color
→ draw mask
→ draw contours
→ draw bbox
→ draw hotspot crosshair
→ draw status text
→ save final preview image
```

这样保存出来的图片才能直接展示“热成像增强 + 热点 + 候选框 + 描边”的最终效果。

### 优先级 2：增加轮廓描边

新增模块建议：

```text
core/contour_overlay.py
```

建议功能：

- 从 candidate mask 提取轮廓。
- 过滤太小轮廓。
- 计算轮廓面积、周长、圆度、宽高比。
- 对目标边缘绘制高亮描边。
- 支持描边颜色、粗细和透明度可调。

### 优先级 3：增加人像/物品规则分类

新增模块建议：

```text
core/target_classifier.py
```

建议输出：

```python
class TargetType:
    PERSON_CANDIDATE = "person_candidate"
    OBJECT_CANDIDATE = "object_candidate"
    HOTSPOT_NOISE = "hotspot_noise"
    UNKNOWN = "unknown"
```

可以先实现规则评分：

```text
person_score =
  面积得分
  + 宽高比得分
  + 上下热区结构得分
  + 连续性得分
  - 点状热源惩罚
```

分类结果建议：

- `person_candidate`：绿色或青色描边。
- `object_candidate`：黄色描边。
- `hotspot_noise`：红色小框或忽略。
- `unknown`：灰色描边。

### 优先级 4：改进 Mock 模式

当前 Mock 模式是高斯亮斑，形状不像人。建议增加：

- 人形热源：椭圆头部 + 躯干 + 双臂/腿部弱热区。
- 物品热源：杯子、矩形热源、点状热源。
- 多目标场景：人 + 热杯子 + 背景噪声。

这样才能调试“区分人像和物品”的视觉效果。

## 5. 可直接丢给 AI 的下一步开发提示词

```text
你现在是我的 PySide6 热成像上位机开发助手。请在现有工程基础上继续开发，不要重建项目。

现有工程路径：
F:\final_design\thermal_sim

请先阅读以下文件：
- F:\final_design\thermal_sim\main.py
- F:\final_design\thermal_sim\core\camera_capture.py
- F:\final_design\thermal_sim\core\thermal_processing.py
- F:\final_design\thermal_sim\core\hotspot_detector.py
- F:\final_design\thermal_sim\core\color_map.py
- F:\final_design\thermal_sim\core\frame_recorder.py
- F:\final_design\thermal_sim\ui\main_window.py
- F:\final_design\thermal_sim\ui\video_widget.py
- F:\final_design\thermal_sim\README.md

当前工程已经实现：
- USB-UVC 摄像头读取；
- Mock 热图输入；
- 灰度归一化；
- CLAHE；
- 降噪；
- 锐化；
- 伪彩色显示；
- 热点十字准星；
- 候选热区 mask；
- 候选区域矩形框；
- UI 叠加显示；
- 截图和录像。

现在我要进一步增强最终显示效果，重点实现两个功能：

1. 热目标轮廓描边；
2. 粗略区分人像候选和物品候选。

请注意：
- 不要声称这是训练完成的 AI 人体识别；
- 不要编造准确率、召回率、F1；
- 不要显示真实摄氏温度；
- 当前只做 PC 端视觉预览和规则辅助分类；
- 分类名称建议写成“人体候选 person_candidate”，不要写成“已确认人体”。

一、请新增轮廓描边功能

新增文件：
F:\final_design\thermal_sim\core\contour_overlay.py

要求实现：

1. 输入 candidate mask 和 candidate regions；
2. 使用 OpenCV `cv2.findContours()` 提取外轮廓；
3. 对轮廓进行面积过滤；
4. 计算每个轮廓的：
   - bbox；
   - area；
   - perimeter；
   - aspect_ratio；
   - extent；
   - circularity；
   - centroid；
5. 输出可供 UI 绘制的 contour 数据；
6. 支持描边开关；
7. 支持描边粗细设置；
8. 支持不同目标类型使用不同颜色：
   - 人像候选：青色或绿色；
   - 物品候选：黄色；
   - 点状热源/噪声：红色或忽略；
   - 未知：白色或灰色。

二、请新增人像/物品规则分类功能

新增文件：
F:\final_design\thermal_sim\core\target_classifier.py

请实现一个轻量规则分类器，不要接真实 AI 模型。

输入：
- processed_gray；
- candidate mask；
- contour features；
- candidate bbox；
- frame size。

输出：
- target_type；
- confidence；
- reason；
- draw_color。

分类规则建议：

1. 人像候选 person_candidate：
   - 面积不能太小；
   - bbox 高度较大；
   - 宽高比接近人体站立或坐姿；
   - 热区不是单纯圆点；
   - 上方有较亮的小区域可以视为头部候选；
   - 下方有较大的连续区域可以视为躯干候选。

2. 物品候选 object_candidate：
   - 宽高比过宽或过圆；
   - 面积较小；
   - 热区结构简单；
   - 类似杯子、灯、电子设备、热源块。

3. 点状热源 hotspot_noise：
   - 面积很小；
   - circularity 较高；
   - bbox 很小；
   - 亮度高但区域不连续。

4. 未知 unknown：
   - 不满足以上稳定规则。

请把规则写清楚，并在代码注释中说明这是“规则辅助分类”，不是神经网络识别结果。

三、请修改 UI 显示

修改：
F:\final_design\thermal_sim\ui\video_widget.py
F:\final_design\thermal_sim\ui\main_window.py

要求：

1. 在左侧控制面板增加：
   - “目标描边”开关；
   - “分类标签”开关；
   - 描边粗细 slider；
   - 人像规则阈值 slider。

2. 中间画面叠加：
   - 对候选目标绘制轮廓线；
   - 在轮廓附近显示标签，例如：
     `PERSON CANDIDATE 0.72`
     `HOT OBJECT 0.58`
     `SMALL HOTSPOT`
   - 不要用“Human Confirmed”这种过度确定的词。

3. 右侧数据面板增加：
   - target_count；
   - person_candidate_count；
   - object_candidate_count；
   - main_target_type；
   - main_target_score；
   - main_target_reason。

四、请修复截图和录像逻辑

当前截图保存的是伪彩色 BGR 图，不包含 UI 里的 QPainter 叠加层。请修复为保存最终叠加效果。

可选实现方式：

方案 A：
在 OpenCV 层新增一个 `render_preview_frame()` 函数，直接把 mask、contour、bbox、hotspot、文字画到 BGR 图上，然后：
- UI 显示这个 rendered_bgr；
- 截图保存 rendered_bgr；
- 录像写入 rendered_bgr。

方案 B：
从 `VideoWidget` grab 当前 QWidget 画面并保存。

优先推荐方案 A，因为它能保证截图、录像和 UI 处理逻辑一致，也方便后续生成论文效果图。

五、请改进 Mock 模式

修改：
F:\final_design\thermal_sim\core\camera_capture.py

新增 Mock 场景选择：

1. blob_basic：
   当前高斯热源模式。

2. person_scene：
   生成类人形热源：
   - 椭圆头部；
   - 躯干；
   - 两侧手臂弱热区；
   - 腿部弱热区；
   - 背景噪声。

3. object_scene：
   生成热杯子/热块/电子设备等物品热源。

4. mixed_scene：
   同时包含一个人体候选和一个热物品候选，用来测试分类显示效果。

六、视觉效果目标

最终画面应该具有以下效果：

- 热图主体使用 Iron 或 Turbo 伪彩色；
- 人像候选区域边缘有青绿色连续描边；
- 热物品区域用黄色描边；
- 点状高亮噪声用红色小标记或直接过滤；
- 最热点仍然有十字准星；
- 半透明 mask 可以开关；
- 候选框可以保留，但视觉重点改成轮廓描边；
- 标签文字小而清晰，不遮挡主体画面；
- 截图保存出来必须包含伪彩色、轮廓描边、目标标签、热点准星和状态文字。

七、验收标准

请完成后运行或提供运行说明，至少保证：

1. `python main.py` 可以启动；
2. Mock 模式可以看到人形热源；
3. 人形热源被标记为 `PERSON CANDIDATE`；
4. 物品热源被标记为 `HOT OBJECT` 或 `OBJECT CANDIDATE`；
5. 热源边缘有连续描边，不只是矩形框；
6. 截图文件中能看到描边和标签；
7. 没有摄像头时仍可使用 Mock 模式；
8. README 更新，说明该分类是规则辅助，不是真实 AI 准确识别。

请直接修改工程代码，并把新增和修改的文件列出来。
```

## 6. 毕设表述建议

当前可以这样描述：

> PC 端上位机实现了基于 USB-UVC 热成像输入的实时预览功能。软件对输入灰度热图进行归一化、局部对比度增强、降噪、伪彩色映射和热点检测，并通过百分位阈值、形态学处理和连通域分析生成候选热区。当前候选目标以矩形框和半透明 mask 方式叠加显示，用于验证热成像增强和候选区域提取流程。

后续实现描边和分类后，可以升级为：

> 在候选热区基础上，系统进一步提取目标轮廓，并根据面积、宽高比、热区连续性和上下结构等特征进行规则辅助分类，区分人体候选、热物品候选和点状热源。该分类结果用于改善显示叠加效果，不作为真实神经网络识别精度结果。

