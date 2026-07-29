# MaixCAM2 拼图视觉模块

本目录 `maixcam2_puzzle_vision/` 是独立于既有工程的 E 题视觉实现。它只做相机标定、碎片识别、矩形拼接求解和抓取/放置/旋转位姿输出，不控制龙门架、电磁铁、MSPM0 或串口。

## 能力边界

- 支持源区内相互分离的 1 至 4 片碎片，每片 3 至 5 边。
- 支持由配置选择纵向 `210 x 297 mm` 或横向 `297 x 210 mm` 的 A4 坐标系。
- 使用背景颜色距离分割、多边形轮廓、等长切割边匹配、刚体布局搜索和矩形评分。
- 目标尺寸限制为短边 `50-90 mm`、长边 `90-120 mm`。
- 几何方案接近时，将各候选布局的源碎片重投影并比较接触带的花纹连续性；无法唯一判断时输出 `AMBIGUOUS`，绝不输出移动记录。
- `OK` 的 JSON 命令采用毫米坐标，正角度为 A4 平面逆时针。

## 桌面验证

安装工作区开发依赖后运行：

```powershell
python -m pytest -q tests_maixcam2_puzzle_vision
python -m maixcam2_puzzle_vision.cli --image tests_maixcam2_puzzle_vision/generated/four_piece.png --config config/maixcam2_puzzle_vision.default.json --output tmp/four_piece_overlay.png
```

该示例图片已是 `420 x 594` 的矫正板面图，所以默认的恒等单应矩阵可用于离线演示。真实相机的 `1280 x 720` 原始帧不能使用默认配置。

## MaixCAM2 标定和部署

1. 当前部署使用横向 A4：确保完整纸面入镜，四块碎片放在左半区，右半区保持清空作为拼接区。标定后不得改变焦距、分辨率、曝光、白平衡或相机与 A4 的相对位置。
2. 部署包内的 `config/maixcam2_puzzle_vision.json` 是横向配置，坐标左上角为 `(0, 0)`，右下角为 `(297, 210)`；左侧为源区、右侧为目标区。仓库中的单应矩阵只对应当前示例相机位置，换用设备、相机高度或 A4 纸位置后必须重新标定。

在电脑上重新标定时，先从 MaixCAM2 下载新的 `calibration.jpg`，然后运行：

```powershell
python calibrate.py --image calibration.jpg --config config/maixcam2_puzzle_vision.json
```

在弹出的窗口内依次单击 A4 左上、右上、右下、左下四个角，按 Enter 保存；按 `R` 重选，按 Esc 取消。该工具直接写入像素到毫米的矩阵，无需固定到龙门架高度。
3. 将本部署文件夹的全部内容复制到 `/root/app1/`，并用刚标定的 `config/maixcam2_puzzle_vision.json` 覆盖板端同名文件。
4. 在 MaixCAM2 上执行：

```sh
cd /root/app1
python3 main.py
```

短按设备按键采集一帧并只求解一次，串口打印一行 JSON，屏幕保留带红色抓取点、洋红色黑边目标点和旋转箭头的结果。按键时长不小于 `800 ms` 时仅保存 `/root/app1/calibration.jpg`，不会触发求解。

## 状态和动作边界

未来龙门架只能消费 `status: "OK"` 的 `commands`。其他状态的 `commands` 始终是空数组：

- `NO_BOARD`: 缺少、错误或不匹配当前分辨率的标定。
- `SEGMENTATION_FAILED`: 上半区未检测到有效碎片。
- `INVALID_PIECE_COUNT`: 碎片数量不在 1 至 4 内，或碎片接触/触及 ROI 边界。
- `NO_RECTANGLE_SOLUTION`: 不存在合法矩形，或受限搜索未得到解。
- `AMBIGUOUS`: 多个几何或花纹方案无法可靠区分。
- `LOW_CONFIDENCE`: 候选矩形的填充率、接缝残差或得分间隔不符合阈值。

## 现场验收

在尚未接入电磁铁前，用 2、3、4 片纯色卡纸和白底 PCB 图案卡分别验证：

1. 每个红色抓取点在其源碎片内部。
2. 每个洋红色目标点在 A4 右半区，且组合后的矩形边长合法。
3. `OK` 结果的目标多边形无重叠，分界处相邻顶点距离满足题目 `2 cm` 要求。
4. 随机去掉一片、故意让两片相接或遮挡镜头时，输出必须为非 `OK` 且命令为空。

只有以上检查完成后，才应定义龙门架坐标零点、电磁铁启停、到位确认和急停协议。
