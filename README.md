# 多时间基音视频确定性渲染引擎

面向离线成片生产的 Python 3.12 渲染引擎。项目解析、有理时间映射、分段
计划、缓存、恢复与验证全部自行实现；FFmpeg/ffprobe 仅承担编解码与固定
滤镜。所有渲染为 CPU 软件编解码，不使用 Docker、GPU 或云服务。

## 快速开始

```powershell
python -m timeline_engine.cli generate workspace/sources
python -m timeline_engine.cli render workspace/sources/demo_project.json workspace/job --segment-frames 45
python -m timeline_engine.cli validate workspace/sources/demo_project.json workspace/job workspace/job/published/master_lossless_1.mkv
python -m unittest discover -s tests -v
```

## 交付物

| 路径 | 说明 |
| --- | --- |
| `timeline_engine/` | 核心库（有理时间、模型、探测解码、规划、分段、渲染、验证） |
| `timeline_engine/cli.py` | 批量渲染/生成/验证入口 |
| `timeline_engine/generator.py` | 确定性素材生成器 |
| `timeline_engine/validator.py` | 独立验证器 |
| `tests/` | 有理数学单元测试 + 端到端集成测试 |
| `docs/project_format.md` | 工程 JSON 格式与拒绝条件 |
| `docs/architecture.md` | 统一时间系统、选帧、缓存与恢复设计 |
| `docs/windows_operations.md` | Windows 命令、中断恢复、AAC 说明 |

## 关键确定性保证

* 权威时间轴只用整数分子/正整数分母，拒绝浮点时间值。
* 选帧基于解码显示 PTS 二分查找，CFR/VFR/B 帧/非零起点统一处理。
* 帧中心、样本中心、边界、淡变、叠化共用同一全局时间坐标。
* 分段闭开区间无重无漏；整段与任意分段恢复结果像素全等、音频误差
  `<= 1e-6`。
* 缓存键覆盖内容哈希、时间映射、邻接转场/包络、工作格式与 FFmpeg 版本。
* 半成品、完整段、已发布成片三态分离；崩溃后不误用无完成标记的产物。
* 双锚点仿射时钟精确求解，不从内容猜测偏差。

## 实测工具链

* Python 3.12.8
* FFmpeg/ffprobe 9.0.2 essentials build（CPU）
* numpy 2.5.4

详见 `docs/windows_operations.md` 与 `docs/architecture.md`。
