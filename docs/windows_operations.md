# Windows 原生操作说明

## 环境

* Windows 原生 Python 3.12（实测 3.12.8）
* 本地 FFmpeg / ffprobe（实测 9.0.2 essentials build，仅 CPU 软编解码）
* 依赖：`numpy`（实测 2.5.4），无 Docker、无 GPU、无云服务、无付费组件

若 FFmpeg 不在 PATH，可用环境变量显式指定：

```powershell
$env:TIMELINE_ENGINE_FFMPEG="D:\path\to\ffmpeg.exe"
$env:TIMELINE_ENGINE_FFPROBE="D:\path\to\ffprobe.exe"
```

## 一键生成素材与示例工程

```powershell
python -m timeline_engine.cli generate workspace/sources --width 256 --height 144
```

生成内容：

| 文件 | 覆盖点 |
| --- | --- |
| `src_cfr25.mkv` | 25 fps CFR、48 kHz、非零起始 PTS |
| `src_cfr2997.mkv` | 30000/1001 fps、44.1 kHz、非零起点 |
| `src_vfr.mkv` | 真实可变 PTS 间隔、48 kHz |
| `src_bframes.mp4` | H.264 B 帧（PTS≠DTS）、limited range YUV |
| `src_videoonly.mkv` | 无音轨素材 |
| `demo_project.json` | 综合剪辑工程 |
| `anchor_project.json` | 32 分钟双锚点长时工程 |
| `invalid_project.json` | 三重重叠非法工程 |

视频帧在固定像素块中写入 32 位源 ID、32 位帧序号、奇偶位与签名；
音频左声道为连续正弦+三角脉冲，右声道仅保留三角脉冲，供验证器独立定位。

## 渲染

```powershell
# 分段大小（帧）与并行度均可调
python -m timeline_engine.cli render `
  workspace/sources/demo_project.json `
  workspace/job_demo `
  --segment-frames 45 `
  --parallelism 1

# 只要无损母版，不出 MP4
python -m timeline_engine.cli render project.json workspace/job --no-mp4
```

任务目录：

```text
job/
  source_cache/        内容寻址的源解码缓存（RGB24 / float32）
  segments/<key>/      每段独立目录：video.mkv(FFV1) + audio.wav(f32) + done.json
  staging/             发布前临时区
  published/           已发布母版、MP4 与发布元数据（版本号递增，不覆盖）
  job_state.json       running/failed/segments_complete/published 状态
  render_report.json   缓存命中/未命中、工具链版本、产物哈希
```

## 独立验证

```powershell
python -m timeline_engine.cli validate `
  workspace/sources/demo_project.json `
  workspace/job_demo `
  workspace/job_demo/published/master_lossless_1.mkv
```

验证器自行调用 FFmpeg 解码产物，读取像素中的帧标记，并按工程有理时间
重新计算期望选帧；音频则在脉冲专用声道上定位三角脉冲，与期望输出样本
位置逐点对照。退出码 0 表示通过，2 表示验证失败。

## 整段 vs 任意分段确定性比对

```powershell
python -m timeline_engine.cli compare `
  project.json `
  job_whole/published/master_lossless_1.mkv `
  job_split/published/master_lossless_1.mkv `
  --tolerance 0.000001
```

* 视频：像素完全一致（`video_pixel_mismatches == 0`）。
* 音频：形状一致，最大绝对误差不超过 `1e-6`。

## 中断恢复演示

崩溃注入由库 API 提供（段文件已写完、`done.json` 尚未原子提交）：

```python
from timeline_engine.render import RenderConfig, RenderJob

config = RenderConfig(
    job_dir="workspace/job_crash",
    segment_frames=30,
    parallelism=1,
    crash_before_done_segments=(2,),
)
RenderJob(project, config).run()   # 抛出注入异常，状态 failed

# 去掉注入后用同一 job_dir 重启：无 done 标记的半成品不会被当缓存，
# 已完成段继续命中，全部段与总时长校验通过后才发布新版本。
RenderConfig(job_dir="workspace/job_crash", segment_frames=30)
RenderJob(project, config).run()
```

## 自动化测试

```powershell
python -m unittest discover -s tests -v
```

测试矩阵包含三种分段大小、两种并行度、整段/分段/恢复对照、缓存命中与
定向失效、同名素材替换、转场长度修改、崩溃恢复、非法工程拒绝。

## AAC 延迟与填充说明

最终 MP4 使用 AAC，编码器会引入 priming 样本和尾部填充。发布元数据
`published/publish_*.json` 记录 `aac_priming_samples`；时间线漂移验证
始终以无损母版为准，MP4 仅作可播放产物，不允许用编码器填充掩盖样本
错位。压缩 MP4 不要求字节一致。
