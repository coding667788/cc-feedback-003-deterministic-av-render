# 实测验证结果（Windows 原生）

## 工具链

| 组件 | 实测版本 |
| --- | --- |
| Python | 3.12.8 |
| FFmpeg | 9.0.2-essentials_build（CPU 软件编解码） |
| ffprobe | 9.0.2-essentials_build |
| numpy | 2.5.4 |

## 自动化测试

`python -m unittest discover -s tests`：**16/16 通过**。

| 测试 | 覆盖要求 |
| --- | --- |
| `test_rational_plan` | 拒绝浮点时间、有理转换、round-half-up、N=ceil 量化、分段无重无漏 |
| `test_clock_mapping` | 双锚点仿射精确过点、默认时钟、逆序锚点拒绝、VFR 按 PTS 选帧 |
| `test_integration.test_01` | 整段 + 4 种分段（45/23/17/7 帧）+ 两种并行度（1/2），像素全等、音频 `max|err|<=1e-6` |
| `test_integration.test_02` | 验证器独立解码标记与脉冲，不依赖渲染日志 |
| `test_integration.test_03` | 三重重叠非法工程在渲染前拒绝，无发布产物 |
| `test_integration.test_04` | 段文件已写、done 标记未提交时崩溃；重启不承认半成品，已完成段命中后恢复发布 |
| `test_integration.test_05` | 二次运行全命中；仅改转场长度只失效接触段；同名素材内容变更定向失效 |
| `test_integration.test_06` | 帧标记 32 位源 ID + 32 位帧序号 + 奇偶 + 签名往返 |

## 素材矩阵实测

ffprobe 独立确认：

| 素材 | 实测特征 |
| --- | --- |
| `src_cfr25.mkv` | 25 fps、48 kHz、首 PTS `123/1000` |
| `src_cfr2997.mkv` | 30000/1001 fps、44.1 kHz、首 PTS `77/1000` |
| `src_vfr.mkv` | 真实可变显示间隔，帧间隔交替 0.05s/0.075s |
| `src_bframes.mp4` | 包级 PTS≠DTS，存在 B 帧重排，limited-range YUV |
| `src_videoonly.mkv` | 仅视频流，无音轨 |

## 8 秒综合工程

* 输出 240 帧、384384 样本，规范化时长 `1001/125`。
* 验证器结果 `ok=true`：129 个单片段帧标记全部匹配，叠化帧通过混合重建路径覆盖。
* 空隙帧实测全黑（RGB24=0），空隙样本 `max|x|=0.0`。
* 包含非关键帧入点、1/3s 边界、3/2 与 2/3 倍速、跨段叠化、44.1/48 kHz 混音。

## 32 分钟锚点长时工程

* 工程时长 1920s，输出 **57543 帧 / 92160869 样本**。
* 960 个携带双锚点的片段，仿射斜率 `1600000/33`，两个锚点精确成立。
* 独立验证：57542 个可读标记全部正确，**3840 个音频脉冲全部找到，最大样本偏差 1**。
* 分段对照：1000 帧/并行 2 的母版 vs 333 帧/并行 1 的母版，流式比对
  `video_pixel_mismatches=0`、`audio_max_abs_error=0.0`（容差 1e-6）。
* 结论：跨 32 分钟、173 个分段边界没有累积一帧或一个采样点错位。

## 缓存失效范围（可核查）

段缓存键为 SHA-256，命中记录写入 `render_report.json` 的 `cache_hits` /
`cache_misses`：

* 同工程二次渲染：全部段命中，`cache_misses=0`。
* 仅修改一处叠化 overlap：只有时间上接触该叠化的段未命中，其余段复用。
* 同名文件内容追加导致 SHA-256 变化：引用该媒体的段全部未命中，无关段复用。
* 仅修改音频锚点：锚点进入段键，携带该锚点的段定向失效。

## 中断恢复

崩溃注入点位于 MKV/WAV 已 `replace` 落盘、`done.json` 原子 rename 之前。
实测重启后：

* 无 `done.json` 的段目录不进入命中集合；
* 已有完整标记的段直接命中；
* `job_state.json` 状态为 `failed`，恢复成功后才转为 `published`；
* `published/` 使用递增版本号，上一份母版与 MP4 不被覆盖。

## MP4 / AAC 说明

最终 MP4 为 CPU `libx264 yuv420p` + `aac`。对实测工程解码后与无损母版
做整数偏移相关搜索，最佳偏移为 0、样本数一致（384384）。发布元数据仍
保留 `aac_priming_samples` 字段与说明；时间线正确性始终以 FFV1/PCM 无损
母版为准，MP4 压缩字节不要求一致。

## 已明确的边界

* 仅支持 SDR Rec.709；HDR、HLG、广色域不在范围内。
* 音频变速允许变调，使用引擎自有线性重采样，不做保调。
* 单视频轨同时最多两个活动片段；更多轨道叠加需扩展模型。
* 帧标记要求输出宽度至少 256 像素（32 位块 × 8 像素）。
