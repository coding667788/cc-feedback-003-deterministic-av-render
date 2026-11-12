# 架构与统一时间系统

## 分层

```text
rational.py      精确有理运算、ceil/round-half-up、整数 tick 速率
model.py         JSON schema、dataclass、渲染前结构校验
ffmpeg_tools.py  版本捕获、ffprobe、子进程、内容哈希
probe_extra.py   显示序帧/解码序包探测（识别 B 帧、VFR）
source.py        源解码到统一工作格式 + 内容寻址缓存
generator.py     确定性素材、帧标记、音频脉冲、VFR/B 帧生成
timeline.py      量化规划、仿射时钟、选帧、包络、混音、分段边界
segment.py       分段缓存键、FFV1/f32 编码、有界并行、原子提交
render.py        作业状态、崩溃恢复、无损拼接、MP4 发布
validator.py     独立解码验证、标记/脉冲核对、跨渲染比对
project_factory.py  示例工程与 32 分钟锚点工程
cli.py           generate / render / validate / compare
```

## 一个全局时间坐标

视频帧中心与音频样本中心都从同一全局秒坐标求得：

```text
frame n  -> t = (n + 1/2) * fps_den / fps_num
sample k -> t = (k + 1/2) / sample_rate
```

片段边界、淡入淡出、叠化权重、背景音包络全部在该坐标上求值，不对每段
独立取整后拼接。权威边界用 `Fraction`；向量化 numpy 仅用于样本插值，
其系数由精确有理值导出，误差远低于声明的 `1e-6`。

## 选帧（CFR/VFR 统一）

1. 源解码阶段用 `ffprobe -show_frames` 取得显示序 PTS 列表，并校验严格
   递增、无 `N/A`；`-show_packets` 比较 PTS/DTS 判断是否存在 B 帧。
2. 时间线时刻 `t` 映射到源视频时刻
   `s = src_in + (t - timeline_start) * speed`。
3. 在显示 PTS 数组上二分查找 `PTS <= s` 的最近帧。不使用平均 fps，不
   假设解码顺序等于显示顺序，因此 VFR、B 帧、非零起点走同一条路径。

## 音频重采样与仿射时钟

无锚点时音视频视为容器共时钟；有两个锚点时用唯一仿射映射
`sample = slope * video_time + intercept`。输出样本中心先映射到源视频
时刻，再反解为源样本位置（浮点），对原生采样率的 float32 音轨做线性
插值，边界样本由全局索引决定，不依赖 FFmpeg 滤镜的跨段历史。

## 滤镜前后文为何不会丢

引擎不把 atempo/setpts/fps 等有状态滤镜跨段使用。每个输出帧/样本都从
全局时间和完整源 PTS/样本数组重新计算，因此分段位置可以落在 GOP 内、
变速片段内或叠化中途，而不需要“补前缀帧再裁掉”这类易漂移的技巧。
FFmpeg 只负责：源编解码、固定 bicubic 缩放/黑边、YUV→RGB 全范围转换、
FFV1/PCM 与最终 H.264/AAC 编码。

## 分段所有权

分段边界取相邻帧中心的精确中点：

```text
boundary = (center(frame_end-1) + center(frame_end)) / 2
sample_end = ceil(boundary * R - 1/2)
```

相邻段满足 `left.end == right.start`，所有段帧数之和等于 `N`、样本数
之和等于 `round_half_up(Dq*R)`，没有重复负责也没有空洞。

## 缓存键覆盖的依赖

段键（SHA-256）包含：

* 段拥有的全局帧/样本闭开区间；
* 影响该段的每个片段/背景项的源内容 SHA-256、源区间、时间线起点、
  速度、增益、淡入淡出、双锚点、源首 PTS/采样率；
* 邻接转场的两端、overlap、kind；
* 固定工作格式与输出规格；
* FFmpeg、ffprobe 完整版本横幅与引擎版本。

同名文件被替换会改变内容哈希；只改一处转场只会使接触该叠化的段失效；
只改锚点只会使携带该锚点的项所在段失效；无关段继续命中。

## 三态产物与崩溃安全

* `.tmp` / 无 `done.json`：半成品，恢复时不承认。
* 段目录内 `video.mkv` + `audio.wav` + 原子 rename 的 `done.json`：
  完整中间片段，`done.json` 记录双产物哈希与全局索引。
* `published/` 下版本号递增的母版/MP4/发布元数据：已发布成片，重渲染
  永不覆盖，只有全部段与总帧数/样本数校验通过后才发布。

状态文件 `job_state.json` 通过临时文件 + 原子替换写入，进程强杀后
重启仍能区分 failed/running 与 published。

## 验证独立性

验证器不读取渲染器内部计划日志，而是：

1. 自行 FFmpeg 解码无损母版；
2. 从像素位块读回源 ID/帧序号并校验奇偶与签名；
3. 用工程 JSON 与源 PTS 独立重算期望帧；
4. 在音频脉冲声道定位峰值，与有理公式算出的期望样本位置对照；
5. 对整段与分段/恢复产物做像素全等与 `1e-6` 音频误差比较。
