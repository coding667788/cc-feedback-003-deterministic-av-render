# 工程 JSON 格式（schema 1.0）

所有时间、速度、时间基都用整数分子/正整数分母表示。渲染器拒绝 JSON
浮点数作为权威时间值，避免 `0.1` 这类不可精确表示的二进制浮点进入时间轴。

## 顶层结构

```json
{
  "schema_version": "1.0",
  "output": {
    "width": 256,
    "height": 144,
    "fps": {"num": 30000, "den": 1001},
    "sample_rate": 48000,
    "channels": 2
  },
    "duration": {"num": 8, "den": 1},
  "media": [
    {
      "id": "cfr25",
      "path": "src_cfr25.mkv",
      "pixel_format": "rgb24",
      "color_range": "pc",
      "color_space": "bt709"
    }
  ],
  "clips": [
    {
      "id": "c1",
      "media_id": "cfr25",
      "src_in":  {"num": 403, "den": 1000},
      "src_out": {"num": 2003, "den": 1000},
      "timeline_start": {"num": 0, "den": 1},
      "speed": {"num": 1, "den": 1},
      "track": 0,
      "gain": {"num": 1, "den": 1},
      "fade_in": {"num": 1, "den": 3},
      "fade_out": {"num": 0, "den": 1},
      "anchors": []
    }
  ],
  "transitions": [
    {
      "from_clip": "c1",
      "to_clip": "c2",
      "overlap": {"num": 1, "den": 3},
      "kind": "dissolve"
    }
  ],
  "backgrounds": [
    {
      "id": "bg1",
      "media_id": "cfr25",
      "src_in": {"num": 1, "den": 10},
      "src_out": {"num": 21, "den": 10},
      "timeline_start": {"num": 1, "den": 3},
      "speed": {"num": 1, "den": 1},
      "gain": {"num": 1, "den": 2},
      "fade_in": {"num": 1, "den": 6},
      "fade_out": {"num": 1, "den": 6}
    }
  ]
}
```

## 有理值写法

* 对象：`{"num": 3, "den": 2}`
* 字符串：`"3/2"`、`"7"`
* 整数：`7`
* 浮点：**禁止**，加载时报 `RationalError`

## 时间规则

* 源区间 `[src_in, src_out)` 与时间线区间均为左闭右开。
* 片段时间线时长 = `(src_out - src_in) / speed`，`speed > 0`。
* 输出帧数 `N = ceil(D * fps_num / fps_den)`，规范化时长
  `Dq = N * fps_den / fps_num`。
* 第 `n` 帧在 `(n + 1/2) * fps_den / fps_num` 采样；选择显示 PTS 不大于
  映射后源时刻的最近一帧（VFR 安全，不使用平均 fps 或帧序号除法）。
* 输出样本数 `round_half_up(Dq * sample_rate)`，第 `k` 个样本中心为
  `(k + 1/2) / sample_rate`。
* 空隙与区间外输出黑场（RGB24 全 0）与静音；末帧不足区同样按采样中心规则。

## 双锚点时钟校准

`anchors` 为空时，假定源音视频共用容器时钟：
`sample_position = (video_time - audio_first_pts) * sample_rate`。

提供两个锚点时，二者唯一确定仿射关系
`sample_position = slope * video_time + intercept`：

```text
slope     = (a2.sample - a1.sample) / (a2.video_time - a1.video_time)
intercept = a1.sample - slope * a1.video_time
```

锚点必须严格单调；逆序锚点、零长度区间、非正速度在渲染前拒绝。校准后
裁切、变速、转场都继续使用同一条仿射映射，不从内容猜测时钟偏差。

## 转场与轨道约束

* 同一视频轨同一时刻最多两个片段活动；三重重叠直接拒绝。
* 叠化只允许出现在同轨、真实重叠的两个片段之间，声明 overlap 不得超过
  实际重叠。
* 叠化区权重和恒为 1，权重在全局时间坐标上求值；硬切 `kind: "cut"`
  不产生混合。
* 背景音轨只贡献音频，按全局样本时刻求包络后与主音轨相加，超界硬限幅
  到 `[-1, 1]`。

## 固定工作格式

| 维度 | 固定值 |
| --- | --- |
| 视频中间格式 | FFV1 level 3 / Matroska，`rgb24`，全范围 PC，Rec.709 |
| 缩放 | bicubic，保持宽高比 |
| 补边 | 黑色 letterbox/pillarbox |
| 音频中间格式 | 立体声 `pcm_f32le` WAV，原生源采样率进入，混音时线性重采样到 48 kHz |
| 像素舍入 | `floor(x + 0.5)` |
| 音频溢出 | 硬限幅 `[-1, 1]` |
| 最终 MP4 | CPU `libx264` + `aac`，仅用于播放，不参与无损比对 |

## 拒绝条件（渲染前显式报错，不产生伪成功成片）

* 缺失/`N/A` 的显示 PTS、显示 PTS 非严格递增。
* YUV 源未声明 `color_range`。
* 零长度区间、非正速度、逆序锚点、三重重叠、转场超出实际重叠。
* 引用不存在的媒体、片段超出工程时长、淡入淡出总和超过片段时长。
