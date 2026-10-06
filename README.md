# kcd-chinese-dubbing

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

给《天国：拯救》和《天国：拯救 2》做中文配音的流水线。

游戏原本只有英配。这个脚本把英文语音转成中文语音——**用它自己的英文原音当参考**，
所以中文听起来是同一个演员在说，而不是随便找个 AI 声音念稿。

二代做了 30 万条。序章、主线、支线、DLC、战斗喊话，全做了。

---

## 原理

一句话：**每条中文语音，都拿它对应的那句英文原音当音色参考和情绪参考。**

IndexTTS 2.5 的 `infer()` 有两个独立入口：

- `spk_audio_prompt` —— 决定"像谁"（音色）
- `emo_audio_prompt` —— 决定"什么语气"（情绪）

大部分 demo 只传第一个，第二个留空。但代码里这行：

```python
if emo_audio_prompt is None:
    emo_audio_prompt = spk_audio_prompt
    emo_alpha = 1.0
```

意味着**每个说话人会共用一条固定的情绪**。如果挑到的那条参考音正好是句喊叫
（我的 `ldal` 就被挑中了 `"I yield, enough!"`），那这个人说的每一句话都会很激动。

所以这里两个参数都传同一条英文原音：

```python
tts.infer(
    spk_audio_prompt = src_ogg,   # 这句的英文原音
    emo_audio_prompt = src_ogg,   # 同一句
    emo_alpha        = 1.0,
    lang  = "ZH",
    text  = 官方中文译文,
    output_path = out_wav,
)
```

文本用游戏自带的 `text_ui_dialog.xml` 第三列（官方中文），**不调任何翻译 API**。

---

## 时长

中文和英文的信息密度不一样，念出来长度经常对不上。做法是先按模型自然语速生成，
再用 ffmpeg 把实测时长往英文原时长上靠：

- 偏差 >3% 就 `atempo` 变速（保住音高不变调），系数限制在 0.80~1.50
- 短了就 `apad` 补静音到精确时长
- 超了 20% 以内**不裁**（怕切掉尾音），超更多才裁

实测下来中文的自然长度跟英文原句经常已经很接近，所以大部分情况 `tempo≈1.0`、不补静音。

---

## 用法

```bash
# 环境自检（不加载模型）
python kcd_indextts.py check

# 先看看有多少条要跑、多少条没译文
python kcd_indextts.py stats --part english-part0

# 跑 5 条试试
python kcd_indextts.py run --part english-part0 --limit 5 --device cuda:0

# 正式跑，4 个分片铺到 2 张卡上
for i in 1 2 3 4; do
  dev=$([ $i -le 2 ] && echo 0 || echo 1)
  nohup python kcd_indextts.py run --part english-part0 --shard $i 4 --device cuda:$dev \
      > logs/part0_$i.out 2>&1 &
done
```

输出目录结构和输入完全一样，文件名也一样。**已存在的输出会自动跳过**，所以中途挂了
直接重跑同一条命令就行，不会重做。失败条目会写到 `logs/failed_*.txt`，用
`--files-from` 补跑。

其他子命令：

| 命令 | 干嘛的 |
|---|---|
| `check` | 检查环境、模型路径、缺哪些文件 |
| `stats` | 只统计不生成，先看清楚了再跑 |
| `map` | 生成 `speaker_map.json`（只有用 `spk_source=map` 时才需要）|
| `set-speaker` | 手动给某个说话人指定音色 |

---

## 配置

所有配置都在 `kcd_indextts_config.json`，常改的几个：

```json
"input_root":  "./voicein",
"output_root": "./voiceout",
"xml_path":    "./text_ui_dialog.xml",
"indextts": {
  "code_root": "./IndexTTS2",
  "cfg_path":  "./IndexTTS2/checkpoints/config.yaml",
  "model_dir": "./IndexTTS2/checkpoints",
  "device":    "cuda:0"
}
```

情绪相关的（默认就是效果最好的那套，一般不用动）：

```json
"emotion": {
  "spk_source": "line",   // 音色取自该条英文原音
  "emo_source": "line",   // 情绪也取自该条英文原音
  "emo_alpha":  1.0,      // 1.0 = 完全用原音情绪，调小会保留一点本音
  "line_min_sec": 0.8     // 短于 0.8 秒的句子退回 speaker_map（太短的音频算不准）
}
```

---

## 要自己准备的东西

仓库里**只有代码**。下面这些自己下：

**1. IndexTTS 2.5**

```bash
git clone https://github.com/index-tts/index-tts.git
```

**2. 模型权重** —— 放 `checkpoints/`，或者改配置指向你的路径

| 来源 | 怎么下 |
|---|---|
| ModelScope | `modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints` |
| HuggingFace | `hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints` |
| 国内镜像 | `export HF_ENDPOINT=https://hf-mirror.com` |

需要 `config.yaml`、`gpt.pth`、`s2mel.pth`、`codec.pth`、`bpe.model`、`feat1.pt`、`feat2.pt`、
`wav2vec2bert_stats.pt`、`qwen0.6bemo4-merge/`。

> 2.5 的仓库里**没有** `bpe.model`，从 `IndexTeam/IndexTTS-2` 拿一个（475KB），放同一个目录就行。
> 实际推理路径不读它，但检查脚本会提示缺失。

**3. 辅助模型** —— 首次运行自动下到 `checkpoints/hf_cache/`。没网就手动放：

| 文件 | 来源 |
|---|---|
| `hf_cache/w2v-bert-2.0/` | `facebook/w2v-bert-2.0` |
| `hf_cache/campplus_cn_common.bin` | `funasr/campplus` |
| `hf_cache/bigvgan/` | `nvidia/bigvgan_v2_22khz_80band_256x` |
| `hf_cache/semantic_codec_model.safetensors` | `amphion/MaskGCT` 的 `semantic_codec/model.safetensors` |

**4. 游戏素材** —— 英文 `.ogg` 和 `text_ui_dialog.xml`，从游戏文件里自己提取。
**本仓库不提供。**

**5. Python 环境**

```bash
pip install -U "transformers==4.52.1" "numpy<2" descript-audiotools==0.7.2
```

`descript-audiotools` 容易漏——`indextts/s2mel/dac/__init__.py` 依赖它（`import audiotools`），
但官方依赖清单里没列，报 `ModuleNotFoundError: No module named 'audiotools'` 就是缺它。

---

## 避坑

留个记录，省得你再踩一遍。

**别把 `emo_audio_prompt` 留空。** 上面说过，会变成整个说话人共用一个情绪。

**`transformers` 版本要 4.52+。** 2.5 的 `qwen0.6bemo4-merge` 是 `model_type: qwen3`，
4.47 及以前不认识这个架构，加载情绪模型时会报
`does not recognize this architecture`。

**`use_accel=True` 不一定更快。** 需要 `flash_attn`，我装了 2.5.8 版本实测：
GPT 解码确实快了 16%（1.48s → 1.24s），但总推理时间没变，整体反而慢 10%，
而且每进程多吃 1.8GB 显存。2 张卡各跑 2 个进程时，显存带宽已经满了，
省下的时间被 accel 自己的开销吃回去了。**每卡只跑 1 个进程时才可能划算。**

**`use_torch_compile=True` 实测没用。** 稳态 s2mel 耗时和不开基本一样，还多花 25 秒编译。

**别盲目加进程。** A5000 上每进程约 6.1GB 显存。24GB 的卡放 4 个就是 24.4GB，直接 OOM。
2 个进程时功耗已经到 228W/230W 的上限了，加进程不会更快。

**这张表里 `bpe.model` 是个坑。** 见上面模型那节。

---

## 已知问题

- **语音和口型对不上。** 时长已经尽量对齐，但逐音素的节奏是模型重新生成的，中文也搬不过来。
- **官方没译文的条目会保留英文原声。** KD2 约 3.7%（大多是喘气、喊叫这类拟声），
  KD1 约 12%（KD1 官方中文本身就比二代少）。
- **同一句话重跑结果会有细微差别**，采样是随机的，不是确定性输出。

---

## 性能参考

2 × A5000，4 个进程（每卡 2 个），约 **4500~5000 条/小时**。

- KD2：29.99 万条，约 2.6 天
- KD1：15.2 万条，约 1.4 天

单进程约 1200 条/小时。

---

## 许可

代码 MIT。

游戏语音、文本、模型权重的版权归各自所有者所有，本仓库不包含这些内容。

用之前先确认你手上的游戏素材来源合法合规，转载需注明出处。
