# KCD2 中文配音流水线（IndexTTS 2.x 版）操作手册

用 IndexTTS 替换原来的 CosyVoice：读入英文 `.ogg`，用 `text_ui_dialog.xml` 里的**官方中文翻译**合成中文语音，
输出**同名同路径**的 `.ogg`。找不到译文 / 译文为空 / 译文里没有中文 → **直接复制英文原文件**。

---

## 0. 交付文件（4 个）

| 文件 | 作用 |
|---|---|
| `kcd_indextts.py` | 主脚本（含 `check / stats / map / set-speaker / run` 五个子命令） |
| `kcd_indextts_config.json` | 全部配置：路径、模型、对齐、编码、运行参数 |
| `speaker_map.json` | 说话人 → 音色（参考音频）映射，`map` 命令自动生成，可手工编辑 |
| `README_indextts.md` | 本文档 |

运行时自动产出：`refs/`（每个说话人一个参考 wav）、`work/`（索引缓存、临时文件）、`logs/`（每次运行的日志）。

## 1. 数据事实（已在本机核对，可直接采信）

| 项 | 值 |
|---|---|
| 输入文件总数 | 299892 个 `.ogg`（part0/1/2/3 = 65535/65535/65535/63435，IPL = 39852） |
| XML 行数 | 177930 |
| 能拿到官方中文的比例 | **288900 / 299892 = 96.3%** |
| 需要复制英文的 | 10992（都是译文为 `<...>` 或与英文相同这类空译文） |
| 说话人代号 | 165 个（= 文件名第一个 `_` 之前的 token），其中 163 个能自动找到参考音频 |
| 音频规格 | OGG/Vorbis，48000 Hz，单声道，标称 54 kbps |

匹配规则（脚本已实现）：XML 里的 id **不含**说话人前缀，文件名**含**。
所以先按文件名整名精确匹配，再去掉最左边的 token 逐级匹配（最多剥 4 个），实测全部能命中。

说话人提取：`<speaker>_<其他>_<4位随机>.ogg` 的第一段就是说话人代号（如 `tmck`、`jhig1`）。
少数文件名格式不同（`na_`、`v_`、`cs_` 等），会单独成为一个"说话人"，用 `set-speaker` 手工指到正确音色即可。

---

## 2. 一次性准备

### 2.1 环境

```bash
conda activate /home/caijin/private/text2audio/kcdcdm_env/
cd /home/caijin/private/text2audio/KDCCDM_Script
python -V                      # 该环境是 Python 3.10.8, torch 2.3.1+cu118
```

> 如果 IndexTTS2.5 与这个环境里的 torch / transformers 冲突，就新建一个环境（例如
> `/home/caijin/private/text2audio/indextts_env`）装 IndexTTS，然后用**那个环境的 python** 跑本脚本，
> 脚本本身不依赖任何特定环境。

### 2.2 需要手动下载的东西（脚本绝不下载）

**① IndexTTS 代码仓库 → `/home/caijin/private/text2audio/IndexTTS2`**

```bash
git clone https://github.com/index-tts/index-tts.git /home/caijin/private/text2audio/IndexTTS2
# 如果 IndexTTS2.5 是另一个仓库/分支，就 clone 对应版本到同一路径
```

**② 模型权重 → `/home/caijin/private/text2audio/IndexTTS2/checkpoints`**

以 ModelScope `IndexTeam/IndexTTS-2`（或你手上 2.5 对应的仓库）为例，把 `checkpoints/` 整个目录下下来：

| 需要的东西 | 放到 |
|---|---|
| `config.yaml` | `…/IndexTTS2/checkpoints/config.yaml` |
| `gpt.pth`（约 1.6 GB） | `…/checkpoints/gpt.pth` |
| `s2mel.pth`（约 1.1 GB） | `…/checkpoints/s2mel.pth` |
| `bpe.model` | `…/checkpoints/bpe.model` |
| `feat1.pt`、`feat2.pt` | `…/checkpoints/` |
| `wav2vec2bert_stats.pt` | `…/checkpoints/` |
| `qwen0.6bemo4-merge/`（config.json + model.safetensors + tokenizer） | `…/checkpoints/qwen0.6bemo4-merge/` |

- 缺失的文件用 `python kcd_indextts.py check` 一条命令就能看出来（它会打印这份清单）。
- 若运行时报缺 `w2v-bert-2.0` 之类的 HF 模型：下载到 `…/IndexTTS2/w2v-bert-2.0`，或先联网跑一次让它自动
  缓存；之后可以 `export HF_HUB_OFFLINE=1` 免得每次都联网。国内网络可 `export HF_ENDPOINT=https://hf-mirror.com`。
- 缺 Python 包时才装（按报错装即可）：
  `pip install "transformers>=4.36" sentencepiece omegaconf munch einops`

### 2.3 路径确认

打开 `kcd_indextts_config.json`，确认这几项（其他保持默认即可）：

```
input_root   /home/caijin/private/text2audio/voicein
output_root  /home/caijin/private/text2audio/voiceout
xml_path     …/KDCCDM_Script/text_ui_dialog.xml
indextts.code_root / cfg_path / model_dir / device(cuda:0 或 cuda:1)
```

---

## 3. 运行步骤（照抄即可）

### 第 0 步：自检（不加载模型，几秒）

```bash
python kcd_indextts.py check
```
看到 `模型文件: 齐全`、`import indextts.infer_v2_5 OK` 之类即可。
如果报 `IndexTTS 加载失败`，把 `check` 输出里那个能 import 成功的"模块名 + 类名"填到配置的
`indextts.class_candidates` **第一位**。

### 第 1 步：统计（dry-run，不写任何音频）

```bash
python kcd_indextts.py stats
python kcd_indextts.py stats --part IPL_english        # 只统计一个 part
```
输出每个 part 的"待合成 / 待复制"数量、合成量最大的说话人、以及哪些说话人还没有参考音频。

### 第 2 步：生成说话人音色表（**当前方案下可跳过**，见第 5.5 节）

音色/情绪默认取自各条英文原音（`spk_source=line`），所以 `map` 不再是必需的，
只在你想给"短于 0.8 秒的短句"准备兜底音色、或想回到"一人一音"模式时才需要跑：

```bash
python kcd_indextts.py map   # 可选
```
- 每个说话人挑一条**他自己**的英文音频做参考（英文文本 15~130 字符、时长 2.5~9 秒），
  转成 `refs/<speaker>.wav`，写进 `speaker_map.json`。
- 想只处理部分 part：`python kcd_indextts.py map --part IPL_english english-part0`
- 结果里 `没有参考音频` 的说话人（通常是喘气/拟声）会走"复制英文"。

### 第 3 步：小批量测试（强烈建议先用 IPL_english 试）

```bash
# 先挑一个最小的子目录 + 5 个文件
python kcd_indextts.py run --part IPL_english --subdir open_world/boj_a_zraneni --limit 5

# 单个说话人试听音色是否稳定
python kcd_indextts.py run --part IPL_english --speaker jelm --limit 20

# 只跑某几个文件（把文件路径写进 list.txt，一行一个，可用相对 input_root 的路径）
python kcd_indextts.py run --files-from list.txt
```
输出在 `/home/caijin/private/text2audio/voiceout/IPL_english/dialog/...`，路径/文件名与输入完全一致。
试听后不满意就改 `speaker_map.json`（见第 6 节），`--force` 只重跑这批。

### 第 4 步：全量（双卡并行，各跑一半）

```bash
# 卡 0
python kcd_indextts.py run --shard 1 2 --device cuda:0
# 卡 1（另开一个终端）
python kcd_indextts.py run --shard 2 2 --device cuda:1
```
按 part 分也可以（比如一台机器一个 part）：
```bash
python kcd_indextts.py run --part english-part0 english-part1 --device cuda:0
python kcd_indextts.py run --part english-part2 english-part3 --device cuda:1
```
长时间跑建议挂后台：
```bash
nohup python kcd_indextts.py run --shard 1 2 --device cuda:0 > /dev/null 2>&1 &
tail -f logs/run_*.log
```

---

## 4. 断点续跑 / 失败重跑

- **断点续跑**：输出 `.ogg` 已存在且 >512 字节就跳过，直接重跑原命令即可，不会重做。
- **失败清单**：每次 `run` 结束会在 `logs/failed_YYYYmmdd_HHMMSS.txt` 写下失败文件的绝对路径，重跑：
  ```bash
  python kcd_indextts.py run --files-from logs/failed_20260101_120000.txt
  ```
- **危险保护**：连续失败 20 次（`run.max_consecutive_failures`）会主动停下，避免模型/环境坏了还空跑几小时。
- 合成失败的条目**不会**自动复制英文（`copy_on_tts_failure: false`），这样重跑才有意义；想改成"失败就复制英文"把它设成 `true`。
- `--force` 强制重做；`--keep-temp-wav` 保留中间 wav 便于排查。

---

## 5. 时长对齐（已内置，无需手工 ffmpeg）

**只依据 ffprobe 的实测时长动作，不做任何估算**（早期版本用"中文 X 字/秒"估算语速，实测误差高达 75%，
已删除——它会把好好的音频加速坏）。每条音频的流程：

1. IndexTTS 按**自然语速**生成 wav（音色/情绪取自各条英文原音，见第 5.5 节）。
2. 量原始英文 ogg 的目标时长 `T` 和生成音频的实测时长 `G`。
3. 偏差超过 3%（`align.tolerance`）时用 ffmpeg `atempo` 变速（保音高、不变调），
   系数 = `G/T`，限制在 **0.80 ~ 1.50**（`atempo_min/max`）。
4. 变速后不足 `T` → `apad=whole_len` 补静音到**精确** `T`；
   超出 `T` 但不超过 `align.allow_overflow_sec`（默认 0.20s）→ **保留不裁**，避免切掉尾音；
   超出更多才裁到 `T + allow_overflow_sec`。
5. 编码：默认 **soundfile 写 OGG/VORBIS**（本机 ffmpeg 4.0.5 没有 `libvorbis`，已验证 soundfile 能写出
   合法的 `vorbis 48000Hz 单声道` ogg），失败时依次回退 `ffmpeg libvorbis → ffmpeg vorbis → ffmpeg libopus`。
6. 输出参数对齐游戏原文件：`48000 Hz / 单声道 / vorbis`（`encode` 段可改）。

日志里每条都有 `时长 1.960->1.894 tempo=1.035 pad=0.000 over=+0.000 enc=soundfile_vorbis`，一眼能看出对齐情况。

**实测经验**：IndexTTS 中文自然语速生成的时长，跟英文原句往往已经很接近（实测比值 0.99~1.14），
所以绝大多数条目是 `tempo≈1.0、pad≈0`，听感最自然。

---

## 5.5 情绪 / 语气：1:1 迁移英文原音（**默认已开启，这是最终采用方案**）

IndexTTS2 把**音色**和**情绪**分开控制（代码里 `emovec = gpt.merge_emovec(spk_cond_emb, emo_cond_emb, alpha)`）：

- `spk_audio_prompt` → 决定"像谁"（音色）
- `emo_audio_prompt` → 决定"什么语气"（情绪）

**本流水线现在把每一条音频自己的英文 `.ogg` 同时用作这两者**：

```json
"emotion": {"emo_source": "line", "spk_source": "line",
            "emo_alpha": 1.0, "line_min_sec": 0.8}
```

| 字段 | 作用 |
|---|---|
| `spk_source: "line"` | 音色直接取自**该条英文原音**（同一位演员、同一句的语气下）→ 最贴近原声 |
| `emo_source: "line"` | 情绪取自**该条英文原音** → 原句的激动/平静/戏谑被 1:1 迁移 |
| `emo_alpha` | 情绪强度，1.0 = 完全用原音情绪；0.6~0.8 = 保留一点本人语气 |
| `line_min_sec` | **短句保护**：英文短于 0.8 秒时，音色与情绪退回 `speaker_map.json` 的固定参考音（极短音频的说话人嵌入不可靠） |

**为什么必须这样**：如果不传 `emo_audio_prompt`，模型内部会执行
`emo_audio_prompt = spk_audio_prompt; emo_alpha = 1.0`——即把那条固定的说话人参考音的情绪
100% 复制到他说过的**每一句**上。如果那条参考音恰好是喊叫台词（例如 `ldal` 的
`"I yield, enough!"`），就会出现"所有台词感情过于充沛、和人物不符"的问题。

**结论（实测对比 5+4 组参数后确定）**：用每条英文原音做音色+情绪参考（`spk_source/spk_emo = line`）
效果最好，中文能够还原原演员的音色与情绪强度；代价是**每条都要重算一次参考音特征**
（音色缓存失效，约 +0.1~0.2 秒/条），可忽略。

**副作用（是好事）**：`map` 这一步不再必需——因为音色/情绪来自原音，`speaker_map.json` 现在只是
短句的兜底。也可以不再 `map`，直接 `run`。

> 注意："1:1"指的是**情绪色彩与强度的迁移**（用原句音频作为条件），不是逐波形复刻；
> 中文音节和英文不同，TTS 是重新合成，时长仍由第 5 节的算法对齐。

### 可选：二次生成（`align.use_native_speed`，默认 `false`）

当某条中文比英文长/短很多时（`|G/T - 1| > align.native_trigger`，默认 0.45），
可以用 IndexTTS2.5 原生的 `duration_factor` **精确重生成一次**再对齐：

```json
"align": {"use_native_speed": true}
```
代价是这些条目要跑两遍模型（约 2 倍耗时）。默认关闭，因为实测大部分条目不需要。

> 极端情况（中文 3 秒、原文只有 1 秒）：`atempo_max` 会卡住，超出 `allow_overflow_sec` 的部分被裁掉。
> 这时可以调大 `align.atempo_max`（例如 1.6）、打开 `use_native_speed`，或把 `align.enabled` 设 `false` 先不管时长。

---

## 6. 说话人音色修正（防止一人多音）

优先级：`speaker_map.json[i].ref_audio` → `use` 指向的说话人 → `default.ref_audio` → 都没有就复制英文。

- **换某个说话人的音色**（用他另一条音频）：
  ```bash
  python kcd_indextts.py set-speaker --speaker tmck \
      --src-ogg /home/caijin/private/text2audio/voicein/english-part0/dialog/xxx/tmck_xxx.ogg --lock
  ```
  `--lock` = 以后 `map` 不再自动改他。`--ref-text "英文原文"` 可手动指定参考文本。
- **把两个代号合成一个人**（同一演员不同代号）：
  ```bash
  python kcd_indextts.py set-speaker --speaker M24 --use M26 --lock
  ```
  或直接在 `speaker_map.json` 里手写 `"M24": {"use": "M26"}`。
- **一批人都用同一个音色**：把 `default.ref_audio` 指到一个 wav，并把这些人的 `ref_audio` 设为 `null`。
- 改完直接重跑（记得 `--force`，否则已生成的不会重做）：
  ```bash
  python kcd_indextts.py run --part IPL_english --speaker tmck --force
  ```

关于"88 个配音演员"：文件名里能提取出 165 个代号（包含 `M24/F48/tonya` 这类通用语音类型和少量
`na_/v_/cs_` 这类不规范命名），不是 88 个演员的精确切分。脚本按"代号 = 音色"处理，能保证
**同一个代号不会出现多个音色**；要严格合并到 88 人，用上面的 `use` / `locked` 手工挂即可。

---

## 7. 常见错误处理

| 现象 | 原因 / 处理 |
|---|---|
| `IndexTTS 加载失败 ... 最后一个错误` | `code_root/cfg_path/model_dir` 不对，或类名不对：把 `check` 里能 import 成功的模块+类名填到 `class_candidates` 第一位 |
| `FileNotFoundError: gpt.pth / s2mel.pth` | 模型没下全，按 `check` 打印的清单补齐到 `model_dir` |
| `缺 w2v-bert-2.0 / semantic codec` | 该 HF 模型未缓存；下载到 `code_root/w2v-bert-2.0` 或先联网跑一次再 `HF_HUB_OFFLINE=1` |
| `CUDA out of memory` | 配置里 `init_kwargs.use_fp16=true`、`use_deepspeed=false`；确认没有别的进程占卡（`nvidia-smi`）；仍不够就单卡跑，别两个进程用同一张卡 |
| 卡住不动 | 日志最后一行是哪个文件；`--limit 1 --keep-temp-wav` 单文件复现。真的挂了就 Ctrl-C，已完成的不会重做 |
| `Unknown encoder 'libvorbis'` | 正常（本机 ffmpeg 4.0.5 就没有）。默认走 `soundfile_vorbis`；日志里的 `enc=` 会告诉你实际用的编码器 |
| 输出 0 字节 / 播放无声 | `encode.min_output_bytes` 会拦住这种文件（不会写出去）；看日志该条是否 `[fail]` |
| 中文里有 `<...>`、`%s` | 脚本会自动清掉；清完没中文就复制英文 |
| `连续失败 N 次，提前停止` | 先解决模型/环境问题，再 `--files-from logs/failed_*.txt` 续跑 |
| 译文明显是机翻而不是官方中文 | 不用管：脚本只读 XML 第 3 列（官方中文），不会调用任何翻译 API |

---

## 8. 常用命令速查

```bash
python kcd_indextts.py check                                   # 自检
python kcd_indextts.py stats [--part X] [--subdir a/b] [--limit N]
python kcd_indextts.py map   [--part X] [--part Y] [--rebuild-index]
python kcd_indextts.py set-speaker --speaker tmck --src-ogg /abs/x.ogg --lock
python kcd_indextts.py run   [--part X] [--subdir a/b] [--speaker S] [--limit N]
                             [--files-from list.txt] [--shard 1 2]
                             [--device cuda:0] [--force] [--dry-run] [--keep-temp-wav]
```

## 9. 结果自检（跑完之后）

```bash
# 数量对比（应与输入一致）
for d in english-part0 english-part1 english-part2 english-part3 IPL_english; do
  echo -n "$d in=$(find ~/private/text2audio/voicein/$d -name '*.ogg' | wc -l) "
  echo    "out=$(find ~/private/text2audio/voiceout/$d -name '*.ogg' 2>/dev/null | wc -l)"
done

# 抽查一条：时长是否与原文件一致、编码是否是 vorbis 48000 单声道
f=dialog/open_world/boj_a_zraneni/jelm_heka_boj_hekani__pAqQ.ogg
ffprobe -v error -show_entries format=duration -of default=nw=1 voicein/IPL_english/$f
ffprobe -v error -show_entries format=duration -of default=nw=1 voiceout/IPL_english/$f
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -of default=nw=1 voiceout/IPL_english/$f
```

## 10. 性能参考

单卡 A5000（24 GB）串行跑，IndexTTS fp16 大约占 8~10 GB 显存；短句一般每句 0.5~2 秒
（含 ffmpeg 对齐与编码）。30 万条约需 2~5 天/卡，所以：
- 一开始就 `--shard 1 2 / --shard 2 2` 双卡一起跑；
- `run.group_by_speaker=true` 会让同一说话人连续处理，命中 IndexTTS 的音色缓存，明显更快；
- 每 200 条自动 `torch.cuda.empty_cache()`（`run.empty_cache_every`），长跑不会慢慢涨显存。
