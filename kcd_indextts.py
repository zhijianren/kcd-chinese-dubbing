#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KCD2 中文配音流水线 (IndexTTS 2.x)
=================================

输入: <input_root>/<part>/.../<name>.ogg        英文语音（目录/文件名保持原样）
输出: <output_root>/<part>/.../<name>.ogg       中文语音（结构与输入完全一致）
译文: text_ui_dialog.xml 第 3 列（官方中文）；找不到 / 译文为空 / 无中文 时，直接复制英文 ogg。

只依赖: ffmpeg + ffprobe + numpy + soundfile（可选 torch/indextts 仅生成时需要）。
找不到 libvorbis 的机器上默认用 soundfile 写 OGG/VORBIS（已在本机验证可用）。

子命令:
  check         环境 / 模型路径自检（不加载模型）
  stats         只统计不生成: 多少条要合成、多少条要复制、多少条已完成
  map           扫描输入 → 生成/更新 speaker_map.json + refs/<speaker>.wav
  set-speaker   手动修正某个说话人的参考音频（音色）
  run           正式生成（支持断点续跑 / 小批量 / 分片 / 指定文件清单）

示例:
  python kcd_indextts.py check
  python kcd_indextts.py stats --part IPL_english
  python kcd_indextts.py map
  python kcd_indextts.py run  --part IPL_english --subdir open_world/boj_a_zraneni --limit 5
  python kcd_indextts.py run  --part english-part0 --shard 1 2 --device cuda:0
详见 README_indextts.md
"""

from __future__ import annotations

import argparse
import collections
import html
import importlib
import inspect
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import time

LOG = logging.getLogger("kcd")

# --------------------------------------------------------------------------- #
# 默认配置（会被 kcd_indextts_config.json 覆盖；这里不写死任何模型路径之外的假设）
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "input_root": "./voicein",
    "output_root": "./voiceout",
    "xml_path": "./text_ui_dialog.xml",
    "speaker_map_path": "./speaker_map.json",
    "ref_dir": "./refs",
    "work_dir": "./work",
    "log_dir": "./logs",

    "parts": ["english-part0", "english-part1", "english-part2",
              "english-part3", "IPL_english"],
    "audio_ext": ".ogg",
    "copy_other_files": False,

    # 说话人提取: 文件名第一个 '_' 之前的部分就是说话人代号
    "speaker_regex": "^([^_]+)_",
    "max_strip_tokens": 4,          # 匹配 xml 时最多从左边剥掉几个 token
    "speaker_aliases": {},          # 例: {"M24": "M26"} 合并到同一音色

    "ref": {
        "min_text_chars": 15,       # 参考音频对应英文文本长度范围
        "max_text_chars": 130,
        "min_sec": 2.5,             # 参考音频时长范围（秒）
        "max_sec": 9.0,
        "probe_candidates": 40,     # 每个说话人最多 ffprobe 几个候选
        "wav_sr": 24000,            # 参考 wav 采样率
        "target_bytes_per_sec": 6750.0,   # 54000bps/8，用于按体积粗排候选
        # 优先挑"平静"的台词当参考音：IndexTTS2 默认会把参考音的情绪 100%
        # 复制到该说话人的每一句上，挑到喊叫台词就会出现"感情过于充沛"。
        "prefer_neutral": True,
        "neutral_min_score": 1.0,
    },

    "emotion": {
        # 情绪来源: "line" = 拿这条音频自己的英文原音当情绪参考(1:1 迁移原声感情，推荐)
        #           "ref"  = 用说话人参考音的情绪(旧行为，会造成全说话人一个情绪)
        #           "none" = 不指定，等同 "ref"
        "emo_source": "line",
        # 情绪融合强度(传给 gpt.merge_emovec 的 alpha)。1.0 = 完全用情绪参考音，
        # 想保留一点本人音色自带的语气就调到 0.6~0.8。
        "emo_alpha": 1.0,
        # 可选: 显式 8 维情绪向量 [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]
        # 设了它就会覆盖 emo_source（两者不能同时用）。
        "emo_vector": None,
        # 音色参考: "line" = 用这条音频自己的英文原音(音色+情绪全部 1:1，实测效果最好)
        #           "map"  = 用 speaker_map.json 里固定给该说话人挑的(一人一音)
        "spk_source": "line",
        # 短句保护: 英文原音短于该秒数时，音色与情绪都退回 speaker_map 里的固定参考音
        # (极短音频算出的说话人嵌入不可靠，容易音色漂移)
        "line_min_sec": 0.8,
    },

    "indextts": {
        "code_root": "<IndexTTS2 目录>",
        "extra_python_paths": [],
        "cfg_path": "<IndexTTS2 目录>/checkpoints/config.yaml",
        "model_dir": "<IndexTTS2 目录>/checkpoints",
        "device": "cuda:0",
        # 依次尝试 (模块, 类名)，适配不同版本的 IndexTTS / IndexTTS2 / IndexTTS2.5
        "class_candidates": [
            ["indextts.infer_v2_5", "IndexTTS2"],     # IndexTTS 2.5 官方写法
            ["indextts.infer_v2_5", "IndexTTS2_5"],
            ["indextts.infer_v2", "IndexTTS2"],
            ["indextts.infer", "IndexTTS"],
        ],
        "init_kwargs": {"use_bf16": True, "use_fp16": False,
                        "use_cuda_kernel": False, "use_deepspeed": False},
        "infer_kwargs": {"verbose": False,
                         "max_text_tokens_per_segment": 120},
        "allow_unknown_infer_kwargs": False,   # true 时把 infer_kwargs 里未知参数也强塞给 infer()
        # IndexTTS 2.5 原生参数（脚本会自动探测 infer() 的签名，不支持就自动省略）
        "lang": "ZH",              # 2.5 需要显式语言: ZH / EN / JA / ES / AR
        "duration_factor": 1.0,    # 2.5 原生语速: >1 变慢, <1 变快, 范围 0.5~2.0
        "use_duration_factor": True,  # 是否用模型原生语速对齐时长(ffmpeg 仍是兜底)
        "required_files": ["config.yaml", "gpt.pth", "s2mel.pth", "codec.pth",
                           "bpe.model", "feat1.pt", "feat2.pt",
                           "wav2vec2bert_stats.pt"],
        "required_dirs": ["qwen0.6bemo4-merge"],
        "sample_rate_fallback": 22050,   # 2.5 输出 22.05 kHz；模型返回数据而非写文件时用
    },

    "text": {"max_chars_per_segment": 100, "max_segments": 8, "min_cjk_chars": 1},

    "align": {
        "enabled": True,
        "tolerance": 0.03,          # 时长偏差小于 3% 就不动
        "atempo_min": 0.80,         # 最多放慢到 0.80x
        "atempo_max": 1.50,         # 最多加快到 1.50x
        "global_speed": 1.0,        # 整体语速系数，1.0 = 严格对齐时长
        "max_target_sec": 30.0,     # 目标时长超过该值的文件不做对齐
        "pad_to_target": True,      # 不足目标时长则补静音到精确时长
        "allow_overflow_sec": 0.20, # 超出目标时长但不超过这个值就不裁，避免切掉尾音
        # 二次生成: 先按自然语速生成并实测，偏差太大才用 IndexTTS2.5 的
        # duration_factor 重生成一次(精确，不靠估算)。默认关，避免每条都跑两遍。
        "use_native_speed": False,
        "native_trigger": 0.45,     # |实测/目标 - 1| 超过该比例才二次生成
        "native_min": 0.55,         # duration_factor 下限(官方范围 0.5~2.0)
        "native_max": 1.80,         # duration_factor 上限
    },

    "encode": {
        "ffmpeg": "ffmpeg",
        "ffprobe": "ffprobe",
        "sample_rate": 48000,
        "channels": 1,
        "bitrate": "54k",
        "encoder_chain": ["soundfile_vorbis", "ffmpeg_libvorbis",
                          "ffmpeg_vorbis", "ffmpeg_libopus"],
        "min_output_bytes": 512,
        "timeout_sec": 300,
    },

    "run": {
        "limit": 0,
        "group_by_speaker": True,       # 同一说话人连续处理，命中 IndexTTS 音色缓存
        "empty_cache_every": 200,
        "max_consecutive_failures": 20, # 连续失败太多就停，避免白跑
        "copy_on_tts_failure": False,   # 合成失败是否退回复制英文（默认不复制，方便重跑）
        "progress_every": 200,
        "keep_temp_wav": False,
        "cleanup_tmp_older_than_hours": 24,
    },
}

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CELL_RE = re.compile(r"<[^<>]{0,60}>")


class FatalError(RuntimeError):
    """致命错误：继续重试其它文件没有意义（例如模型加载失败）。"""


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def deep_merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = deep_merge(base[k], v) if k in base else v
        return out
    return over


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        cfg = deep_merge(cfg, user)
    return cfg


def setup_logging(log_dir, tag):
    os.makedirs(log_dir, exist_ok=True)
    # 带 pid：多个分片同时启动时不会写进同一个日志文件
    logfile = os.path.join(log_dir, "%s_%s_%d.log" % (
        tag, time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [fh, sh]
    return logfile


def run_cmd(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return -9, "", "timeout: %r" % (e,)
    except FileNotFoundError as e:
        return -1, "", "not found: %s" % e
    except Exception as e:                                  # noqa: BLE001
        return -1, "", repr(e)


def probe_duration(cfg, path):
    rc, out, err = run_cmd([cfg["encode"]["ffprobe"], "-v", "error",
                            "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", path], 90)
    if rc != 0:
        return None
    try:
        return float(out.strip())
    except ValueError:
        return None


def probe_sample_rate(cfg, path):
    rc, out, err = run_cmd([cfg["encode"]["ffprobe"], "-v", "error",
                            "-select_streams", "a:0",
                            "-show_entries", "stream=sample_rate",
                            "-of", "default=nw=1:nk=1", path], 90)
    if rc != 0:
        return None
    try:
        return int(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def probe_audio_info(cfg, path):
    """返回 (duration, sample_rate)。

    优先用 soundfile 直接读文件头（毫秒级、不起子进程）；失败再退回 ffprobe。
    热路径上每条音频能省掉 2~3 次子进程启动，约 0.1~0.2s/条。
    """
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate), int(info.samplerate)
    except Exception:                                       # noqa: BLE001
        pass
    return probe_duration(cfg, path), probe_sample_rate(cfg, path)


def stem_of(path):
    b = os.path.basename(path)
    return os.path.splitext(b)[0]


def sanitize_name(s):
    return re.sub(r"[^0-9A-Za-z_.\-]", "_", s) or "unknown"


def filter_kwargs(func, kwargs):
    """只保留目标函数签名支持的参数，用来兼容不同版本的 IndexTTS API。"""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# XML 索引 / 文本处理 / 说话人
# --------------------------------------------------------------------------- #
def load_dialog_index(cfg, rebuild=False):
    xml_path = cfg["xml_path"]
    if not os.path.exists(xml_path):
        raise SystemExit("找不到 xml: %s" % xml_path)
    os.makedirs(cfg["work_dir"], exist_ok=True)
    cache = os.path.join(cfg["work_dir"], "dialog_index.pkl")
    key = (os.path.abspath(xml_path), os.path.getmtime(xml_path),
           os.path.getsize(xml_path))
    if not rebuild and os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                ckey, data = pickle.load(f)
            if ckey == key:
                LOG.info("XML 索引缓存命中: %d 条", len(data))
                return data
        except Exception as e:                              # noqa: BLE001
            LOG.warning("索引缓存损坏(%r)，重新解析", e)
    import xml.etree.ElementTree as ET
    LOG.info("解析 XML: %s", xml_path)
    t0 = time.time()
    root = ET.parse(xml_path).getroot()
    data = {}
    for row in root:
        cells = [c.text for c in row]
        if len(cells) < 3:
            continue
        rid = (cells[0] or "").strip()
        if not rid or rid in data:
            continue
        data[rid] = ((cells[1] or "").strip(), (cells[2] or "").strip())
    LOG.info("XML 解析完成: %d 条, 用时 %.1fs", len(data), time.time() - t0)
    try:
        with open(cache, "wb") as f:
            pickle.dump((key, data), f, protocol=4)
    except Exception as e:                                  # noqa: BLE001
        LOG.warning("写索引缓存失败: %r", e)
    return data


def match_row(cfg, index, stem):
    """文件名可能自带说话人前缀(xml 里没有)，从左边逐个剥 token 去精确匹配。"""
    parts = stem.split("_")
    for k in range(min(len(parts), int(cfg["max_strip_tokens"]) + 1)):
        key = "_".join(parts[k:])
        if key in index:
            return index[key]
    return None


def clean_text(t):
    if not t:
        return ""
    t = html.unescape(t)
    t = t.replace("\\n", " ").replace("\r", " ").replace("\n", " ")
    t = CELL_RE.sub("", t)                       # 去掉 <...> 之类的标记
    t = re.sub(r"%(?:\d+\$)?[sdf]", "", t)       # 去掉 %s %d
    t = t.replace("«", "").replace("»", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(" -—–·")


# "平静度"打分用的关键字（文件名里的捷克语语境词）
_EMO_TOKENS = ("heka", "posl", "smrtelny", "raneny", "boj", "vyho", "utok", "krik",
               "rva", "bark", "zranen", "bolest", "zemr", "umr", "zbab", "prch",
               "krve", "pomoc", "boli", "auj", "souboj", "vrazd")
_CALM_TOKENS = ("chat", "dial", "rozh", "povid", "proslov", "smluv", "obch",
                "kost", "alko", "jist", "hospod", "cesta", "mesi")


def neutral_score(path, en):
    """越大越"平静"。IndexTTS2 会把参考音的情绪复制到每一句，所以要挑平静的参考。"""
    s = 0.0
    if "!" in en:
        s -= 3.0
    if "?" in en:
        s -= 2.0
    if "..." in en or "…" in en:
        s -= 1.0
    if re.search(r"\b[A-Z]{3,}\b", en):
        s -= 2.0
    if not (25 <= len(en) <= 110):
        s -= 2.0
    low = path.lower()
    for t in _EMO_TOKENS:
        if t in low:
            s -= 3.0
    for t in _CALM_TOKENS:
        if t in low:
            s += 1.0
    return s


def speaker_of(cfg, stem):
    m = re.match(cfg["speaker_regex"], stem)
    spk = m.group(1) if m else stem
    aliases = cfg.get("speaker_aliases") or {}
    return aliases.get(spk, spk)


def decide(cfg, index, src):
    """返回 (mode, cn_text, en_text, reason)，mode ∈ {'tts','copy'}"""
    stem = stem_of(src)
    row = match_row(cfg, index, stem)
    if row is None:
        return "copy", None, None, "no_xml"
    en, cn = row
    cn2 = clean_text(cn)
    en2 = clean_text(en)
    if not cn2:
        return "copy", None, en2, "empty_zh"
    if len(CJK_RE.findall(cn2)) < int(cfg["text"]["min_cjk_chars"]):
        return "copy", None, en2, "no_cjk"
    return "tts", cn2, en2, ""


def split_text(cfg, text):
    max_chars = int(cfg["text"]["max_chars_per_segment"])
    if len(text) <= max_chars:
        return [text]
    pieces = [p.strip() for p in re.findall(r"[^。！？；!?;…]*[。！？；!?;…]?", text)]
    pieces = [p for p in pieces if p]
    segs, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > max_chars:
            segs.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        while len(s) > max_chars:
            out.append(s[:max_chars])
            s = s[max_chars:]
        if s:
            out.append(s)
    m = int(cfg["text"]["max_segments"])
    if len(out) > m:
        out = out[:m - 1] + ["".join(out[m - 1:])]
    return out or [text]


# --------------------------------------------------------------------------- #
# speaker_map.json
# --------------------------------------------------------------------------- #
EMPTY_MAP = {
    "_readme": [
        "每个说话人一条记录，决定这个说话人用哪个参考音频（音色）。",
        "ref_audio: 参考音频(wav)绝对路径；ref_text: 参考音频对应的英文文本（可选，部分版本会用）。",
        "use: 指向另一个说话人代号，共用其音色（人工合并同一个人）。",
        "locked: true 时 build/set 不会自动覆盖这条记录。",
        "files: 该说话人的音频文件数量（统计信息）。",
    ],
    "default": {"ref_audio": None, "ref_text": None},
    "speakers": {},
}


def load_speaker_map(cfg, create=True):
    path = cfg["speaker_map_path"]
    smap = json.loads(json.dumps(EMPTY_MAP))
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            smap = deep_merge(smap, data)
            smap.setdefault("speakers", {})
        except Exception as e:                              # noqa: BLE001
            LOG.warning("读取 speaker_map 失败(%r)，使用空表", e)
    elif create:
        LOG.info("speaker_map.json 不存在，将新建: %s", path)
    else:
        LOG.info("speaker_map.json 还不存在: %s（先跑 map 或 set-speaker）", path)
    return smap


def save_speaker_map(cfg, smap):
    path = cfg["speaker_map_path"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(smap, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)
    LOG.info("已写入 %s", path)


def resolve_ref(cfg, smap, speaker):
    """跟随 use 链找参考音频，返回 (ref_audio, ref_text) 或 (None, '')"""
    seen = set()
    entry = smap["speakers"].get(speaker)
    while entry and entry.get("use") and entry["use"] not in seen:
        seen.add(entry["use"])
        entry = smap["speakers"].get(entry["use"])
    if entry and entry.get("ref_audio") and os.path.exists(entry["ref_audio"]):
        return entry["ref_audio"], entry.get("ref_text") or ""
    d = smap.get("default") or {}
    if d.get("ref_audio") and os.path.exists(d["ref_audio"]):
        return d["ref_audio"], d.get("ref_text") or ""
    return None, ""


# --------------------------------------------------------------------------- #
# IndexTTS 适配层（不同版本 API 差异用签名过滤 + 名字探测兜底）
# --------------------------------------------------------------------------- #
AUDIO_KEYS = ("spk_audio_prompt", "speaker_audio_prompt", "audio_prompt",
              "prompt_speech_path", "prompt_speech", "prompt_audio", "ref_audio",
              "spk_audio", "prompt_wav", "speaker_wav", "voice_audio", "prompt")
TEXT_KEYS = ("text", "input_text", "txt", "content")
OUT_KEYS = ("output_path", "out_path", "output", "save_path", "wav_path", "out_file")
REFTEXT_KEYS = ("prompt_text", "spk_text", "ref_text", "speaker_text")


class IndexTTSEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.icfg = cfg["indextts"]
        self.tts = None
        self.cls = None

    # -- 加载 ------------------------------------------------------------- #
    def load(self):
        icfg = self.icfg
        for p in list(icfg.get("extra_python_paths") or []) + [icfg.get("code_root")]:
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        base = {"cfg_path": icfg["cfg_path"], "model_dir": icfg["model_dir"],
                "device": icfg["device"]}
        base.update(icfg.get("init_kwargs") or {})
        last = None
        for mod_name, cls_name in icfg["class_candidates"]:
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:                          # noqa: BLE001
                last = e
                LOG.info("跳过 %s.%s: %s", mod_name, cls_name, e)
                continue
            cls = getattr(mod, cls_name, None)
            if cls is None:
                LOG.info("模块 %s 里没有 %s", mod_name, cls_name)
                continue
            kwargs = filter_kwargs(cls.__init__, base)
            LOG.info("加载 IndexTTS: %s.%s(%s)", mod_name, cls_name,
                     ", ".join(sorted(kwargs)))
            try:
                self.tts = cls(**kwargs)
            except ImportError as e:
                # 缺 Python 包是环境问题，换下一个候选类也解决不了 → 直接终止
                miss = getattr(e, "name", None) or str(e)
                raise FatalError(
                    "初始化 %s.%s 时缺少依赖 / 导入失败: %s\n"
                    "  补装: python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
                    "\"numpy<2\" <pip包名>\n"
                    "  （import 名 -> pip 名 常见对照: audiotools->descript-audiotools, "
                    "cv2->opencv-python, unidic_lite->unidic-lite, g2p_en->g2p-en, "
                    "flash_attn->下载对应 wheel: "
                    "https://github.com/Dao-AILab/flash-attention/releases）"
                    % (mod_name, cls_name, miss)) from e
            except Exception as e:                          # noqa: BLE001
                last = e
                LOG.warning("初始化 %s.%s 失败: %r", mod_name, cls_name, e)
                continue
            self.cls = cls
            LOG.info("IndexTTS 加载成功: %s.%s", mod_name, cls_name)
            return
        raise FatalError(
            "IndexTTS 加载失败（最后一个错误: %r）。请检查 kcd_indextts_config.json 里 "
            "indextts.code_root / cfg_path / model_dir 是否指向你下载好的版本，"
            "必要时把真实的模块名+类名加到 class_candidates 第一位。" % (last,))

    # -- 调用 ------------------------------------------------------------- #
    def _infer_kwargs(self, text, ref_audio, ref_text, out_wav, duration_factor=None,
                      emo_audio=None):
        fn = self.tts.infer
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        names = list(params.keys())
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        kw = {}
        if has_var_kw:
            kw = {"spk_audio_prompt": ref_audio, "text": text, "output_path": out_wav}
        else:
            for k in AUDIO_KEYS:
                if k in names:
                    kw[k] = ref_audio
                    break
            for k in TEXT_KEYS:
                if k in names:
                    kw[k] = text
                    break
            for k in OUT_KEYS:
                if k in names:
                    kw[k] = out_wav
                    break
            if ref_text:
                for k in REFTEXT_KEYS:
                    if k in names:
                        kw[k] = ref_text
                        break
        # IndexTTS 2.5: lang 是必填参数(无默认值), duration_factor 是原生语速。
        # 必须在 has_var_kw 分支之外处理 —— 2.5 的 infer 带 **generation_kwargs，
        # 走上面的分支会漏掉它们。
        if "lang" in names and self.icfg.get("lang"):
            kw["lang"] = self.icfg["lang"]
        if duration_factor and "duration_factor" in names:
            kw["duration_factor"] = float(duration_factor)
        # 情绪控制: 优先级 显式向量 > 情绪参考音(line/ref)
        emo = self.cfg.get("emotion") or {}
        if emo.get("emo_vector") is not None:
            if "emo_vector" in names:
                kw["emo_vector"] = [float(x) for x in emo["emo_vector"]]
            if "emo_alpha" in names and emo.get("emo_alpha") is not None:
                kw["emo_alpha"] = float(emo["emo_alpha"])
        elif emo_audio and "emo_audio_prompt" in names:
            # 把这条音频对应的英文原音作为"情绪参考音" → 1:1 迁移原声的感情
            kw["emo_audio_prompt"] = emo_audio
            if "emo_alpha" in names:
                a = emo.get("emo_alpha")
                kw["emo_alpha"] = float(a) if a is not None else 1.0
        # 配置里的 infer_kwargs 只透传目标函数真正认得的具名参数：带 **kwargs 的版本
        # 会把未知参数塞进 generation_kwargs 传给模型，导致崩溃。
        # 真需要强制透传时，把 allow_unknown_infer_kwargs 设成 true。
        extra = {}
        strict = not self.icfg.get("allow_unknown_infer_kwargs")
        for kk, vv in (self.icfg.get("infer_kwargs") or {}).items():
            if kk in names or (has_var_kw and not strict):
                extra[kk] = vv
            elif kk not in names:
                LOG.debug("infer_kwargs 里的 %s 不在 infer() 签名中，已忽略", kk)
        extra.update(kw)
        return filter_kwargs(fn, extra)

    def synth(self, text, ref_audio, ref_text, out_wav, duration_factor=None,
              emo_audio=None):
        if os.path.exists(out_wav):
            os.remove(out_wav)
        kw = self._infer_kwargs(text, ref_audio, ref_text, out_wav, duration_factor,
                                emo_audio)
        ret = None
        try:
            ret = self.tts.infer(**kw)
        except TypeError as e:
            LOG.warning("infer(**kwargs) 类型不匹配(%s)，改用位置参数兜底", e)
            try:
                ret = self.tts.infer(ref_audio, text, out_wav)
            except TypeError:
                ret = self.tts.infer(text, ref_audio, out_wav)
        if inspect.isgenerator(ret):
            for _ in ret:
                pass
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 256:
            return True
        # 有些实现返回音频数据而不是写文件
        data = sr = None
        if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], int):
            sr, data = ret[0], ret[1]
        elif isinstance(ret, dict):
            for k in ("tts_speech", "wav", "audio", "speech"):
                if k in ret:
                    data = ret[k]
                    break
            sr = ret.get("sample_rate") or ret.get("sr")
        if data is not None:
            try:
                import numpy as np
                import soundfile as sf
                arr = data
                if hasattr(arr, "detach"):
                    arr = arr.detach().cpu().numpy()
                arr = np.asarray(arr).squeeze()
                sf.write(out_wav, arr, int(sr or self.icfg["sample_rate_fallback"]))
            except Exception as e:                          # noqa: BLE001
                LOG.error("模型返回了音频数据但写盘失败: %r", e)
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 256:
            return True
        return False


# --------------------------------------------------------------------------- #
# 音频: 时长对齐 + 编码 OGG/VORBIS
# --------------------------------------------------------------------------- #
def _enc_soundfile_vorbis(cfg, wav, out):
    import soundfile as sf
    data, sr = sf.read(wav, dtype="float32", always_2d=False)
    sf.write(out, data, sr, format="OGG", subtype="VORBIS")
    return True, ""


def _enc_ffmpeg(cfg, wav, out, codec):
    enc = cfg["encode"]
    cmd = [enc["ffmpeg"], "-hide_banner", "-v", "error", "-y", "-i", wav,
           "-ar", str(enc["sample_rate"]), "-ac", str(enc["channels"]),
           "-c:a", codec]
    if codec == "vorbis":
        cmd += ["-strict", "-2"]
    if codec in ("libvorbis", "libopus"):
        cmd += ["-b:a", str(enc["bitrate"])]
    cmd += ["-f", "ogg", out]
    rc, o, e = run_cmd(cmd, enc["timeout_sec"])
    return rc == 0, (e or o)[-300:]


ENCODERS = {
    "soundfile_vorbis": _enc_soundfile_vorbis,
    "ffmpeg_libvorbis": lambda c, w, o: _enc_ffmpeg(c, w, o, "libvorbis"),
    "ffmpeg_vorbis": lambda c, w, o: _enc_ffmpeg(c, w, o, "vorbis"),
    "ffmpeg_libopus": lambda c, w, o: _enc_ffmpeg(c, w, o, "libopus"),
}


def encode_ogg(cfg, wav_path, out_ogg):
    enc = cfg["encode"]
    tmp = out_ogg + ".tmp.ogg"
    errs = []
    for name in enc["encoder_chain"]:
        fn = ENCODERS.get(name)
        if fn is None:
            continue
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
            ok, err = fn(cfg, wav_path, tmp)
            if ok and os.path.exists(tmp) and os.path.getsize(tmp) >= enc["min_output_bytes"]:
                os.replace(tmp, out_ogg)
                return True, name
            errs.append("%s:%s" % (name, err))
        except Exception as e:                              # noqa: BLE001
            errs.append("%s:%r" % (name, e))
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return False, " | ".join(errs)[-400:]


def align_and_encode(cfg, in_wav, out_ogg, target_dur):
    """把 in_wav 对齐到 target_dur 后编码为 ogg。返回 (ok, info_dict)

    只依据 ffprobe 的实测时长动作，不做任何估算：
      - 偏差 > tolerance 时用 atempo 变速(系数 = 实测/目标，夹在 atempo_min~max)
      - 变速后不足目标时长 -> apad 补静音到精确时长
      - 超出目标时长 -> 只在超过 allow_overflow_sec 时才裁掉尾巴
    """
    al, enc = cfg["align"], cfg["encode"]
    gen, gen_sr = probe_audio_info(cfg, in_wav)
    info = {"gen": gen, "target": target_dur, "tempo": 1.0, "pad": 0.0,
            "over": 0.0, "encoder": ""}
    filters, trim_args = [], []
    if al["enabled"] and target_dur and 0 < target_dur <= float(al["max_target_sec"]):
        if gen and gen > 0.05:
            ratio = gen / target_dur
            f = 1.0
            if abs(ratio - 1.0) > float(al["tolerance"]):
                f = min(float(al["atempo_max"]),
                        max(float(al["atempo_min"]), ratio * float(al["global_speed"])))
            if abs(f - 1.0) > 0.01:
                filters.append("atempo=%.4f" % f)
                info["tempo"] = f
            gen2 = gen / f
            info["over"] = gen2 - target_dur
            if gen2 < target_dur and al.get("pad_to_target", True):
                # ffmpeg 4.0 的 apad 用 whole_len(采样点数)，不是 whole_dur
                sr = gen_sr or probe_sample_rate(cfg, in_wav) or 24000
                filters.append("apad=whole_len=%d" % int(round(target_dur * sr)))
                info["pad"] = target_dur - gen2
            trim_args = ["-t", "%.4f" % (target_dur + float(al["allow_overflow_sec"]))]
    aligned = in_wav + ".aligned.wav"
    cmd = [enc["ffmpeg"], "-hide_banner", "-v", "error", "-y", "-i", in_wav]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += trim_args + ["-ar", str(enc["sample_rate"]), "-ac", str(enc["channels"]),
                        "-c:a", "pcm_s16le", aligned]
    rc, o, e = run_cmd(cmd, enc["timeout_sec"])
    if rc != 0 or not os.path.exists(aligned):
        if os.path.exists(aligned):
            os.remove(aligned)
        return False, {"error": "align_ffmpeg: %s" % (e or o)[-300:]}
    ok, who = encode_ogg(cfg, aligned, out_ogg)
    info["encoder"] = who
    if os.path.exists(aligned):
        os.remove(aligned)
    return ok, info


def native_factor_needed(cfg, gen_dur, target_dur):
    """是否需要二次生成。依据实测时长判断，绝不估算。

    返回 (duration_factor 或 None, 说明文字)。
    IndexTTS 2.5 的 duration_factor: >1 变慢(时长变长), <1 变快。
    实测偏短(gen<target) -> 要变慢 -> factor = target/gen > 1；偏长则反之。
    """
    al = cfg["align"]
    if not al.get("use_native_speed"):
        return None, ""
    if not gen_dur or gen_dur <= 0.05 or not target_dur:
        return None, ""
    if target_dur > float(al["max_target_sec"]):
        return None, ""
    ratio = gen_dur / target_dur
    if abs(ratio - 1.0) <= float(al["native_trigger"]):
        return None, ""
    f = min(float(al["native_max"]), max(float(al["native_min"]), 1.0 / ratio))
    if abs(1.0 / f - ratio) > float(al["native_trigger"]):
        return f, "实测 %.3fs / 目标 %.3fs -> duration_factor=%.3f" % (
            gen_dur, target_dur, f)
    return None, ""


def synth_segments(cfg, engine, segments, ref_audio, ref_text, tmp_root, tag,
                   duration_factor=None, collect=None, emo_audio=None):
    """合成(可多段)并拼接，返回处理好的单个 wav 路径。"""
    wavs = []
    for k, seg in enumerate(segments):
        w = os.path.join(tmp_root, "%s_seg%d.wav" % (tag, k))
        if collect is not None:
            collect.append(w)
        if not engine.synth(seg, ref_audio, ref_text, w, duration_factor=duration_factor,
                            emo_audio=emo_audio):
            raise RuntimeError("合成未产出音频(segment %d)" % k)
        wavs.append(w)
    if len(wavs) > 1:
        merged = os.path.join(tmp_root, "%s_merged.wav" % tag)
        if collect is not None:
            collect.append(merged)
        got = concat_wavs(cfg, wavs, merged)
        if not got:
            raise RuntimeError("多段拼接失败")
        return merged
    return wavs[0]


def concat_wavs(cfg, wavs, out_wav):
    if len(wavs) == 1:
        return wavs[0]
    listfile = out_wav + ".list.txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for w in wavs:
            f.write("file '%s'\n" % os.path.abspath(w).replace("'", "'\\''"))
    rc, o, e = run_cmd([cfg["encode"]["ffmpeg"], "-hide_banner", "-v", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", listfile,
                        "-c", "copy", out_wav], cfg["encode"]["timeout_sec"])
    if os.path.exists(listfile):
        os.remove(listfile)
    if rc != 0 or not os.path.exists(out_wav):
        return None
    return out_wav


# --------------------------------------------------------------------------- #
# 任务收集
# --------------------------------------------------------------------------- #
def resolve_roots(cfg, parts, subdir):
    """把 --part / --subdir 解析成实际要遍历的目录列表。

    --subdir 的两种写法都支持：
      1) 相对 part:      --part IPL_english --subdir dialog/open_world/boj_a_zraneni
      2) 相对 input_root: --subdir IPL_english/dialog/open_world/boj_a_zraneni
    """
    if not subdir:
        return [os.path.join(cfg["input_root"], p) for p in parts]
    s = subdir.strip().strip("/\\")
    first = re.split(r"[\\/]", s, 1)[0]
    if os.path.isabs(subdir) or first in (cfg.get("parts") or []):
        return [os.path.join(cfg["input_root"], s)]
    return [os.path.join(cfg["input_root"], p, s) for p in parts]


def hint_dirs(cfg, root):
    """目录不存在时，打印最近一层存在的父目录下的子目录，方便用户改正路径。"""
    probe = os.path.dirname(os.path.abspath(root))
    for _ in range(6):
        if os.path.isdir(probe):
            break
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.path.isdir(probe):
        return
    try:
        names = sorted(n for n in os.listdir(probe)
                       if os.path.isdir(os.path.join(probe, n)))
    except OSError:
        return
    if names:
        LOG.error("  %s 下现有的目录: %s%s", probe, ", ".join(names[:20]),
                  " ..." if len(names) > 20 else "")


def collect_inputs(cfg, args, index=None):
    tasks = []
    files_from = getattr(args, "files_from", None)
    if files_from:
        with open(files_from, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line if os.path.isabs(line) else os.path.join(cfg["input_root"], line)
                if os.path.isfile(p):
                    tasks.append(os.path.abspath(p))
                else:
                    LOG.warning("清单里的文件不存在，跳过: %s", p)
    else:
        parts = getattr(args, "part", None) or cfg["parts"]
        subdir = getattr(args, "subdir", None)
        roots = resolve_roots(cfg, parts, subdir)
        for root in roots:
            if not os.path.isdir(root):
                LOG.error("目录不存在，跳过: %s", root)
                hint_dirs(cfg, root)
                continue
            LOG.info("扫描 %s", root)
            for dirpath, _dirnames, fns in os.walk(root):
                for fn in fns:
                    if fn.lower().endswith(cfg["audio_ext"]):
                        tasks.append(os.path.join(dirpath, fn))
                    elif cfg["copy_other_files"]:
                        tasks.append(os.path.join(dirpath, fn))
    tasks = sorted(set(tasks))

    spk_filter = getattr(args, "speaker", None)
    if spk_filter:
        want = set(spk_filter)
        tasks = [t for t in tasks if speaker_of(cfg, stem_of(t)) in want]

    shard = getattr(args, "shard", None)
    if shard:
        i, n = int(shard[0]), int(shard[1])
        if n > 1:
            if not (1 <= i <= n):
                raise SystemExit("--shard 的 i 必须在 1..n")
            tasks = [t for k, t in enumerate(tasks) if k % n == (i - 1)]

    if cfg["run"]["group_by_speaker"]:
        tasks.sort(key=lambda p: (speaker_of(cfg, stem_of(p)), p))

    limit = int(getattr(args, "limit", 0) or cfg["run"]["limit"] or 0)
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def out_path_of(cfg, src):
    rel = os.path.relpath(src, cfg["input_root"])
    return os.path.join(cfg["output_root"], rel)


# --------------------------------------------------------------------------- #
# 统计 / dry-run
# --------------------------------------------------------------------------- #
def analyze_tasks(cfg, index, tasks, verbose=True):
    st = collections.Counter()
    per_part = collections.defaultdict(collections.Counter)
    spk_tts = collections.Counter()
    spk_copy = collections.Counter()
    spk_count = collections.Counter()
    copy_reasons = collections.Counter()
    need_speakers = set()
    for src in tasks:
        rel = os.path.relpath(src, cfg["input_root"])
        part = rel.split(os.sep)[0]
        spk = speaker_of(cfg, stem_of(src))
        spk_count[spk] += 1
        if not src.lower().endswith(cfg["audio_ext"]):
            st["other"] += 1
            continue
        out = out_path_of(cfg, src)
        if os.path.exists(out) and os.path.getsize(out) >= cfg["encode"]["min_output_bytes"]:
            st["done"] += 1
            per_part[part]["done"] += 1
            continue
        mode, _cn, _en, why = decide(cfg, index, src)
        if mode == "tts":
            st["tts"] += 1
            per_part[part]["tts"] += 1
            spk_tts[spk] += 1
            need_speakers.add(spk)
        else:
            st["copy"] += 1
            per_part[part]["copy"] += 1
            spk_copy[spk] += 1
            copy_reasons[why] += 1
    if verbose:
        LOG.info("=" * 72)
        LOG.info("文件总数 %d | 需要合成 %d | 复制英文 %d | 已完成(跳过) %d | 其他 %d",
                 len(tasks), st["tts"], st["copy"], st["done"], st["other"])
        LOG.info("复制原因: %s", dict(copy_reasons))
        LOG.info("涉及说话人 %d 个", len(spk_count))
        LOG.info("%-16s %8s %8s %8s", "part", "total", "tts", "copy")
        for part in sorted(per_part):
            c = per_part[part]
            LOG.info("%-16s %8d %8d %8d", part, c["tts"] + c["copy"] + c["done"],
                     c["tts"], c["copy"])
        LOG.info("合成量最大的说话人:")
        for spk, n in spk_tts.most_common(15):
            LOG.info("   %-12s tts=%-7d copy=%-6d (文件共 %d)", spk, n,
                     spk_copy.get(spk, 0), spk_count[spk])
        smap = load_speaker_map(cfg, create=False)
        if str((cfg.get("emotion") or {}).get("spk_source", "line")).lower() == "map":
            missing = sorted(s for s in need_speakers if not resolve_ref(cfg, smap, s)[0])
            if missing:
                LOG.warning("以下 %d 个说话人还没有参考音频(会被复制英文): %s",
                            len(missing), ", ".join(missing))
        else:
            LOG.info("音色/情绪取自各条英文原音(spk_source=line)，无需 speaker_map")
        LOG.info("=" * 72)
    return st


# --------------------------------------------------------------------------- #
# 子命令: check
# --------------------------------------------------------------------------- #
DOWNLOAD_CHECKLIST = """
需要手动下载的东西（脚本不会自动下载）:
  1) IndexTTS 代码仓库 → {code_root}
     git clone https://github.com/index-tts/index-tts.git {code_root}
     （如果有 IndexTTS2.5 的官方仓库，就 clone 它的版本；本脚本只依赖 import 的模块名，
       不匹配时改 config 里 indextts.class_candidates 第一位即可）
  2) 模型权重 → {model_dir}
     ModelScope: IndexTeam/IndexTTS-2  (或你手上 IndexTTS2.5 对应的仓库)
     典型的 checkpoints 内容:
        config.yaml
        gpt.pth
        s2mel.pth
        bpe.model
        feat1.pt
        feat2.pt
        wav2vec2bert_stats.pt
        qwen0.6bemo4-merge/   (目录: config.json / model.safetensors / tokenizer...)
     缺哪个补哪个；以官方 README 的清单为准。
  3) 如果运行时报缺 w2v-bert / semantic codec 之类的 HF 模型:
     把它下载到 {code_root}/w2v-bert-2.0 (或按报错提示的路径)，
     或者先联网跑一次让它自动缓存；之后可设 HF_HUB_OFFLINE=1。
"""


def cmd_check(cfg, args):
    print("=" * 72)
    print("Python :", sys.version.split()[0], sys.executable)
    for prog in (cfg["encode"]["ffmpeg"], cfg["encode"]["ffprobe"]):
        rc, out, _ = run_cmd([prog, "-version"], 30)
        print("%-8s: %s" % (prog, out.splitlines()[0] if rc == 0 and out else "!! 找不到"))
    try:
        import soundfile as sf
        print("soundfile:", sf.__libsndfile_version__,
              "OGG 子类型:", sf.available_subtypes("OGG"))
    except Exception as e:                                  # noqa: BLE001
        print("soundfile: !! 不可用 ->", e)
    try:
        import numpy
        print("numpy  :", numpy.__version__)
    except Exception as e:                                  # noqa: BLE001
        print("numpy  : !! 不可用 ->", e)
    try:
        import torch
        print("torch  :", torch.__version__, "cuda:", torch.cuda.is_available(),
              "设备数:", torch.cuda.device_count())
    except Exception as e:                                  # noqa: BLE001
        print("torch  : !! 不可用 ->", e)

    icfg = cfg["indextts"]
    print("-" * 72)
    print("code_root:", icfg["code_root"],
          "OK" if os.path.isdir(icfg["code_root"]) else "!! 不存在")
    print("cfg_path :", icfg["cfg_path"],
          "OK" if os.path.isfile(icfg["cfg_path"]) else "!! 不存在")
    print("model_dir:", icfg["model_dir"],
          "OK" if os.path.isdir(icfg["model_dir"]) else "!! 不存在")
    missing = []
    for f in icfg.get("required_files") or []:
        p = os.path.join(icfg["model_dir"], f)
        if not os.path.exists(p):
            missing.append(f)
    for d in icfg.get("required_dirs") or []:
        p = os.path.join(icfg["model_dir"], d)
        if not os.path.isdir(p):
            missing.append(d + "/")
    if missing:
        print("!! 缺失模型文件:", ", ".join(missing))
        print(DOWNLOAD_CHECKLIST.format(code_root=icfg["code_root"],
                                        model_dir=icfg["model_dir"]))
    else:
        print("模型文件: 齐全")
    try:
        for mod_name, cls_name in icfg["class_candidates"]:
            try:
                mod = importlib.import_module(mod_name)
                print("import %-22s OK  %s=%s" % (mod_name, cls_name,
                      hasattr(mod, cls_name)))
                break
            except Exception as e:                          # noqa: BLE001
                print("import %-22s 失败 %s" % (mod_name, e))
    except Exception as e:                                  # noqa: BLE001
        print("模块探测异常:", e)

    print("-" * 72)
    print("xml      :", cfg["xml_path"],
          "OK" if os.path.isfile(cfg["xml_path"]) else "!! 不存在")
    print("input    :", cfg["input_root"],
          "OK" if os.path.isdir(cfg["input_root"]) else "!! 不存在")
    for part in cfg["parts"]:
        p = os.path.join(cfg["input_root"], part)
        print("   %-16s %s" % (part, "OK" if os.path.isdir(p) else "!! 不存在"))
    print("output   :", cfg["output_root"])
    print("refs     :", cfg["ref_dir"],
          "OK" if os.path.isdir(cfg["ref_dir"]) else "(还没生成)")
    print("map      :", cfg["speaker_map_path"],
          "OK" if os.path.isfile(cfg["speaker_map_path"]) else "(还没生成)")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# 子命令: stats
# --------------------------------------------------------------------------- #
def cmd_stats(cfg, args):
    index = load_dialog_index(cfg, rebuild=args.rebuild_index)
    tasks = collect_inputs(cfg, args, index)
    LOG.info("待处理文件 %d 个", len(tasks))
    analyze_tasks(cfg, index, tasks)


# --------------------------------------------------------------------------- #
# 子命令: map （生成 speaker_map.json + refs/*.wav）
# --------------------------------------------------------------------------- #
def cmd_map(cfg, args):
    index = load_dialog_index(cfg, rebuild=args.rebuild_index)
    refc = cfg["ref"]
    smap = load_speaker_map(cfg)
    os.makedirs(cfg["ref_dir"], exist_ok=True)

    counts = collections.Counter()
    cands = collections.defaultdict(list)
    parts = args.part or cfg["parts"]
    for part in parts:
        root = os.path.join(cfg["input_root"], part)
        if not os.path.isdir(root):
            LOG.error("目录不存在，跳过: %s", root)
            continue
        for dirpath, _d, fns in os.walk(root):
            for fn in fns:
                if not fn.lower().endswith(cfg["audio_ext"]):
                    continue
                stem = os.path.splitext(fn)[0]
                spk = speaker_of(cfg, stem)
                counts[spk] += 1
                row = match_row(cfg, index, stem)
                if not row:
                    continue
                en = clean_text(row[0])
                if not (int(refc["min_text_chars"]) <= len(en) <= int(refc["max_text_chars"])):
                    continue
                cands[spk].append((os.path.join(dirpath, fn), en))

    LOG.info("扫描到 %d 个说话人，其中 %d 个有可用参考候选",
             len(counts), len(cands))
    target_bytes = float(refc["target_bytes_per_sec"]) * (
        float(refc["min_sec"]) + float(refc["max_sec"])) / 2.0
    no_ref = []
    for spk in sorted(counts):
        entry = smap["speakers"].get(spk) or {}
        if entry.get("locked") and entry.get("ref_audio"):
            entry["files"] = counts[spk]
            smap["speakers"][spk] = entry
            LOG.info("[locked] %s 保留手工配置", spk)
            continue
        lst = cands.get(spk) or []
        chosen = None
        # 先按"平静度"筛出最好的一档，再在其中挑时长最接近理想值的
        ranked = lst
        if refc.get("prefer_neutral", True) and lst:
            scored = [(neutral_score(p, en), p, en) for p, en in lst]
            best = max(s for s, _p, _e in scored)
            if best >= float(refc.get("neutral_min_score", 1.0)):
                ranked = [(p, en) for s, p, en in scored if s >= best - 1.0]
                LOG.info("   %s: 平静档=%d/%d 条候选 (最高分 %.1f)", spk, len(ranked),
                         len(lst), best)
        ranked = sorted(ranked, key=lambda t: abs(os.path.getsize(t[0]) - target_bytes))
        for p, en in ranked[:int(refc["probe_candidates"])]:
            d = probe_duration(cfg, p)
            if d and float(refc["min_sec"]) <= d <= float(refc["max_sec"]):
                chosen = (p, en, d)
                break
        if chosen is None and ranked:
            p, en = ranked[0]
            d = probe_duration(cfg, p)
            chosen = (p, en, d)
        entry = dict(entry)
        entry["files"] = counts[spk]
        if chosen is None:
            entry.setdefault("ref_audio", None)
            entry.setdefault("ref_text", None)
            entry["note"] = "该说话人没有找到可用英文文本(可能是拟声/喘息等)，默认复制英文"
            no_ref.append(spk)
        else:
            src, en, dur = chosen
            ref_wav = os.path.join(cfg["ref_dir"], sanitize_name(spk) + ".wav")
            rc, o, e = run_cmd([cfg["encode"]["ffmpeg"], "-hide_banner", "-v", "error",
                                "-y", "-i", src, "-ac", "1",
                                "-ar", str(refc["wav_sr"]), "-c:a", "pcm_s16le",
                                ref_wav], 300)
            if rc != 0 or not os.path.exists(ref_wav):
                LOG.error("生成参考音频失败 %s: %s", spk, (e or o)[-200:])
                entry.setdefault("ref_audio", None)
                no_ref.append(spk)
            else:
                entry["ref_audio"] = ref_wav
                entry["ref_text"] = en
                entry["src_ogg"] = src
                entry["duration"] = round(dur, 3) if dur else None
                entry["neutral_score"] = neutral_score(src, en)
                entry.pop("note", None)
                LOG.info("[ref] %-12s %5.2fs  平静分%+.1f  %s | %s", spk, dur or -1,
                         entry["neutral_score"], en[:60], os.path.basename(src))
        entry.setdefault("locked", False)
        entry.setdefault("use", None)
        smap["speakers"][spk] = entry

    save_speaker_map(cfg, smap)
    LOG.info("完成: %d 个说话人，%d 个没有参考音频 -> %s",
             len(smap["speakers"]), len(no_ref), ", ".join(no_ref) or "-")
    LOG.info("人工修正: 手改 speaker_map.json 里的 use / ref_audio，"
             "或运行 python kcd_indextts.py set-speaker --speaker XXX --src-ogg /path/x.ogg --lock")


def cmd_set_speaker(cfg, args):
    index = load_dialog_index(cfg, rebuild=False)
    smap = load_speaker_map(cfg)
    spk = args.speaker
    entry = dict(smap["speakers"].get(spk) or {})
    entry.setdefault("files", 0)
    if args.src_ogg:
        src = os.path.abspath(os.path.expanduser(args.src_ogg))
        if not os.path.exists(src):
            raise SystemExit("参考音频不存在: %s" % src)
        os.makedirs(cfg["ref_dir"], exist_ok=True)
        ref_wav = os.path.join(cfg["ref_dir"], sanitize_name(spk) + ".wav")
        rc, o, e = run_cmd([cfg["encode"]["ffmpeg"], "-hide_banner", "-v", "error", "-y",
                            "-i", src, "-ac", "1", "-ar", str(cfg["ref"]["wav_sr"]),
                            "-c:a", "pcm_s16le", ref_wav], 300)
        if rc != 0:
            raise SystemExit("转 wav 失败: %s" % (e or o)[-300:])
        entry["ref_audio"] = ref_wav
        entry["src_ogg"] = src
        entry["duration"] = probe_duration(cfg, src)
        if args.ref_text:
            entry["ref_text"] = args.ref_text
        elif not entry.get("ref_text"):
            row = match_row(cfg, index, stem_of(src))
            entry["ref_text"] = clean_text(row[0]) if row else ""
    if args.use:
        entry["use"] = args.use
    if args.lock:
        entry["locked"] = True
    if args.unlock:
        entry["locked"] = False
    smap["speakers"][spk] = entry
    save_speaker_map(cfg, smap)
    LOG.info("已更新说话人 %s: %s", spk, json.dumps(entry, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# 子命令: run
# --------------------------------------------------------------------------- #
def cleanup_stale_tmp(cfg):
    tmp_root = os.path.join(cfg["work_dir"], "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    limit = time.time() - float(cfg["run"]["cleanup_tmp_older_than_hours"]) * 3600
    n = 0
    for fn in os.listdir(tmp_root):
        p = os.path.join(tmp_root, fn)
        try:
            if os.path.getmtime(p) < limit:
                os.remove(p)
                n += 1
        except OSError:
            pass
    if n:
        LOG.info("清掉 %d 个过期临时文件", n)
    return tmp_root


def cmd_run(cfg, args):
    if args.device:
        cfg["indextts"]["device"] = args.device
    index = load_dialog_index(cfg, rebuild=args.rebuild_index)
    tasks = collect_inputs(cfg, args, index)
    LOG.info("任务文件 %d 个 (part=%s, subdir=%s, limit=%s, shard=%s, speaker=%s)",
             len(tasks), args.part or cfg["parts"], args.subdir, args.limit,
             args.shard, args.speaker)
    if not tasks:
        return
    if args.dry_run:
        analyze_tasks(cfg, index, tasks)
        LOG.info("dry-run 结束，没有加载模型、没有写任何输出")
        return

    smap = load_speaker_map(cfg)
    ep = cfg.get("emotion") or {}
    if str(ep.get("spk_source", "line")).lower() == "line":
        LOG.info("音色+情绪 = 各条英文原音 1:1 迁移 (spk_source=line, emo_source=%s, "
                 "emo_alpha=%s, 短句<%.1fs 才用 speaker_map 兜底)",
                 ep.get("emo_source"), ep.get("emo_alpha"),
                 float(ep.get("line_min_sec", 0.8)))
    else:
        LOG.info("音色 = speaker_map 固定参考音, 情绪 = %s",
                 "显式向量 %s" % ep.get("emo_vector") if ep.get("emo_vector") is not None
                 else ep.get("emo_source"))
    tmp_root = cleanup_stale_tmp(cfg)
    pid = os.getpid()
    enc_min = int(cfg["encode"]["min_output_bytes"])
    force = bool(args.force)
    keep_tmp = bool(args.keep_temp_wav or cfg["run"]["keep_temp_wav"])
    copy_on_fail = bool(cfg["run"]["copy_on_tts_failure"])
    max_fail = int(cfg["run"]["max_consecutive_failures"])
    every = int(cfg["run"]["progress_every"])
    empty_every = int(cfg["run"]["empty_cache_every"])

    engine = None
    stats = collections.Counter()
    spk_ref_cache = {}
    failed = []
    t_start = time.time()
    consecutive = 0

    for i, src in enumerate(tasks, 1):
        rel = os.path.relpath(src, cfg["input_root"])
        out = out_path_of(cfg, src)
        spk = speaker_of(cfg, stem_of(src))
        if not src.lower().endswith(cfg["audio_ext"]):
            stats["other"] += 1
            if cfg["copy_other_files"] and not os.path.exists(out):
                try:
                    shutil.copy2(src, out)
                except Exception as e:                      # noqa: BLE001
                    LOG.error("[复制非音频失败] %s -> %r", rel, e)
            continue

        # 1) 断点续跑
        if not force and os.path.exists(out) and os.path.getsize(out) >= enc_min:
            stats["skip"] += 1
            continue

        mode, cn, en, why = decide(cfg, index, src)
        os.makedirs(os.path.dirname(out), exist_ok=True)

        # 2) 没有译文 / 译文没中文 -> 复制英文
        if mode == "copy":
            try:
                shutil.copy2(src, out)
                stats["copy_" + why] += 1
                LOG.info("[copy:%s] %s", why, rel)
            except Exception as e:                          # noqa: BLE001
                stats["fail"] += 1
                failed.append(src)
                LOG.error("[copy 失败] %s -> %r", rel, e)
            continue

        # 3) 参考音频 / 情绪参考
        emo_cfg = cfg.get("emotion") or {}
        spk_kind = str(emo_cfg.get("spk_source", "line")).lower()
        emo_kind = str(emo_cfg.get("emo_source", "line")).lower()
        if spk not in spk_ref_cache:
            spk_ref_cache[spk] = resolve_ref(cfg, smap, spk)
            if not spk_ref_cache[spk][0] and spk_kind == "map":
                LOG.warning("说话人 %s 没有参考音频，将复制英文（可先跑 map 或 set-speaker）", spk)
        mapped_ref, mapped_text = spk_ref_cache[spk]

        # 用"该条英文原音"当音色+情绪参考(1:1)——实测最贴近原声；
        # 太短的句子(说话人嵌入不稳)或没有英文原音时退回 map 里的固定参考音。
        src_dur = probe_audio_info(cfg, src)[0] or 0.0
        use_line = src_dur >= float(emo_cfg.get("line_min_sec", 0.8))
        emo_audio = None
        if emo_cfg.get("emo_vector") is None and emo_kind == "line" and use_line:
            emo_audio = src
        if spk_kind == "line" and use_line:
            ref_audio, ref_text = src, (en or mapped_text)
        else:
            ref_audio, ref_text = mapped_ref, mapped_text
            if not ref_audio:
                if spk_kind == "line":
                    ref_audio, ref_text = src, (en or "")   # 没跑 map 也能用
                else:
                    try:
                        shutil.copy2(src, out)
                        stats["copy_no_ref"] += 1
                    except Exception as e:                  # noqa: BLE001
                        stats["fail"] += 1
                        failed.append(src)
                        LOG.error("[copy 失败] %s -> %r", rel, e)
                    continue

        # 4) 合成
        tmp_files = []
        try:
            if engine is None:
                eng = IndexTTSEngine(cfg)
                try:
                    eng.load()
                except FatalError:
                    LOG.error("模型加载失败，终止本次运行（重试其它文件没有意义）")
                    raise
                engine = eng
            segments = split_text(cfg, cn)
            target, _t_sr = probe_audio_info(cfg, src)
            tag = "%d_%d" % (pid, i)

            # 第一遍: 按模型自然语速生成，然后实测
            merged = synth_segments(cfg, engine, segments, ref_audio, ref_text,
                                    tmp_root, tag, None, tmp_files, emo_audio)
            dfactor = None
            if cfg["align"].get("use_native_speed"):
                dfactor, why = native_factor_needed(cfg, probe_audio_info(cfg, merged)[0],
                                                    target)
                if dfactor:
                    LOG.info("二次生成(%s): %s", why, rel)
                    merged = synth_segments(cfg, engine, segments, ref_audio, ref_text,
                                            tmp_root, tag + "_p2", dfactor, tmp_files,
                                            emo_audio)

            ok, info = align_and_encode(cfg, merged, out, target)
            if not ok:
                raise RuntimeError("编码失败: %s" % info)
            if not os.path.exists(out) or os.path.getsize(out) < enc_min:
                raise RuntimeError("输出文件异常")
            stats["ok"] += 1
            consecutive = 0
            LOG.info("[ok] %-70s 时长 %.3f->%.3f d_factor=%s tempo=%.3f pad=%.3f "
                     "over=%+.3f enc=%s | %s",
                     rel, info.get("gen") or -1, target or -1,
                     "-" if not dfactor else "%.3f" % dfactor,
                     info.get("tempo", 1.0), info.get("pad", 0.0),
                     info.get("over", 0.0), info.get("encoder", ""),
                     (cn[:30] + ("…" if len(cn) > 30 else "")))
        except FatalError:
            raise
        except KeyboardInterrupt:
            LOG.warning("收到 Ctrl-C，正在退出（已完成的文件不会重做）")
            raise
        except Exception as e:                              # noqa: BLE001
            stats["fail"] += 1
            failed.append(src)
            consecutive += 1
            LOG.error("[fail] %s -> %r", rel, e, exc_info=True)
            if copy_on_fail:
                try:
                    if os.path.exists(out):
                        os.remove(out)
                    shutil.copy2(src, out)
                    stats["copy_on_fail"] += 1
                except Exception as e2:                     # noqa: BLE001
                    LOG.error("退回复制也失败: %r", e2)
        finally:
            if not keep_tmp:
                for p in tmp_files:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass

        if every and i % every == 0:
            el = time.time() - t_start
            rate = i / el if el > 0 else 0
            LOG.info("---- 进度 %d/%d (%.1f%%) 成功 %d 跳过 %d 复制 %d 失败 %d | "
                     "%.2f 文件/s | 已用 %.1f 分钟 | 预计剩余 %.1f 分钟",
                     i, len(tasks), 100.0 * i / len(tasks), stats["ok"], stats["skip"],
                     stats["copy_no_xml"] + stats["copy_empty_zh"] + stats["copy_no_cjk"]
                     + stats["copy_no_ref"] + stats["copy_on_fail"], stats["fail"],
                     rate, el / 60.0, (len(tasks) - i) / rate / 60.0 if rate > 0 else -1)
        if empty_every and i % empty_every == 0:
            empty_cuda_cache()
        if max_fail and consecutive >= max_fail:
            LOG.error("连续失败 %d 次，提前停止（多半是模型/环境问题，先看日志再重跑）", consecutive)
            break

    el = time.time() - t_start
    LOG.info("=" * 72)
    LOG.info("完成: 成功 %d | 跳过(已完成) %d | 复制英文 %d | 失败 %d | 用时 %.1f 分钟",
             stats["ok"], stats["skip"],
             stats["copy_no_xml"] + stats["copy_empty_zh"] + stats["copy_no_cjk"]
             + stats["copy_no_ref"] + stats["copy_on_fail"], stats["fail"], el / 60.0)
    LOG.info("明细: %s", dict(stats))
    if failed:
        fp = os.path.join(cfg["log_dir"], "failed_%s_%d.txt" % (
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
        with open(fp, "w", encoding="utf-8") as f:
            for p in failed:
                f.write(p + "\n")
        LOG.warning("失败清单已写入 %s，重跑: python kcd_indextts.py run --files-from %s", fp, fp)
    LOG.info("=" * 72)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="KCD2 中文配音流水线 (IndexTTS 2.x)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--config", default="kcd_indextts_config.json",
                   help="配置文件路径 (默认 ./kcd_indextts_config.json)")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("check", help="环境和模型自检")

    for name, help_text in (("stats", "只统计不生成"), ("run", "正式生成"),
                            ("map", "生成 speaker_map.json")):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--part", action="append", help="只处理某个 part，可重复")
        s.add_argument("--subdir", help="part 内的子目录(相对 part，如 dialog/open_world)；"
                                        "也可写以 part 开头的完整相对路径(如 IPL_english/dialog)")
        s.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件")
        s.add_argument("--rebuild-index", action="store_true", help="强制重建 xml 索引缓存")
        if name in ("stats", "run"):
            s.add_argument("--speaker", action="append", help="只处理指定说话人，可重复")
            s.add_argument("--files-from", help="从文本清单读文件(每行一个，支持绝对/相对 input_root)")
            s.add_argument("--shard", nargs=2, metavar=("I", "N"), help="分片: 第 I 片 / 共 N 片(双卡分别跑)")
        if name == "run":
            s.add_argument("--device", help="cuda:0 / cuda:1 ...")
            s.add_argument("--force", action="store_true", help="已有输出也重做")
            s.add_argument("--keep-temp-wav", action="store_true", help="保留中间 wav 便于排查")
            s.add_argument("--dry-run", action="store_true", help="只统计，不加载模型、不写文件")

    s = sub.add_parser("set-speaker", help="手动修正某个说话人的参考音频")
    s.add_argument("--speaker", required=True)
    s.add_argument("--src-ogg", help="参考音频(英文 ogg/wav)")
    s.add_argument("--ref-text", help="参考音频对应的英文文本(可选)")
    s.add_argument("--use", help="改用另一个说话人的音色")
    s.add_argument("--lock", action="store_true", help="锁定，不被 map 覆盖")
    s.add_argument("--unlock", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 0
    cfg = load_config(args.config)
    logfile = setup_logging(cfg["log_dir"], args.cmd)
    LOG.info("配置: %s | 日志: %s", os.path.abspath(args.config) if os.path.exists(args.config)
             else "(未找到，使用内置默认值)", logfile)
    if args.cmd != "check":
        for d in (cfg["output_root"], cfg["work_dir"], cfg["ref_dir"]):
            os.makedirs(d, exist_ok=True)

    handlers = {
        "check": cmd_check,
        "stats": cmd_stats,
        "map": cmd_map,
        "set-speaker": cmd_set_speaker,
        "run": cmd_run,
    }
    try:
        handlers[args.cmd](cfg, args)
    except FatalError as e:
        LOG.error("%s", e)
        return 2
    except KeyboardInterrupt:
        LOG.warning("被用户中断。已生成的输出会被保留，直接重跑即可续跑。")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
