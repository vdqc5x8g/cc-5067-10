from __future__ import annotations

import math
import os
import struct
import wave
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Audio:
    samples: List[float]
    sample_rate: int
    channels: int = 1

    @property
    def n(self) -> int:
        return len(self.samples)


def _read_chunks(data: bytes):
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("不是合法的 WAV/RIFF 文件")
    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + 8
        end = start + size
        if end > len(data):
            raise ValueError(f"WAV 块 {chunk_id!r} 长度越界")
        if chunk_id == b"fmt ":
            fmt = data[start : start + min(size, 40)]
        elif chunk_id == b"data":
            pcm = data[start:end]
        pos = end + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("WAV 缺少 fmt 或 data 块")
    return fmt, pcm


def read_wav_bytes(data: bytes) -> Audio:
    fmt, pcm = _read_chunks(data)
    if len(fmt) < 16:
        raise ValueError("WAV fmt 块过短")
    audio_format, channels, sample_rate, _byte_rate, _block_align, bits = struct.unpack_from(
        "<HHIIHH", fmt, 0
    )
    if audio_format not in (1, 3):
        raise ValueError(f"仅支持 PCM/FLOAT WAV，格式码 {audio_format}")
    if channels < 1:
        raise ValueError("声道数非法")

    frames = len(pcm) // ((bits // 8) * channels)
    out: List[float] = []
    width = bits // 8
    for frame in range(frames):
        total = 0.0
        for ch in range(channels):
            off = (frame * channels + ch) * width
            raw = pcm[off : off + width]
            if audio_format == 3:
                if bits == 32:
                    v = struct.unpack("<f", raw)[0]
                elif bits == 64:
                    v = struct.unpack("<d", raw)[0]
                else:
                    raise ValueError(f"不支持的浮点 WAV 位深 {bits}")
            elif bits == 8:
                v = (raw[0] - 128) / 128.0
            elif bits == 16:
                v = struct.unpack("<h", raw)[0] / 32768.0
            elif bits == 24:
                i = raw[0] | (raw[1] << 8) | (raw[2] << 16)
                if i & 0x800000:
                    i -= 1 << 24
                v = i / 8388608.0
            elif bits == 32:
                v = struct.unpack("<i", raw)[0] / 2147483648.0
            else:
                raise ValueError(f"不支持的 PCM 位深 {bits}")
            total += v
        out.append(total / channels)
    return Audio(out, sample_rate, 1)


def read_wav(path: str) -> Audio:
    with open(path, "rb") as f:
        return read_wav_bytes(f.read())


def wav_bytes(samples: Sequence[float], sample_rate: int, subtype: str = "float32") -> bytes:
    if subtype == "float32":
        audio_format, bits = 3, 32
        raw = b"".join(struct.pack("<f", float(x)) for x in samples)
    elif subtype == "pcm16":
        audio_format, bits = 1, 16

        def conv(x: float) -> int:
            x = max(-1.0, min(1.0, float(x)))
            return int(round(x * 32767.0))

        raw = b"".join(struct.pack("<h", conv(x)) for x in samples)
    else:
        raise ValueError("subtype 必须是 float32 或 pcm16")

    channels = 1
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH", audio_format, channels, sample_rate, byte_rate, block_align, bits
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(raw))
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(raw)) + raw
    return header + fmt_chunk + data_chunk


def write_wav(path: str, samples: Sequence[float], sample_rate: int, subtype: str = "float32") -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(wav_bytes(samples, sample_rate, subtype))


def write_pcm16_preview(path: str, samples: Sequence[float], sample_rate: int, monitor_gain: float = 1.0) -> None:
    """Write a safely clamped audition copy. This is not the reconstruction master."""
    write_wav(path, [x * monitor_gain for x in samples], sample_rate, "pcm16")


def rms_db(samples: Sequence[float]) -> float:
    if not samples:
        return -math.inf
    e = sum(float(x) * float(x) for x in samples) / len(samples)
    if e <= 1e-20:
        return -120.0
    return 10.0 * math.log10(e)


def peak_abs(samples: Sequence[float]) -> float:
    return max((abs(float(x)) for x in samples), default=0.0)
