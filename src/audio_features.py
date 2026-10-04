"""Audio analysis shared by the renderers: beat tracking and per-frame envelopes.

    python3 src/audio_features.py audio/made-of-weights-suno-analog.flac out/x.features.npz
"""

import subprocess
import sys

import numpy as np

SR = 22050


def decode(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def spectrogram(x, hop, win=2048):
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (win // 2, win // 2)), win)[::hop]
    n = len(x) // hop
    return np.abs(np.fft.rfft(frames[:n] * np.hanning(win), axis=1)), np.fft.rfftfreq(win, 1 / SR)


def estimate_tempo(onset, fps, lo=70, hi=180):
    t = np.arange(len(onset)) / fps
    cands = np.arange(lo, hi, 0.05)
    scores = [abs((onset * np.exp(2j * np.pi * t * b / 60)).sum()) for b in cands]
    return float(cands[int(np.argmax(scores))])


def track_beats(onset, fps, bpm, tightness=100.0):
    """Dynamic-programming beat tracker (Ellis 2007)."""
    period = 60.0 * fps / bpm
    o = onset / (onset.std() + 1e-9)
    n = len(o)
    score = o.copy()
    back = -np.ones(n, int)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    lags = np.arange(lo, hi + 1)
    penalty = -tightness * np.log(lags / period) ** 2
    for i in range(hi, n):
        cand = score[i - lags] + penalty
        k = int(np.argmax(cand))
        score[i] = o[i] + cand[k]
        back[i] = i - lags[k]
    # start from the best-scoring frame in the last period
    i = n - hi + int(np.argmax(score[n - hi:]))
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
        if i < 0 or (beats and beats[-1] - i < lo):
            break
    return np.array(beats[::-1]) / fps


def transient(e, decay):
    d = np.maximum(0, np.diff(e, prepend=e[0]))
    d = d / (np.percentile(d, 99.5) + 1e-9)
    out = np.zeros_like(d)
    for i in range(len(d)):
        out[i] = max(d[i], (out[i - 1] if i else 0) * decay)
    return np.clip(out, 0, 1.2)


def analyze(path, out_path, video_fps=30):
    x = decode(path)

    # onset envelope for beat tracking
    hop = 256
    fps = SR / hop
    X, f = spectrogram(x, hop)
    L = np.log1p(X * 10)
    onset = np.r_[0, np.maximum(0, np.diff(L, axis=0)).sum(1)]
    onset = np.maximum(onset - np.convolve(onset, np.ones(32) / 32, "same"), 0)
    bpm = estimate_tempo(onset, fps)
    beats = track_beats(onset, fps, bpm)

    # downbeat phase: the beat phase (mod 4) with the most low-end energy
    low_full = X[:, (f >= 30) & (f < 150)].sum(1)
    idx = np.clip((beats * fps).astype(int), 0, len(low_full) - 1)
    phase = int(np.argmax([low_full[idx[p::4]].mean() for p in range(4)]))
    downbeats = beats[phase::4]

    # per video frame envelopes
    vhop = SR // video_fps
    Xv, fv = spectrogram(x, vhop)

    def band(lo, hi):
        e = Xv[:, (fv >= lo) & (fv < hi)].sum(1)
        return e / (np.percentile(e, 99) + 1e-9)

    low, high = band(30, 150), band(5000, 11000)
    n = len(x) // vhop
    rms = np.sqrt((x[: n * vhop].reshape(n, vhop) ** 2).mean(1))
    rms = rms / (np.percentile(rms, 99) + 1e-9)
    np.savez(out_path, wave=x, sr=SR, bpm=bpm, beats=beats, downbeats=downbeats,
             low=low, high=high, rms=rms, kick=transient(low, 0.78), hat=transient(high, 0.6))
    return bpm, beats, downbeats


if __name__ == "__main__":
    bpm, beats, downbeats = analyze(sys.argv[1], sys.argv[2])
    ibi = np.diff(beats)
    print(f"bpm {bpm:.2f}  beats {len(beats)}  first {beats[0]:.3f}s  last {beats[-1]:.3f}s  "
          f"ibi mean {ibi.mean():.4f} sd {ibi.std():.4f}  downbeat0 {downbeats[0]:.3f}s")
