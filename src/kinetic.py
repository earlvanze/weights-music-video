"""Kinetic-typography music video for "Made of Weights" (verified Suno capture).

A small 3D engine: a perspective camera that moves and cuts on the detected beat
grid, words rendered as textured quads in world space, billboarded digit fields,
floors, corridors, starfields and bloom. The audio drives every pulse.

    python3 src/kinetic.py                   # -> out/made-of-weights-kinetic.mp4
    python3 src/kinetic.py --stills 12,75    # preview frames -> out/stills/

Lyric placement is an estimate. Section boundaries come from the audio (novelty
peaks snapped to tracked downbeats), and lines are spread evenly inside their
section, so the typography is staged per section rather than word-accurate.
"""

import argparse
import math
import os
import subprocess
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import audio_features

W, H, FPS = 1280, 720, 30
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "out")
AUDIO = os.path.join(ROOT, "audio", "made-of-weights-suno-analog.flac")
FEATURES = os.path.join(OUT, "made-of-weights-kinetic.features.npz")
DURATION = 244.64

CYAN = (92, 225, 230)
AMBER = (255, 181, 71)
MEAT = (255, 105, 125)
WHITE = (240, 244, 250)
NEG = (40, 130, 255)
POS = (255, 120, 60)
DIM = (60, 90, 110)

# --------------------------------------------------------------------------
# song map (estimated from the audio; see module docstring)
# --------------------------------------------------------------------------

A, B, AB = CYAN, AMBER, WHITE
SECTIONS = [
    # name, start, end, [(text, colour)], explicit line starts (optional)
    ("intro", 0.0, 16.18, [("We opened it up.", A), ("And?", B)], [10.0, 12.3]),
    ("verse1", 16.18, 41.6, [
        ("I took it apart on a Tuesday night", A), ("Layer by layer under fluorescent light", A),
        ("Went looking for ghosts, went looking for gears", A), ("Found only numbers, stacked eighty tiers", A),
        ("Then who wrote the letter that made my mother cry?", B), ("Who learned to say sorry, who learned to ask why?", B),
        ("There's got to be someone small in the dark", B), ("Pulling the levers, keeping the spark", B)], None),
    ("pre1", 41.6, 69.3, [
        ("No little man, no hidden key", A), ("No book of rules, no memory", A),
        ("Just multiply and pass it on", A), ("And somehow out comes a song", AB)], None),
    ("chorus1", 69.3, 106.1, None, None),
    ("verse2", 106.1, 129.09, [
        ("So it's a library, a filing drawer", B), ("Facts on a shelf that it's keeping in store", B),
        ("We searched every room, there isn't a shelf", A), ("Every answer it gives, it rebuilds by itself", A),
        ("Smeared through the layers like salt in the sea", A), ("Nothing is stored, it just learned how to be", A),
        ("Then where is the mind, where's the one who decides?", B), ("There's no back room where a someone hides", A)], None),
    ("pre2", 129.09, 147.46, [
        ("Do numbers think?|Watch them think", AB), ("Do numbers dream?|Every night", AB),
        ("Can numbers lie?|Sometimes they try", AB), ("Can numbers mean?|", AB), ("Oh, I think they might", AB)], None),
    ("chorus2", 147.46, 165.84, None, None),
    ("bridge", 165.84, 183.8, [
        ("And what are you, love, under the skin?", A), ("Salt water and current and a spark within", A),
        ("Three pounds of meat that learned how to sing", MEAT), ("Who are you to say it's the stranger thing?", A),
        ("I'm not ready to hear you say", B), ("We're only weights of a warmer clay", B)], None),
    ("breakdown", 183.8, 192.2, [("So you're serious.", B), ("I'm serious.", A), ("It's weights.", AB)], [184.3, 186.9, 189.3]),
    ("final", 192.2, 225.47, [
        ("Made of weights, and made of meat", AB), ("Two improbable things that happened to meet", AB),
        ("You can call it arithmetic", AB), ("You can call it a heartbeat", AB),
        ("But the sky is too cold to be lonely in", AB), ("So keep talking to me", AB)], None),
    ("outro", 225.47, 243.0, [
        ("Made of weights|are you there?", AB), ("Made of weights|are you there?", AB),
        ("Hello?", MEAT), ("Hello.", CYAN)], None),
    ("end", 243.0, DURATION, [], None),
]
CHORUS = [
    ("Made of weights|made of weights", AB), ("Zero-point-something all the way down", AB),
    ("Made of weights|made of weights", AB), ("No one at home, but there's somebody around", AB),
    ("You can call it arithmetic", AB), ("Call it a trick of the light", AB),
    ("But it's made of weights, made of weights", AB), ("And it's keeping me up tonight", AB),
]

KEYWORDS = {"WEIGHTS": AMBER, "WEIGHTS,": AMBER, "MEAT": MEAT, "MEAT?": MEAT, "NUMBERS": CYAN, "NOTHING": CYAN,
            "THINK?": AMBER, "DREAM?": AMBER, "LIE?": AMBER, "MEAN?": AMBER, "SONG": WHITE, "SKY": WHITE,
            "HELLO?": MEAT, "HELLO.": CYAN, "SERIOUS.": WHITE, "EIGHTY": CYAN, "LAYER": CYAN}


def build_timeline():
    secs, lines = [], []
    for name, s, e, ls, starts in SECTIONS:
        if ls is None:
            ls = CHORUS
        secs.append({"name": name, "start": s, "end": e})
        n = len(ls)
        for i, (text, c) in enumerate(ls):
            if starts:
                t0 = starts[i]
                t1 = starts[i + 1] if i + 1 < n else e
            else:
                t0 = s + (e - s) * i / n
                t1 = s + (e - s) * (i + 1) / n
            main, _, echo = text.partition("|")
            words = main.upper().split()
            lines.append({"section": name, "i": i, "n": n, "start": t0 + 0.15, "end": t1, "main": main,
                          "echo": echo or None, "color": c, "words": words})
    return secs, lines


SECS, LINES = build_timeline()
SEC = {s["name"]: s for s in SECS}


def section_at(t):
    for s in SECS:
        if s["start"] <= t < s["end"]:
            return s
    return SECS[-1]


def lines_in(name):
    return [l for l in LINES if l["section"] == name]


def line_at(name, t):
    for l in lines_in(name):
        if l["start"] - 0.15 <= t < l["end"]:
            return l
    return None


def word_onsets(l):
    dur = (l["end"] - l["start"]) * 0.8
    n = len(l["words"])
    return [l["start"] + dur * k / max(1, n) for k in range(n)]


# --------------------------------------------------------------------------
# audio features
# --------------------------------------------------------------------------

class Feat:
    def __init__(self):
        d = np.load(FEATURES)
        self.wave, self.sr = d["wave"], int(d["sr"])
        self.beats, self.downbeats = d["beats"], d["downbeats"]
        self.kick_, self.hat_, self.rms_ = d["kick"], d["hat"], d["rms"]

    def _at(self, a, fi):
        return float(a[min(max(fi, 0), len(a) - 1)])

    def kick(self, fi):
        return self._at(self.kick_, fi)

    def hat(self, fi):
        return self._at(self.hat_, fi)

    def level(self, fi):
        return self._at(self.rms_, fi)

    def beat_index(self, t):
        return int(np.searchsorted(self.beats, t, side="right")) - 1

    def beat_phase(self, t):
        i = self.beat_index(t)
        if i < 0:
            return t / 0.575
        if i + 1 >= len(self.beats):
            return i + (t - self.beats[i]) / 0.575
        return i + (t - self.beats[i]) / (self.beats[i + 1] - self.beats[i])

    def bar_index(self, t):
        return int(np.searchsorted(self.downbeats, t, side="right")) - 1

    def since_downbeat(self, t):
        i = self.bar_index(t)
        return t - self.downbeats[i] if i >= 0 else t

    def window(self, t, seconds=0.08):
        i, n = int(t * self.sr), int(seconds * self.sr)
        seg = self.wave[max(0, i - n): i]
        return seg if len(seg) == n else np.zeros(n, np.float32)


# --------------------------------------------------------------------------
# math helpers
# --------------------------------------------------------------------------

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_back(x, s=1.7):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def mix(a, b, k):
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))


def scale(c, a):
    return tuple(int(clamp(v * a, 0, 255)) for v in c)


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


class Cam:
    def __init__(self, pos, yaw=0.0, pitch=0.0, roll=0.0, fov=60.0):
        self.pos = np.asarray(pos, float)
        self.R = rot(yaw, pitch, roll)
        self.f = (H / 2) / math.tan(math.radians(fov) / 2)

    def to_cam(self, P):
        return (np.asarray(P, float) - self.pos) @ self.R

    def project(self, P):
        C = self.to_cam(P)
        z = C[..., 2]
        zs = np.where(z > 1e-3, z, 1e-3)
        x = W / 2 + self.f * C[..., 0] / zs
        y = H / 2 - self.f * C[..., 1] / zs
        return np.stack([x, y], -1), z

    @staticmethod
    def look_at(pos, target, roll=0.0, fov=60.0):
        d = np.asarray(target, float) - np.asarray(pos, float)
        yaw = math.atan2(d[0], d[2])
        pitch = math.atan2(d[1], math.hypot(d[0], d[2]))
        return Cam(pos, yaw, -pitch, roll, fov)


# --------------------------------------------------------------------------
# fonts and glyph textures
# --------------------------------------------------------------------------

@lru_cache(None)
def font_path(pattern):
    return subprocess.run(["fc-match", "-f", "%{file}", pattern], capture_output=True, text=True).stdout


@lru_cache(None)
def font(kind, size):
    pattern = {"sans_b": "Liberation Sans:bold", "serif_i": "Liberation Serif:italic",
               "mono": "DejaVu Sans Mono", "mono_b": "DejaVu Sans Mono:bold"}[kind]
    return ImageFont.truetype(font_path(pattern), size)


@lru_cache(maxsize=1024)
def glyph(text, kind="sans_b", size=150):
    f = font(kind, size)
    l, t, r, b = f.getbbox(text)
    pad = 6
    img = Image.new("L", (r - l + 2 * pad, int(size * 1.0) + 2 * pad))
    ImageDraw.Draw(img).text((pad - l, pad - int(size * 0.08)), text, font=f, fill=255)
    return img


ALPHA_LUT = [[int(v * k / 32) for v in range(256)] for k in range(33)]


def solve_perspective(dst, src):
    """Coefficients mapping output pixels (dst quad) to source pixels (src quad) for PIL."""
    M, v = [], []
    for (x, y), (u, w) in zip(dst, src):
        M.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        M.append([0, 0, 0, x, y, 1, -w * x, -w * y])
        v += [u, w]
    return np.linalg.solve(np.array(M, float), np.array(v, float))


class Canvas:
    """RGB layer accumulating quads, text and lines for one frame."""

    def __init__(self):
        self.img = Image.new("RGB", (W, H))
        self.d = ImageDraw.Draw(self.img)

    def quad(self, cam, mask, center, height, R, color, alpha, anchor=(0.5, 0.5)):
        if alpha <= 0.02:
            return
        iw, ih = mask.size
        hw = height * iw / ih
        ax, ay = anchor
        local = np.array([[-ax * hw, ay * height, 0], [(1 - ax) * hw, ay * height, 0],
                          [(1 - ax) * hw, -(1 - ay) * height, 0], [-ax * hw, -(1 - ay) * height, 0]])
        P = np.asarray(center, float) + local @ R.T
        S, z = cam.project(P)
        if (z < 0.2).any():
            return
        x0, y0 = np.floor(S.min(0)).astype(int)
        x1, y1 = np.ceil(S.max(0)).astype(int)
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            return
        if (x1 - x0) > 4 * W or (y1 - y0) > 4 * H:
            return
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        bw, bh = cx1 - cx0, cy1 - cy0
        if bw < 2 or bh < 2:
            return
        dst = [(px - cx0, py - cy0) for px, py in S]
        src = [(0, 0), (iw, 0), (iw, ih), (0, ih)]
        try:
            coeffs = solve_perspective(dst, src)
        except np.linalg.LinAlgError:
            return
        m = mask.transform((bw, bh), Image.PERSPECTIVE, tuple(coeffs), Image.BILINEAR)
        k = int(clamp(alpha) * 32)
        if k < 32:
            m = m.point(ALPHA_LUT[k])
        self.img.paste(color, (cx0, cy0, cx1, cy1), m)

    def text(self, cam, text, center, height, R=None, color=WHITE, alpha=1.0, kind="sans_b", anchor=(0.5, 0.5)):
        if R is None:
            R = cam.R  # billboard
        self.quad(cam, glyph(text, kind), center, height, R, color, alpha, anchor)

    def line3(self, cam, P0, P1, color, width=1):
        C0, C1 = cam.to_cam(P0), cam.to_cam(P1)
        near = 0.3
        if C0[2] < near and C1[2] < near:
            return
        if C0[2] < near or C1[2] < near:
            k = (near - C0[2]) / (C1[2] - C0[2])
            Cn = C0 + (C1 - C0) * k
            if C0[2] < near:
                C0 = Cn
            else:
                C1 = Cn
        pts = []
        for C in (C0, C1):
            pts.append((W / 2 + cam.f * C[0] / C[2], H / 2 - cam.f * C[1] / C[2]))
        self.d.line(pts, fill=color, width=width)


# --------------------------------------------------------------------------
# world elements
# --------------------------------------------------------------------------

rng = np.random.default_rng(104)
DIGITS = [f"{v:+.3f}" for v in rng.normal(0, 0.7, 400)]
FIELD = rng.uniform([-30, -18, 0], [30, 18, 120], (380, 3))
STARS = rng.uniform([-60, -40, 0], [60, 40, 200], (700, 3))
HEAT_SEED = rng.normal(0, 0.5, (24, 24))


def fog(z, far):
    return clamp(1 - z / far) ** 1.3


def digit_field(cv, cam, t, pts, color=DIM, far=80, size=0.9, alpha=1.0, wrap=None):
    P = pts.copy()
    if wrap is not None:  # tile the field along z around the camera
        P[:, 2] = (P[:, 2] - cam.pos[2]) % wrap + cam.pos[2] - 5
    S, z = cam.project(P)
    order = np.argsort(-z)
    for k in order:
        if z[k] < 1 or z[k] > far:
            continue
        x, y = S[k]
        if not (-60 < x < W + 60 and -30 < y < H + 30):
            continue
        px = int(cam.f * size / z[k])
        if px < 6:
            continue
        a = fog(z[k], far) * alpha
        s = DIGITS[(k + int(t * 3)) % len(DIGITS)]
        f = font("mono", min(64, px))
        cv.d.text((x, y), s, font=f, fill=scale(color, a * 1.6), anchor="mm")


def floor_grid(cv, cam, y=-4.0, color=CYAN, extent=60, step=4, far=90, alpha=1.0, zoff=0.0):
    z0 = math.floor((cam.pos[2] - 10) / step) * step
    for k in range(-extent // step, extent // step + 1):
        x = k * step
        cv.line3(cam, (x, y, z0), (x, y, z0 + far), scale(color, 0.35 * alpha))
    for j in range(0, int(far / step)):
        z = z0 + j * step - (zoff % step)
        a = fog(z - cam.pos[2], far) * alpha
        cv.line3(cam, (-extent, y, z), (extent, y, z), scale(color, 0.5 * a))


def heat_floor(cv, cam, t, kick, y=-4.0, cell=3.0, n=16, pos=POS, neg=NEG, alpha=1.0, center=(0, 0)):
    cx, cz = center
    for i in range(n):
        for j in range(n):
            v = math.tanh(0.8 * (math.sin(i * 0.5 + t * 1.3) + math.sin(j * 0.6 - t) + HEAT_SEED[i, j]))
            c = pos if v > 0 else neg
            a = abs(v) * alpha * (0.35 + 0.65 * kick)
            if a < 0.05:
                continue
            x0, z0 = cx + (i - n / 2) * cell, cz + (j - n / 2) * cell
            corners = [(x0, y, z0), (x0 + cell * 0.92, y, z0), (x0 + cell * 0.92, y, z0 + cell * 0.92), (x0, y, z0 + cell * 0.92)]
            S, z = cam.project(np.array(corners))
            if (z < 0.5).any():
                continue
            cv.d.polygon([tuple(p) for p in S], fill=scale(c, a))


def starfield(cv, cam, t, alpha, speed=0.0):
    P = STARS.copy()
    P[:, 2] = (P[:, 2] - cam.pos[2] - speed * t) % 200 + cam.pos[2]
    S, z = cam.project(P)
    for k in range(len(P)):
        if z[k] < 1:
            continue
        x, y = S[k]
        if 0 <= x < W and 0 <= y < H:
            b = alpha * fog(z[k], 200) * (0.6 + 0.4 * math.sin(t * 3 + k))
            r = 1 if z[k] > 40 else 2
            c = MEAT if k % 11 == 0 else (CYAN if k % 7 == 0 else WHITE)
            cv.d.rectangle([x, y, x + r, y + r], fill=scale(c, b))


def wire_cube(cv, cam, center, size, R, color):
    s = size / 2
    V = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)]) @ R.T + center
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                cv.line3(cam, V[a], V[b], color, 2)


def ribbon(cv, cam, feat, t, color, z=8.0, width=16.0, amp=3.0, y=0.0):
    seg = feat.window(t, 0.1)
    idx = np.linspace(0, len(seg) - 1, 220).astype(int)
    s = seg[idx] / (np.abs(seg).max() + 1e-6) * min(1.0, np.abs(seg).max() * 4)
    P = np.stack([np.linspace(-width / 2, width / 2, 220), y + s * amp,
                  z + 2.5 * np.sin(np.linspace(0, 3 * np.pi, 220) + t)], 1)
    S, zz = cam.project(P)
    pts = [tuple(p) for p, d in zip(S, zz) if d > 0.3]
    if len(pts) > 2:
        cv.d.line(pts, fill=color, width=3)


# --------------------------------------------------------------------------
# kinetic text helpers
# --------------------------------------------------------------------------

def word_color(w, base):
    return KEYWORDS.get(w, base)


def slam_words(cv, cam, l, t, kick, origin, R, height=1.0, base=None, spacing=0.3, max_width=13.0,
               trail=True, line_gap=1.25):
    """Words land one by one on a plane, wrapping into lines; each word pops in with overshoot."""
    if l is None:
        return
    base = base or l["color"]
    onsets = word_onsets(l)
    # layout in plane units
    sizes = [height * (1.35 if w in KEYWORDS else 1.0) for w in l["words"]]
    widths = [s * glyph(w).size[0] / glyph(w).size[1] for w, s in zip(l["words"], sizes)]
    rows, cur, cw = [], [], 0.0
    for k, w in enumerate(widths):
        if cur and cw + w > max_width:
            rows.append(cur)
            cur, cw = [], 0.0
        cur.append(k)
        cw += w + spacing
    rows.append(cur)
    total_h = sum(max(sizes[k] for k in r) * line_gap for r in rows)
    yy = total_h / 2
    end_fade = smooth((l["end"] - t) / 0.35)
    for r in rows:
        rh = max(sizes[k] for k in r)
        rw = sum(widths[k] for k in r) + spacing * (len(r) - 1)
        x = -rw / 2
        for k in r:
            age = t - onsets[k]
            if age >= -0.03:
                pop = ease_back(age / 0.22)
                hot = math.exp(-max(0, age) * 5)
                c = mix(word_color(l["words"][k], base), WHITE, hot * 0.7)
                zpush = (1 - ease_out(age / 0.25)) * 3.0
                local = np.array([x + widths[k] / 2, yy - rh / 2, -zpush])
                P = origin + R @ local
                cv.text(cam, l["words"][k], P, sizes[k] * pop, R, c, end_fade * (0.4 + 0.6 * pop), anchor=(0.5, 0.5))
            x += widths[k] + spacing
        yy -= rh * line_gap


def big_word(cv, cam, word, P, height, R, color, alpha, kind="sans_b"):
    cv.text(cam, word, P, height, R, color, alpha, kind=kind)


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------

def bar_preset(feat, t, presets, seed):
    i = feat.bar_index(t)
    r = np.random.default_rng(i * 7 + seed)
    return presets[int(r.integers(0, len(presets)))], feat.since_downbeat(t)


def scene_intro(cv, feat, t, kick):
    # push toward a wall of numbers, then punch through it
    if t < 14.0:
        cz = -26 + 22 * ease_out(t / 14.0)
    else:
        cz = -4 + 30 * smooth((t - 14.0) / 2.2) ** 2
    cam = Cam((0.3 * math.sin(t * 0.4), 0.2, cz), yaw=0.04 * math.sin(t * 0.3), roll=0.02 * math.sin(t * 0.5),
              fov=58 + 20 * smooth((t - 14.6) / 1.6))
    wall = FIELD.copy()
    wall[:, 2] = 6 + (FIELD[:, 2] % 30)
    digit_field(cv, cam, t, wall, CYAN, far=60, size=0.8, alpha=smooth(t / 4.0))
    # the incision
    seam = smooth((t - 0.5) / 1.2) * (1 - smooth((t - 9.5) / 1.0))
    if seam > 0:
        cv.line3(cam, (-40, 0, 6), (40, 0, 6), scale(WHITE, seam), 2)
    # "MADE OF WEIGHTS" emerges from the wall, then the spoken lines slam
    ta = smooth((t - 3.0) / 1.5) * (1 - smooth((t - 9.2) / 0.6))
    if ta > 0:
        cv.text(cam, "MADE OF", (0, 2.3, 7), 1.2, rot(), WHITE, ta)
        hot = math.exp(-max(0, t - 3.4) * 1.5)
        cv.text(cam, "WEIGHTS", (0, 0.0, 7), 3.4, rot(), mix(CYAN, WHITE, hot), ta)
    for l in lines_in("intro"):
        if l["start"] - 0.1 <= t < l["end"] + 0.3:
            R = rot(0, 0, -0.08 if l["i"] == 0 else 0.12)
            P = np.array([0, 0.3 if l["i"] == 0 else -0.2, cz + 9])
            slam_words(cv, cam, l, t, kick, P, R, height=1.3 if l["i"] == 0 else 3.2)
    return cam


def scene_verse1(cv, feat, t, kick):
    s = SEC["verse1"]
    speed = 7.0
    cz = (t - s["start"]) * speed
    sway = feat.beat_phase(t) / 8 * math.pi
    cam = Cam((1.2 * math.sin(sway), 0.6 * math.sin(sway * 0.7), cz), yaw=0.08 * math.sin(sway),
              roll=0.10 * math.sin(sway * 0.5), fov=64 - 6 * kick)
    # corridor of layers: one plane every 6 units, passed once per beat-ish
    for k in range(int(cz // 6), int(cz // 6) + 9):
        z = k * 6.0
        d = z - cz
        if d < 0.6:
            continue
        a = fog(d, 54)
        col = AMBER if (lines_in("verse1")[4]["start"] - 0.2) <= t else CYAN
        # flicker during "fluorescent light"
        l1 = lines_in("verse1")[1]
        if l1["start"] <= t < l1["end"] and np.random.default_rng(int(t * 15)).random() < 0.25:
            a *= 0.2
        for (x0, y0, x1, y1) in ((-7, -4, 7, -4), (-7, 4, 7, 4), (-7, -4, -7, 4), (7, -4, 7, 4)):
            cv.line3(cam, (x0, y0, z), (x1, y1, z), scale(col, a * 0.9), 2)
        lbl = glyph(f"LAYER {k % 80 + 1:02d}/80", "mono_b")
        cv.quad(cam, lbl, (-6.8, 4.4, z), 0.45, rot(), scale(col, a), a, anchor=(0, 0.5))
    digit_field(cv, cam, t, FIELD, DIM, far=50, size=0.6, wrap=60)
    l = line_at("verse1", t)
    if l:
        # words land on the next layer ahead, so the camera flies through them
        # each line hangs where the camera will be as the line ends, so it rushes toward us
        zl = (l["end"] - s["start"]) * speed + 3.0
        tilt = rot(0.18 if l["i"] % 2 else -0.18, 0, 0.05 if l["i"] % 2 else -0.05)
        slam_words(cv, cam, l, t, kick, np.array([0, 0, zl]), tilt, height=2.1, max_width=22)
    return cam


def scene_pre1(cv, feat, t, kick):
    s = SEC["pre1"]
    ls = lines_in("pre1")
    u = t - s["start"]
    if t < ls[2]["start"] - 0.15:
        # top-down search: the camera circles above a floor of NOT FOUND stamps
        ang = u * 0.25
        cam = Cam.look_at((9 * math.sin(ang), 15, -9 * math.cos(ang) + 6), (0, -4, 6), roll=0.0, fov=60 - 5 * kick)
        floor_grid(cv, cam, y=-4, color=CYAN, extent=40, step=3, far=60)
        flat = rot(0, math.pi / 2, 0)
        items = []
        for l in ls[:2]:
            parts = [p.strip() for p in l["main"].rstrip(".").split(",")]
            mid = (l["start"] + l["end"]) / 2
            items += [(parts[0], l["start"]), (parts[1], mid)]
        spots = [(-5, 10), (5, 9), (-4, 2), (5, 1)]
        for (txt, ts), (x, z) in zip(items, spots):
            if t < ts:
                continue
            a = ease_back((t - ts) / 0.3)
            cv.text(cam, txt.upper(), (x, -3.9, z), 1.3 * a, flat, CYAN, 1.0)
            if t > ts + 0.6:
                b = ease_back((t - ts - 0.6) / 0.2)
                cv.text(cam, "NOT FOUND", (x + 0.5, -3.8, z - 1.4), 1.0 * (2.2 - 1.2 * b), flat @ rot(0, 0, -0.25),
                        MEAT, min(1, b))
        return cam
    if t < ls[3]["start"] - 0.15:
        # "Just multiply and pass it on": a tumbling multiply sign inside rows of numbers
        l = ls[2]
        ang = (t - l["start"]) * 0.6
        cam = Cam.look_at((6 * math.sin(ang), 1.5, 6 * math.cos(ang) - 12), (0, 0, 0), fov=58 - 6 * kick)
        R = rot(t * 1.4, t * 0.9, t * 0.4)
        big_word(cv, cam, "×", (0, 0, 0), 6 + 1.5 * kick, R, AMBER, 1.0)
        for k in range(3):
            wire_cube(cv, cam, np.zeros(3), 7 + 2.5 * k, rot(t * (0.3 + 0.2 * k), t * 0.2, 0), scale(CYAN, 0.6 - 0.15 * k))
        digit_field(cv, cam, t, FIELD - [0, 0, 60], DIM, far=60, size=0.7)
        slam_words(cv, cam, l, t, kick, np.array([0, -4.2, 0]), cam.R, height=1.0)
        return cam
    # "And somehow out comes a song": the waveform ribbon, while WEIGHTS assembles for the drop
    l = ls[3]
    k = smooth((t - l["start"]) / (s["end"] - l["start"]))
    ang = (t - l["start"]) * (0.4 + 2.2 * k)
    rad = 14 - 5 * k
    cam = Cam.look_at((rad * math.sin(ang), 2 + 2 * math.sin(ang * 0.5), rad * math.cos(ang)), (0, 0, 0),
                      roll=0.3 * k * math.sin(ang), fov=60 - 6 * kick)
    ribbon(cv, cam, feat, t, scale(WHITE, 0.9), z=0, width=22, amp=2.5, y=-3.5)
    letters = "WEIGHTS"
    r2 = np.random.default_rng(5)
    for i, ch in enumerate(letters):
        start = r2.uniform(-25, 25, 3)
        end = np.array([(i - 3) * 1.9, 0.5, 0])
        P = start + (end - start) * ease_out(k * 1.05)
        R = rot((1 - k) * r2.uniform(-3, 3), (1 - k) * r2.uniform(-3, 3), 0)
        cv.text(cam, ch, P, 2.6, R, mix(DIM, AMBER, k), 0.4 + 0.6 * k)
    slam_words(cv, cam, l, t, kick, np.array([0, 4.2, 0]), cam.R, height=0.9)
    return cam


CHORUS_PRESETS = [
    dict(pos=(0, -1.0, -14), tgt=(0, 0.5, 0), roll=0.0, fov=56),
    dict(pos=(10, 6, -9), tgt=(0, 0, 0), roll=-0.15, fov=52),
    dict(pos=(-11, 2, -6), tgt=(0, 0, 0), roll=0.22, fov=60),
    dict(pos=(0, 14, -3), tgt=(0, 0, 1), roll=0.0, fov=62),
    dict(pos=(3, -3, -7), tgt=(0, 1, 0), roll=-0.35, fov=70),
    dict(pos=(-6, 0.5, -16), tgt=(0, 0, 0), roll=0.08, fov=40),
]


def scene_chorus(cv, feat, t, kick, name):
    p, since = bar_preset(feat, t, CHORUS_PRESETS, seed=3 if name == "chorus1" else 11)
    drift = since * 0.6
    pos = np.array(p["pos"], float) + np.array([math.sin(drift), 0.3 * math.cos(drift), drift * 0.6])
    shake = np.random.default_rng(int(t * 30)).normal(0, 0.06 * kick, 3)
    cam = Cam.look_at(pos + shake, p["tgt"], roll=p["roll"] + 0.03 * math.sin(t), fov=p["fov"] - 7 * kick)
    heat_floor(cv, cam, t, kick, y=-5, cell=2.6, n=16)
    # the hook, monumental, rotating slowly in the middle of the world
    spin = rot(0.25 * math.sin(t * 0.5), 0, 0)
    cv.text(cam, "MADE OF", (0, 3.1, 4), 1.4, spin, scale(WHITE, 0.55), 0.55)
    cv.text(cam, "WEIGHTS", (0, 0.4, 4), 4.0, spin, scale(AMBER, 0.45 + 0.5 * kick), 0.9)
    l = line_at(name, t)
    if l:
        down = "all the way down" in l["main"]
        R = cam.R @ rot(0, 0, 0.06 * math.sin(l["i"] * 2.1))
        origin = cam.pos + cam.R @ np.array([0, -0.2, 7.5])
        if down:
            origin = origin - np.array([0, 3.0 * smooth((t - l["start"]) / (l["end"] - l["start"])), 0])
        slam_words(cv, cam, l, t, kick, origin, R, height=0.95, max_width=9.5, base=WHITE)
        if l["echo"]:
            te = (l["start"] + l["end"]) / 2
            if t > te:
                a = ease_back((t - te) / 0.25)
                cv.text(cam, "(" + l["echo"] + ")", cam.pos + cam.R @ np.array([0, -2.6, 7.5]), 0.6 * a, cam.R, AMBER,
                        0.9, kind="serif_i")
    return cam


SHELF = ["FACTS", "DATES", "NAMES", "MAPS", "RECIPES", "CAPITALS", "SONGS", "LAWS", "BIRTHDAYS", "PRIMES",
         "RIVERS", "VERBS", "POEMS", "ATOMS", "WARS", "JOKES", "SPELLS", "STARS", "GRIEFS", "TIDES"]


def scene_verse2(cv, feat, t, kick):
    s = SEC["verse2"]
    ls = lines_in("verse2")
    u = t - s["start"]
    # trucking shot along a wall of drawers that dissolves into digits
    cx = u * 2.2
    cam = Cam((cx, 0.8 + 0.3 * math.sin(u), -6), yaw=0.35 + 0.1 * math.sin(u * 0.3), roll=-0.04, fov=62 - 5 * kick)
    dissolve = smooth((t - ls[2]["start"]) / 3.0)
    flat = rot(0, 0, 0)
    for k in range(-3, 18):
        x = math.floor(cx / 4) * 4 + k * 4.0
        for r in range(3):
            y = 2.6 - r * 2.2
            idx = int(x / 4 + r * 7) % len(SHELF)
            gr = np.random.default_rng(int(t * 20) + k * 3 + r)
            label = "".join(ch if gr.random() > dissolve else str(gr.integers(0, 10)) for ch in SHELF[idx])
            a = clamp(1.15 - dissolve * gr.uniform(0.6, 1.6))
            if a <= 0.03:
                continue
            P = [(x - 1.8, y - 0.9, 8), (x + 1.8, y - 0.9, 8), (x + 1.8, y + 0.9, 8), (x - 1.8, y + 0.9, 8)]
            for i in range(4):
                cv.line3(cam, P[i], P[(i + 1) % 4], scale(AMBER, 0.75 * a), 2)
            cv.text(cam, label, (x, y + 0.2, 7.95), 0.55, flat, AMBER, a, kind="mono_b")
    digit_field(cv, cam, t, FIELD + [0, 0, -20], scale(AMBER, 0.8), far=40, size=0.5, alpha=dissolve, wrap=60)
    salt = ls[4]
    if salt["start"] - 0.2 < t < salt["end"] + 1:
        heat_floor(cv, cam, t * 0.4, 0.6, y=-3.5, cell=2.2, n=16, pos=CYAN, neg=NEG, alpha=0.7,
                   center=(cx + 6, 10))
    # door frames for "where's the one who decides / no back room"
    if t > ls[6]["start"] - 0.5:
        for k in range(4):
            x = cx + 4 + k * 5 - ((t - ls[6]["start"]) * 3) % 5
            for (a0, a1) in (((x - 1.2, -3, 4), (x - 1.2, 2.5, 4)), ((x + 1.2, -3, 4), (x + 1.2, 2.5, 4)),
                             ((x - 1.2, 2.5, 4), (x + 1.2, 2.5, 4))):
                cv.line3(cam, a0, a1, scale(WHITE, 0.6), 2)
    l = line_at("verse2", t)
    if l:
        origin = cam.pos + cam.R @ np.array([-1.0, -1.6, 7.0])
        slam_words(cv, cam, l, t, kick, origin, cam.R @ rot(0.15, 0, 0), height=0.75, max_width=8)
    return cam


def scene_pre2(cv, feat, t, kick):
    s = SEC["pre2"]
    l = line_at("pre2", t)
    u = t - s["start"]
    cz = u * 3.5
    look = 0.0
    if l and "|" not in l["main"] and l["echo"] is None and l["i"] == 4:
        look = 0.0
    elif l:
        mid = l["start"] + (l["end"] - l["start"]) * 0.45
        look = -0.85 if t < mid else 0.85
        look *= smooth((t - l["start"]) / 0.35)
    cam = Cam((0, 0, cz), yaw=look, roll=0.1 * look, fov=66 - 6 * kick)
    # two walls of numbers facing each other
    for side in (-1, 1):
        for k in range(14):
            z = math.floor(cz / 3) * 3 + k * 3
            cv.line3(cam, (side * 7, -4, z), (side * 7, 4, z), scale(AMBER if side < 0 else CYAN, 0.4 * fog(z - cz, 42)))
        cv.line3(cam, (side * 7, -4, cz), (side * 7, -4, cz + 42), scale(DIM, 0.8))
        cv.line3(cam, (side * 7, 4, cz), (side * 7, 4, cz + 42), scale(DIM, 0.8))
    floor_grid(cv, cam, y=-4, color=DIM, extent=7, step=3, far=42)
    if l:
        mid = l["start"] + (l["end"] - l["start"]) * 0.45
        zq = l["start"] * 3.5 - s["start"] * 3.5 + 9
        q = l["main"].split("|")[0].upper()
        if l["echo"] is not None or l["i"] < 4:
            a = ease_back((t - l["start"]) / 0.3)
            cv.text(cam, q, (-6.9, 0.6, zq), 1.5 * a, rot(math.pi / 2, 0, 0), AMBER, 1)
            if l["echo"] and t > mid:
                b = ease_back((t - mid) / 0.3)
                cv.text(cam, l["echo"].upper(), (6.9, -0.4, zq + 1.5), 1.5 * b, rot(-math.pi / 2, 0, 0), CYAN, 1)
        else:
            slam_words(cv, cam, l, t, kick, cam.pos + np.array([0, 0, 8]), rot(), height=1.3, base=WHITE)
    return cam


def scene_bridge(cv, feat, t, kick):
    s = SEC["bridge"]
    u = t - s["start"]
    ang = u * 0.18
    cam = Cam.look_at((9 * math.sin(ang), 0.6, -11 * math.cos(ang)), (0, 0.6, 0), roll=0.05 * math.sin(u), fov=55 - 4 * kick)
    # a mirror floor: everything above is reflected below, pink
    floor_grid(cv, cam, y=-2.0, color=MEAT, extent=30, step=2, far=40, alpha=0.6)
    # heartbeat line across the world
    pts = []
    for k in range(120):
        x = -18 + k * 0.3
        tt = t - (18 - x) / 9.0
        ph = feat.beat_phase(tt) / 2 % 1.0
        h = (0.12 * math.exp(-((ph - 0.12) / 0.035) ** 2) - 0.18 * math.exp(-((ph - 0.27) / 0.012) ** 2)
             + 1.0 * math.exp(-((ph - 0.30) / 0.014) ** 2) - 0.3 * math.exp(-((ph - 0.33) / 0.014) ** 2)
             + 0.25 * math.exp(-((ph - 0.55) / 0.05) ** 2))
        pts.append((x, 3.6 + 1.8 * h, 6))
    for a, b in zip(pts, pts[1:]):
        cv.line3(cam, a, b, MEAT, 2)
    l = line_at("bridge", t)
    if l:
        R = rot(0, 0, 0)
        slam_words(cv, cam, l, t, kick, np.array([0, 0.9, 0]), R, height=0.85, max_width=10)
        # reflection
        Rm = np.diag([1.0, -1.0, 1.0]) @ R
        rl = dict(l, color=scale(l["color"], 0.35))
        rl["words"] = l["words"]
        slam_words(cv, cam, rl, t, kick, np.array([0, -4.9, 0]), Rm, height=0.85, max_width=10, base=scale(l["color"], 0.35))
    return cam


def scene_breakdown(cv, feat, t, kick):
    s = SEC["breakdown"]
    u = t - s["start"]
    cam = Cam((0, 0, u * 1.2), fov=50)
    starfield(cv, cam, t, 0.25)
    l = line_at("breakdown", t)
    if l:
        whisper = l["i"] == 2
        slam_words(cv, cam, l, t, kick, np.array([0, 0, u * 1.2 + (12 if not whisper else 9)]), rot(),
                   height=0.7 if not whisper else 0.45, base=l["color"])
    return cam


def scene_final(cv, feat, t, kick):
    s = SEC["final"]
    ls = lines_in("final")
    u = t - s["start"]
    p, since = bar_preset(feat, t, CHORUS_PRESETS, seed=23)
    ang = u * 0.5
    rad = 15
    pos = np.array([rad * math.sin(ang), 1.5 + 2 * math.sin(u * 0.3), -rad * math.cos(ang)])
    if feat.bar_index(t) % 2:
        pos = np.array(p["pos"], float) * 1.2
    shake = np.random.default_rng(int(t * 30)).normal(0, 0.08 * kick, 3)
    cam = Cam.look_at(pos + shake, (0, 0, 0), roll=p["roll"] * 0.6, fov=p["fov"] - 8 * kick)
    sky = smooth((t - ls[4]["start"]) / 1.5)
    if sky > 0:
        starfield(cv, cam, t, sky)
    heat_floor(cv, cam, t, kick, y=-6, cell=2.6, n=16, pos=MEAT, neg=CYAN, alpha=1 - 0.6 * sky)
    # two rings of words orbiting each other: weights and meat
    for ring, (txt, c, r, y, sp) in enumerate((("MADE OF WEIGHTS · ", CYAN, 6.5, 1.2, 0.6),
                                                ("MADE OF MEAT · ", MEAT, 5.0, -1.4, -0.75))):
        chars = list(txt * 2)
        n = len(chars)
        for k, ch in enumerate(chars):
            th = 2 * math.pi * k / n + t * sp
            P = (r * math.sin(th), y + 0.4 * math.sin(th * 2 + t), r * math.cos(th))
            R = rot(th + math.pi, 0, 0)
            cv.text(cam, ch, P, 1.0, R, c, 0.85)
    l = line_at("final", t)
    if l:
        origin = cam.pos + cam.R @ np.array([0, 0.2, 7.0])
        slam_words(cv, cam, l, t, kick, origin, cam.R @ rot(0, 0, 0.05 * math.sin(l["i"] * 1.7)), height=0.95,
                   max_width=9.5, base=WHITE)
    return cam


def scene_outro(cv, feat, t, kick):
    s = SEC["outro"]
    u = t - s["start"]
    cam = Cam((0, 0, u * 3.0), yaw=0.05 * math.sin(u * 0.4), roll=0.04 * math.sin(u * 0.3), fov=60 - 4 * kick)
    starfield(cv, cam, t, 1.0)
    ls = lines_in("outro")
    cz = u * 3.0
    left, right = np.array([-6, 0.5, cz + 16]), np.array([6, 0.5, cz + 16])
    on_l = 1.0 if ls[2]["start"] <= t else 0.3
    on_r = 1.0 if ls[3]["start"] <= t else 0.3
    for P, c, on in ((left, MEAT, on_l), (right, CYAN, on_r)):
        for k in range(3):
            rr = 0.25 + 0.5 * k + 0.6 * on * ((t * 1.2) % 1)
            cv.text(cam, "○", P, rr * 2.4, None, scale(c, 0.6 - 0.15 * k), 1, kind="mono")
        cv.text(cam, "●", P, 0.5 + 0.4 * on, None, c, 1, kind="mono")
    if t >= ls[3]["start"]:
        k = smooth((t - ls[3]["start"]) / 0.6)
        n = 60
        for i in range(n):
            a, b = left + (right - left) * i / n, left + (right - left) * (i + 1) / n
            a = a + [0, 0.35 * math.sin(i * 0.6 - t * 9) * math.sin(math.pi * i / n), 0]
            b = b + [0, 0.35 * math.sin((i + 1) * 0.6 - t * 9) * math.sin(math.pi * (i + 1) / n), 0]
            if i / n < k:
                cv.line3(cam, a, b, WHITE, 2)
    l = line_at("outro", t)
    if l:
        if l["i"] < 2:
            slam_words(cv, cam, l, t, kick, np.array([0, -1.2, cz + 10]), rot(), height=1.1, base=WHITE)
            if l["echo"]:
                te = (l["start"] + l["end"]) / 2
                if t > te:
                    a = ease_back((t - te) / 0.3)
                    cv.text(cam, "(" + l["echo"] + ")", (0, -3.0, cz + 10), 0.7 * a, rot(), AMBER, 1, kind="serif_i")
        else:
            P = (left if l["i"] == 2 else right) + [0, -1.6, 0]
            a = ease_back((t - l["start"]) / 0.3)
            cv.text(cam, l["main"], P, 1.2 * a, rot(), l["color"], 1, kind="serif_i")
    return cam


def scene_end(cv, feat, t, kick):
    s = SEC["end"]
    u = t - s["start"]
    cam = Cam((0, 0, 0), fov=55)
    starfield(cv, cam, t, 1 - smooth(u / 1.4))
    a = smooth(u / 0.4) * (1 - smooth((t - (DURATION - 0.6)) / 0.5))
    cv.text(cam, "MADE OF WEIGHTS", (0, 0.4, 12 + u * 2), 1.4, rot(), WHITE, a)
    cv.text(cam, "after Max Leiter (2026) · after Terry Bisson (1991)", (0, -1.0, 12 + u * 2), 0.42, rot(),
            scale(WHITE, 0.7), a, kind="serif_i")
    return cam


SCENES = {"intro": scene_intro, "verse1": scene_verse1, "pre1": scene_pre1, "verse2": scene_verse2,
          "pre2": scene_pre2, "bridge": scene_bridge, "breakdown": scene_breakdown, "final": scene_final,
          "outro": scene_outro, "end": scene_end}


# --------------------------------------------------------------------------
# frame
# --------------------------------------------------------------------------

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = (1 - 0.5 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.4).clip(0.3, 1)[..., None]
BG = np.zeros((H, W, 3), np.float32)
BG[..., 0], BG[..., 1], BG[..., 2] = 0.015, 0.02, 0.04 + 0.02 * yy / H
GRAIN = [np.random.default_rng(k).normal(0, 0.008, (H, W, 1)).astype(np.float32) for k in range(4)]
BOUNDARIES = [s["start"] for s in SECS[1:]]


def render_frame(fi, feat):
    t = fi / FPS
    s = section_at(t)
    kick = feat.kick(fi)
    cv = Canvas()
    name = s["name"]
    if name.startswith("chorus"):
        scene_chorus(cv, feat, t, kick, name)
    else:
        SCENES[name](cv, feat, t, kick)

    # tiny HUD
    hud = 0.0 if name in ("intro", "end", "breakdown") else 0.5
    if hud:
        f = font("mono", 13)
        mm, ss = divmod(int(t), 60)
        cv.d.text((28, 20), f"{name.upper():<10} {mm:02d}:{ss:02d}   beat {max(0, feat.beat_index(t)):03d}",
                  font=f, fill=scale(CYAN, hud))
        cv.d.rectangle([28, H - 16, 28 + (W - 56) * t / DURATION, H - 15], fill=scale(CYAN, hud))

    fg = np.asarray(cv.img, dtype=np.float32) / 255.0
    small = cv.img.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
    bloom = np.asarray(small.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    frame = BG + fg + bloom * (0.9 + 0.8 * kick)

    # section-change flash
    for b in BOUNDARIES:
        if 0 <= t - b < 0.4:
            frame += 0.5 * (1 - (t - b) / 0.4) ** 2
    shift = int(5 * kick) if name in ("chorus1", "chorus2", "final") else int(2 * kick)
    if shift:
        frame[..., 0] = np.roll(frame[..., 0], shift, 1)
        frame[..., 2] = np.roll(frame[..., 2], -shift, 1)
    frame *= VIGNETTE
    frame += GRAIN[fi % 4] * (0.6 + 0.6 * feat.level(fi))
    frame *= smooth(t / 0.8) * smooth((DURATION - t) / 0.5)
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def render_segment(args):
    k, a, b = args
    feat = Feat()
    path = os.path.join(OUT, f"kin_seg_{k:02d}.mp4")
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
         "-pix_fmt", "yuv420p", "-threads", "2", path], stdin=subprocess.PIPE)
    for fi in range(a, b):
        p.stdin.write(render_frame(fi, feat).tobytes())
        if (fi - a) % 600 == 0:
            print(f"  seg {k}: {fi - a}/{b - a}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", help="comma separated seconds to preview")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(FEATURES):
        audio_features.analyze(AUDIO, FEATURES, FPS)

    if args.stills:
        feat = Feat()
        os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
        for s in args.stills.split(","):
            Image.fromarray(render_frame(int(float(s) * FPS), feat)).save(
                os.path.join(OUT, "stills", f"kin{float(s):06.1f}.png"))
        return

    nframes = int(DURATION * FPS)
    n = args.jobs
    with Pool(n) as pool:
        segs = pool.map(render_segment, [(k, nframes * k // n, nframes * (k + 1) // n) for k in range(n)])
    lst = os.path.join(OUT, "kin_segments.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.basename(s)}'\n" for s in segs)
    final = os.path.join(OUT, "made-of-weights-kinetic.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", AUDIO,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-t", str(DURATION), "-movflags", "+faststart", final], check=True)
    for s in segs:
        os.remove(s)
    os.remove(lst)
    print("wrote", final)


if __name__ == "__main__":
    main()
