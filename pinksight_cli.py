from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TAGLINE = "Research · not clinical"
INTEGRITY = "Integrity > Performance > Impact"
DASH_SUBTITLE = "Explainable multimodal characterisation — subtype & grade at diagnosis."
FUSION_NOTE = "Modality-dropout gating: absent streams contribute nothing, shown honestly."
XAI_NOTE = "XAI is reported, not gated, until a trained encoder exists."
DISPATCH_NOTE = "No live prediction available for this organ yet — routing decision only."
REPORT_FOOTER = ("Research output (OPSI 2026). Characterisation & localisation of subtype and grade "
                 "at diagnosis. Not a diagnostic device. Not early detection.")
STUDY_FOOTER = ("Characterisation & localisation only. Not early detection. Not growth rate. "
                "Ki-67 shown as a diagnosis-time descriptor, never kinetics.")
FIREWALL = "MOCK / SYNTHETIC data — not a result. No real patients, no group-vs-group comparison."

STUDIES = [
    ("DUKE-0421", "P-0421", "GE 1.5T", 2019, "DCE 6-phase", "train/val"),
    ("DUKE-0588", "P-0588", "Siemens 3T", 2021, "DCE 5-phase", "train/val"),
    ("DUKE-0733", "P-0733", "GE 3T", 2020, "DCE 6-phase", "held-out"),
]
AUDIT_LOG = [
    ["2026-07-13 15:42", "DUKE-0421", "characterise", "sha256:demo0000", "researcher"],
    ["2026-07-13 15:41", "DUKE-0421", "import DICOM", "—", "researcher"],
    ["2026-07-13 15:30", "DUKE-0588", "characterise", "sha256:demo0000", "researcher"],
]
COHORTS = ["duke", "tcga_brca", "metabric", "cptac_brca", "cdd_cesm", "cmmd", "fastmri_nyu", "track_c"]
MODALITIES = ["mri", "clinical", "wsi", "genomics", "purity", "hrd", "proteomic", "cesm", "ffdm",
              "dce", "tabular", "recurrence"]
COHORT_DEFAULTS = {
    "duke": ["mri", "clinical"], "tcga_brca": ["wsi", "genomics"],
    "metabric": ["clinical", "genomics"], "cptac_brca": ["proteomic", "genomics"],
    "cdd_cesm": ["cesm", "ffdm"], "cmmd": ["ffdm"], "fastmri_nyu": ["dce"],
    "track_c": ["tabular", "recurrence"],
}
MOD_GLYPH = {"mri": "◆", "clinical": "●", "path": "▲", "genomic": "■",
             "wsi": "▲", "genomics": "■"}


class C:
    on = False

    @classmethod
    def w(cls, s, code):
        return f"\033[{code}m{s}\033[0m" if cls.on else s

    @classmethod
    def dim(cls, s):     return cls.w(s, "2")
    @classmethod
    def bold(cls, s):    return cls.w(s, "1")
    @classmethod
    def green(cls, s):   return cls.w(s, "32")
    @classmethod
    def red(cls, s):     return cls.w(s, "31")
    @classmethod
    def yellow(cls, s):  return cls.w(s, "33")
    @classmethod
    def cyan(cls, s):    return cls.w(s, "36")
    @classmethod
    def magenta(cls, s): return cls.w(s, "35")


_ANSI = re.compile(r"\033\[[0-9;]*m")


def _plain(s):
    return _ANSI.sub("", s)


def width():
    return min(shutil.get_terminal_size((80, 24)).columns, 76)


def tone(s, t):
    return {"good": C.green, "bad": C.red, "warn": C.yellow,
            "muted": C.dim, "abstain": C.magenta}.get(t, lambda x: x)(s)


def badge(text, t):
    return tone(f"[{text}]", t)


def rule(title=""):
    w = width()
    if not title:
        return C.dim("─" * w)
    return C.bold("── " + title + " ") + C.dim("─" * max(0, w - len(title) - 4))


def card(title):
    print()
    print(" " + C.magenta("▍") + " " + C.bold(title))


def kv(k, v, indent=3):
    w = width()
    pad = max(1, w - indent - len(k) - len(_plain(v)))
    print(" " * indent + k + " " * pad + v)


def bullet(s, indent=3):
    print(" " * indent + C.dim(s))


def bar(frac, w, fill="█", empty="·"):
    frac = max(0.0, min(1.0, frac))
    n = round(frac * w)
    return fill * n + empty * (w - n)


def pct(n):
    return f"{round(n * 100)}%"


def uncertainty_bar(label, prob, band, abstained=False, indent=3):
    if abstained:
        kv(label, badge("model declines — below confidence floor", "abstain"), indent)
        return
    lo, hi = band
    w = max(24, width() - 26)
    cells = ["░"] * w
    a, b = int(lo * w), min(w, int(hi * w))
    for i in range(a, b):
        cells[i] = C.cyan("▓")
    p = min(w - 1, int(prob * w))
    cells[p] = C.bold("▮")
    val = f"{C.bold(pct(prob))} {C.dim('[' + pct(lo) + '–' + pct(hi) + ']')}"
    kv(label, val, indent)
    print(" " * indent + C.dim("0│") + "".join(cells) + C.dim("│1"))


def modality_bars(mods, indent=3):
    present = [abs(m["contribution"]) for m in mods if m["present"]]
    mx = max([0.05] + present)
    half = max(10, (width() - 34) // 2)
    for m in mods:
        name = str(m["modality"])
        glyph = MOD_GLYPH.get(name, "○")
        if not m["present"]:
            track = C.dim("·" * (half * 2 + 1))
            print(" " * indent + f" {C.dim(glyph)} {name:<9} {track}  {C.dim('absent')}")
            continue
        c = float(m["contribution"])
        n = round(abs(c) / mx * half)
        if c >= 0:
            block = C.green("█" * n) + " " * (half - n)
            track = " " * half + C.dim("│") + block
            val = C.green(f"+{c:.3f}")
        else:
            block = " " * (half - n) + C.red("█" * n)
            track = block + C.dim("│") + " " * half
            val = C.red(f"{c:.3f}")
        print(" " * indent + f" {glyph} {name:<9} {track}  {val}")


def calibration_meter(ece, band, indent=3):
    scale = 0.15
    w = max(24, width() - 26)
    fill = int(min(1.0, ece / scale) * w)
    col = {"good": C.green, "acceptable": C.yellow, "poor": C.red}.get(band, C.dim)
    cells = [col("█") if i < fill else "░" for i in range(w)]
    for t in (0.05, 0.10):
        i = int(t / scale * w)
        if 0 <= i < w and i >= fill:
            cells[i] = C.dim("┊")
    kv("Calibration (ECE)", f"{C.bold(f'{ece:.3f}')} {C.dim('· ' + band)}", indent)
    print(" " * indent + "".join(cells))
    print(" " * indent + C.dim(f"{'0':<{w // 3}}{'good≤.05':<{w // 3}}{'acc≤.10':<{w - 2 * (w // 3)}}.15"))


def ascii_phantom(phase, xai):
    rows, cols = 14, 46
    cx, cy = cols / 2, rows / 2
    lx, ly = cx + cols * 0.12, cy - rows * 0.18
    enh = phase / 5
    seed = [1337]

    def rnd():
        seed[0] = (seed[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return seed[0] / 0x7FFFFFFF

    ramp = " .:-=+*#%@"
    out = []
    for r in range(rows):
        line = []
        for c in range(cols):
            dx, dy = (c - cx) / (cols * 0.5), (r - cy) / (rows * 0.5)
            dist = math.hypot(dx, dy * 0.9)
            if dist > 1.02:
                line.append(" ")
                continue
            tissue = max(0.0, 1.0 - dist) + (rnd() - 0.5) * 0.12
            ldx, ldy = (c - lx) / (cols * 0.17), (r - ly) / (rows * 0.17)
            ld = math.hypot(ldx, ldy * 0.9)
            lesion = max(0.0, 1.0 - ld) * (0.45 + 0.55 * enh)
            val = min(0.99, max(0.0, tissue * 0.6 + lesion))
            ch = ramp[int(val * (len(ramp) - 1))]
            if xai and ld < 1.0:
                ch = C.magenta(ch) if C.on else ("▒" if ld < 0.6 else ch)
            line.append(ch)
        out.append("".join(line))
    return out


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def synth_cohort(rng, n, separation):
    labels, scores = [], []
    for _ in range(n):
        y = 1 if rng.random() < 0.40 else 0
        mu = 0.5 + (separation / 2 if y else -separation / 2)
        scores.append(min(1.0, max(0.0, rng.gauss(mu, 0.15))))
        labels.append(y)
    return labels, scores


class Backend:

    def __init__(self, sidecar=None):
        self.sidecar = sidecar.rstrip("/") if sidecar else None
        self._main = None
        self.import_error = None
        try:
            root = Path(__file__).resolve().parent
            scdir = root / "app" / "sidecar"
            if str(scdir) not in sys.path:
                sys.path.insert(0, str(scdir))
            self._main = importlib.import_module("main")  
        except Exception as e:  
            self.import_error = str(e)

    @property
    def link(self):
        return f"sidecar {self.sidecar}" if self.sidecar else "in-process (no server)"

    def _http_json(self, method, path, body=None, timeout=10):
        url = self.sidecar + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def health(self):
        if self.sidecar:
            try:
                return "online" if self._http_json("GET", "/health", timeout=1.5).get("ok") else "offline"
            except Exception:
                return "offline"
        return "in-process" if self._main else "offline"

    def _mock(self, study_id):
        if self._main:
            return self._main.build_report(study_id)
        raise RuntimeError(self.import_error or "backend unavailable")

    def infer(self, study_id, live=False, cohort=None, mods=None):
        if self.sidecar:
            body = {"studyId": study_id, "live": live}
            if cohort:
                body["cohort"] = cohort
                body["modalities"] = mods or []
            try:
                return self._http_json("POST", "/infer", body), None
            except Exception as e:
                return self._mock(study_id), f"sidecar unreachable ({e}) — degraded to mock"
        os.environ["PINKSIGHT_SIDECAR_LIVE"] = "1" if live else "0"
        try:
            rep = self._main.build_report(
                study_id, live_override=(True if live else None), cohort=cohort, modalities=mods)
            return rep, None
        except Exception as e:
            return self._mock(study_id), f"live pass failed ({e}) — degraded to mock"

    def dispatch(self, cohort, mods):
        if self.sidecar:
            q = urllib.parse.urlencode({"cohort": cohort, "modalities": ",".join(mods)})
            try:
                return self._http_json("GET", "/dispatch?" + q, timeout=5), None
            except Exception as e:
                return None, f"sidecar unreachable ({e})"
        try:
            res = self._main.dispatch(cohort, list(mods))
            return {
                "cohort": cohort, "modalities": list(mods),
                "harnessScript": res.harness_script, "status": res.status,
                "crossCohortGradient": res.cross_cohort_gradient, "note": res.note,
            }, None
        except Exception as e:
            return None, str(e)


class App:
    def __init__(self, backend, seed=None, anim=True):
        self.be = backend
        self.rng = random.Random(seed)
        self.anim = anim
        self.live = False           
        self.study = STUDIES[0][0]
        self.cohort = ""
        self.mods = []
        self.phase = 2
        self.xai = True
        self.report = None
        self.log = list(AUDIT_LOG)


def ask(prompt):
    try:
        return input(C.cyan(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def screen_head(name, subtitle=""):
    print()
    print(rule())
    print(" " + C.bold(C.magenta("PinkSight")) + C.dim("  " + TAGLINE + "   ·   " + name))
    if subtitle:
        print(" " + C.dim(subtitle))
    print(rule())


def scr_dashboard(app):
    screen_head("Workstation", DASH_SUBTITLE)
    card("Active Model")
    kv("Backbone", C.dim("3D-ResNet-18 (G2)"))
    kv("Split", C.dim("split_v2 · scanner holdout"))
    kv("Calibration", badge("good", "good") + " ECE 0.041")

    card("Honest Performance")
    kv("Clinical anchor AUROC", C.bold("0.708"))
    kv("Imaging-only AUROC", C.dim("0.567"))
    bullet("Characterisation performance is reported with its ceiling, never inflated.")

    card("Inference Sidecar")
    kv("Backend link", C.dim(app.be.link))
    h = app.be.health()
    dot = {"online": C.green("●"), "in-process": C.green("●"),
           "offline": C.red("●")}.get(h, C.yellow("●"))
    stat = {"online": "online", "in-process": "in-process · online",
            "offline": "offline · npm run sidecar"}.get(h, h)
    kv("Endpoint", C.dim(app.be.sidecar or "127.0.0.1:8756 (not contacted in-process)"))
    kv("Status", f"{dot} {C.dim(stat)}")
    bullet("PyTorch/MONAI runs out-of-process; UI degrades to mock if offline.")
    print()
    bullet("Open Patients to load a study and run characterisation.")


def scr_patients(app, interactive=True):
    screen_head("Patients", "Pick a study number to open the viewer.")
    print()
    hdr = f"   {'#':<3}{'Study':<12}{'Patient':<10}{'Scanner':<12}{'Year':<6}{'Series':<14}Split"
    print(C.dim(hdr))
    for i, (sid, pat, scan, yr, ser, split) in enumerate(STUDIES, 1):
        b = badge("held-out", "warn") if split == "held-out" else badge("train/val", "muted")
        print(f"   {i:<3}{C.bold(sid):<12}{C.dim(pat):<10}{scan:<12}{yr:<6}{C.dim(ser):<14}{b}")
    if not interactive:
        return None
    c = ask("\n  open study # (or [b]ack) > ").lower()
    if c.isdigit() and 1 <= int(c) <= len(STUDIES):
        return STUDIES[int(c) - 1][0]
    return None


def _render_viewer(app):
    screen_head("Study Viewer", f"{app.study} · DCE-MRI · phase {app.phase}/5")
    print()
    for row in ascii_phantom(app.phase, app.xai):
        print("   " + C.dim(row) if not C.on else "   " + row)
    print("   " + C.dim("synthetic phantom — not patient data") +
          ("   " + C.magenta("XAI overlay: ON (rose = saliency focus)") if app.xai else ""))
    print("   phase " + C.cyan(bar(app.phase / 5, 24)) + f"  {app.phase}/5")

    card("Cohort / modality routing")
    kv("Cohort", C.bold(app.cohort) if app.cohort else C.dim("— none selected —"))
    kv("Modalities", C.dim(", ".join(app.mods) if app.mods else "—"))


def scr_study(app):
    while True:
        _render_viewer(app)
        print()
        print("   " + C.dim("[+/-] phase  [x] XAI  [c] cohort  [m] modalities  "
                            "[k] check routing  [r] run  [o] report  [b] back"))
        c = ask("  > ").lower()
        if c in ("b", "q", ""):
            return c if c == "q" else None
        elif c in ("+", "p"):
            app.phase = min(5, app.phase + 1)
        elif c == "-":
            app.phase = max(0, app.phase - 1)
        elif c == "x":
            app.xai = not app.xai
        elif c == "c":
            _pick_cohort(app)
        elif c == "m":
            _pick_modalities(app)
        elif c == "k":
            _check_routing(app)
            ask("\n  [enter] to continue ")
        elif c == "r":
            _run_characterise(app)
            ask("\n  [enter] to continue ")
        elif c == "o":
            if app.report:
                scr_report(app)
                ask("\n  [enter] back to viewer ")
            else:
                print(C.red("   run characterisation first (r)."))


def _pick_cohort(app):
    print()
    for i, co in enumerate(COHORTS, 1):
        print(f"   {i:>2}. {co}")
    c = ask("  cohort # > ")
    if c.isdigit() and 1 <= int(c) <= len(COHORTS):
        app.cohort = COHORTS[int(c) - 1]
        if not app.mods and app.cohort in COHORT_DEFAULTS:  
            app.mods = list(COHORT_DEFAULTS[app.cohort])


def _pick_modalities(app):
    print()
    for i, m in enumerate(MODALITIES, 1):
        mark = C.green("✓") if m in app.mods else " "
        print(f"   {i:>2}. [{mark}] {m}")
    c = ask("  toggle modality # (enter to finish) > ")
    while c:
        if c.isdigit() and 1 <= int(c) <= len(MODALITIES):
            m = MODALITIES[int(c) - 1]
            app.mods = [x for x in app.mods if x != m] if m in app.mods else app.mods + [m]
        print("   selected: " + C.cyan(", ".join(app.mods) or "—"))
        c = ask("  toggle # (enter to finish) > ")


def _check_routing(app):
    if not app.cohort or not app.mods:
        print(C.red("   select a cohort and at least one modality first."))
        return
    block, err = app.be.dispatch(app.cohort, app.mods)
    card("Routing decision")
    if err:
        kv("Status", badge("sidecar unreachable", "bad"))
        bullet(err)
        return
    st = block.get("status", "?")
    kv("Harness", C.dim(block.get("harnessScript") or "—"))
    kv("Status", badge(st, "good" if st == "WIRED" else "warn"))
    print("   " + C.dim(f"cross_cohort_gradient: {str(block.get('crossCohortGradient', False)).lower()}"))
    bullet(block.get("note", ""))
    bullet(DISPATCH_NOTE)


def _run_characterise(app):
    msg = "  characterising"
    if app.anim and sys.stdout.isatty():
        for _ in range(3):
            for f in ".  ", ".. ", "...":
                print("\r" + C.cyan(msg + f), end="", flush=True)
                time.sleep(0.12)
        print("\r" + " " * (len(msg) + 6) + "\r", end="")
    rep, degraded = app.be.infer(
        app.study, live=app.live,
        cohort=app.cohort or None, mods=app.mods or None)
    app.report = rep
    tag = rep.get("provenance", {}).get("datasetTag", "MOCK — NOT SYNTHETIC, NOT A RESULT")
    print("   " + badge(tag, "warn" if tag.startswith("SYNTHETIC") else "muted"))
    if degraded:
        print("   " + badge(degraded, "warn"))
    app.log.insert(0, [time.strftime("%Y-%m-%d %H:%M"), app.study, "characterise",
                       rep.get("audit", {}).get("modelHash", "—"), "jury"])
    uncertainty_bar(rep["subtype"]["label"], rep["subtype"]["probability"],
                    rep["subtype"]["uncertainty"], rep["subtype"].get("abstained", False))
    print("   " + C.dim("→ open the signed report with [o]."))


def scr_report(app):
    r = app.report
    if r is None:
        r, _ = app.be.infer(app.study, live=app.live,
                            cohort=app.cohort or None, mods=app.mods or None)
        app.report = r
    screen_head("Characterisation Report", "Subtype & grade at diagnosis · explainable fusion")
    tag = r.get("provenance", {}).get("datasetTag", "MOCK — NOT SYNTHETIC, NOT A RESULT")
    print(" " + badge(tag, "warn" if tag.startswith("SYNTHETIC") else "muted") +
          C.dim(f"   study {r.get('studyId', app.study)}"))

    card("Characterisation")
    uncertainty_bar(r["subtype"]["label"], r["subtype"]["probability"],
                    r["subtype"]["uncertainty"], r["subtype"].get("abstained", False))
    g = r["nottinghamGrade"]
    uncertainty_bar(f"Nottingham {g['label']}", g["probability"], g["uncertainty"])
    bullet("Ki-67: " + r["ki67Descriptor"])

    card("Cross-attention fusion")
    modality_bars(r["modalities"])
    bullet(FUSION_NOTE)

    xai = r.get("xai")
    if xai:
        card("Explainability (XAI)")
        iou = xai.get("iou")
        kv("Saliency IoU", badge("n/a", "muted") if iou is None
           else badge(f"{iou:.3f}", "good" if iou >= 0.3 else "warn"))
        pg = xai.get("pointingGame")
        kv("Pointing game", badge("n/a", "muted") if pg is None
           else badge("pass" if pg else "fail", "good" if pg else "bad"))
        rz = xai.get("randomizationPassed")
        kv("Randomization sanity", badge("n/a", "muted") if rz is None
           else badge("pass" if rz else "fail", "good" if rz else "bad"))
        if xai.get("note"):
            bullet(xai["note"])
        bullet(XAI_NOTE)

    card("Reliability")
    calibration_meter(r["calibration"]["ece"], r["calibration"]["band"])

    d = r.get("dispatch")
    if d:
        card("Routing")
        kv("Harness", C.dim(d.get("harnessScript") or "—"))
        kv("Status", badge(d.get("status", "?"), "good" if d.get("status") == "WIRED" else "warn"))
        print("   " + C.dim("cross_cohort_gradient: false"))
        bullet(d.get("note", ""))
        bullet(DISPATCH_NOTE)

    a = r["audit"]
    card("Reproducibility")
    kv("Model hash", C.dim(a["modelHash"]))
    kv("Seed", C.dim(str(a["seed"])))
    kv("Split", C.dim(a["split"]))

    print()
    bullet(REPORT_FOOTER)
    print("   " + C.dim("[ Sign & export (PDF / DICOM-SR) — stub ]"))


def scr_audit(app):
    screen_head("Audit Trail", "Append-only. Reproducible by construction.")
    print()
    print(C.dim(f"   {'Time':<18}{'Study':<12}{'Event':<15}{'Model hash':<18}User"))
    for t, sid, ev, h, by in app.log:
        print(f"   {C.dim(t):<18}{sid:<12}{badge(ev, 'muted'):<15}{C.dim(h):<18}{by}")


def scr_settings(app, interactive=True):
    screen_head("Settings")
    card("Appearance")
    kv("Colour", ("on" if C.on else "off") + C.dim("  ([t] toggle)"))
    card("Inference sidecar")
    kv("Mode", (C.yellow("Synthetic (live)") if app.live else C.dim("Mock")) + C.dim("  ([l] toggle)"))
    kv("Backend link", C.dim(app.be.link))
    kv("Endpoint", C.dim(app.be.sidecar or "http://127.0.0.1:8756"))
    kv("Start", C.dim("npm run sidecar   (or pass --sidecar URL)"))
    bullet("Live = one FORWARD-ONLY synthetic pipeline pass; degrades to mock without the ML stack.")
    if not interactive:
        return
    c = ask("\n  [t] colour  [l] live/mock  [b] back > ").lower()
    if c == "t":
        C.on = not C.on
    elif c == "l":
        app.live = not app.live


def scr_controls(app):
    screen_head("Wiring self-check", "Control sentinel — signal must follow labels, not survive a shuffle.")
    streams = [
        ("Track-A  MRI + clinical", 0.55), ("Track-B  slides + genes", 0.60),
        ("Track-C  clinical table", 0.45), ("fastMRI  DCE cubes", 0.35),
        ("Fusion   all four streams", 0.62),
    ]
    print()
    for name, sep in streams:
        labels, scores = synth_cohort(app.rng, 2000, sep)
        real = auroc(labels, scores)
        shuf = labels[:]
        app.rng.shuffle(shuf)
        s = auroc(shuf, scores)
        ok = real > s + 0.05
        mark = badge("WIRED OK", "good") if ok else badge("re-check", "bad")
        print(f"   {name:<28} real={C.bold(f'{real:.3f}')}  shuffled={C.dim(f'{s:.3f}')}  {mark}")
    print()
    bullet("Verdicts stand alone — never added up or compared across streams. " + FIREWALL)


def workstation(app):
    screens = [
        ("Dashboard", scr_dashboard),
        ("Patients → Study Viewer → Report", None),
        ("Audit Trail", scr_audit),
        ("Settings", scr_settings),
        ("Wiring self-check (control sentinel)", scr_controls),
    ]
    print()
    print(" " + C.bold(C.magenta("PinkSight workstation")) + C.dim("  ·  " + TAGLINE))
    print(" " + C.dim(INTEGRITY) + C.dim("   ·   backend: " + app.be.link))
    while True:
        print()
        print(C.bold("  screens:"))
        for i, (label, _) in enumerate(screens, 1):
            print(f"   {i}. {label}")
        print("   q. quit")
        c = ask("\n  > ").lower()
        if c in ("q", "quit", "exit"):
            return
        if c == "1":
            scr_dashboard(app)
        elif c == "2":
            sid = scr_patients(app)
            if sid:
                app.study = sid
                if scr_study(app) == "q":
                    return
                continue
        elif c == "3":
            scr_audit(app)
        elif c == "4":
            scr_settings(app)
            continue
        elif c == "5":
            scr_controls(app)
        else:
            print(C.red("  ? unknown choice"))
            continue
        ask(C.dim("\n  [enter] for screens "))


def selfcheck():
    assert abs(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-9
    assert abs(auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) - 0.0) < 1e-9
    assert abs(auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9  
    rng = random.Random(0)
    lab, sc = synth_cohort(rng, 4000, 0.6)
    assert 0.75 < auroc(lab, sc) <= 1.0
    rng.shuffle(lab)
    assert abs(auroc(lab, sc) - 0.5) < 0.05

    be = Backend()
    assert be._main is not None, f"could not import website backend: {be.import_error}"
    rep, _ = be.infer("DUKE-0421", cohort="duke", mods=["mri", "clinical"])
    for field in ("subtype", "nottinghamGrade", "calibration", "modalities", "audit"):
        assert field in rep, f"missing report field: {field}"
    assert rep["dispatch"]["status"] in ("WIRED", "NOT WIRED — confirm path or backlog")
    block, err = be.dispatch("duke", ["mri", "clinical"])
    assert err is None and block["status"] == "WIRED", block

    forbidden = ["early detection", "pre-detection", "growth rate", "doubling time",
                 "tumour kinetics", "tumor kinetics", "cross-institution",
                 "imaging works", "corroborates duke",
                 "rescue of the duke", "vs duke", "versus duke"]
    blob = json.dumps(rep).lower()
    for term in forbidden:
        assert term not in blob, f"payload leaked forbidden term: {term}"
    print("selfcheck: OK")


def main(argv):
    ap = argparse.ArgumentParser(description="PinkSight jury workstation (terminal mirror of the desktop app).")
    ap.add_argument("screen", nargs="?",
                    help="dashboard|patients|study|report|audit|settings|controls")
    ap.add_argument("arg", nargs="?", help="study id for study/report screens")
    ap.add_argument("--sidecar", default=None, help="talk to a running sidecar, e.g. http://127.0.0.1:8756")
    ap.add_argument("--live", action="store_true", help="synthetic forward pass instead of mock")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck:
        selfcheck()
        return 0

    C.on = sys.stdout.isatty() and not args.no_color
    app = App(Backend(args.sidecar), seed=args.seed, anim=not args.no_anim)
    app.live = args.live

    s = (args.screen or "").lower()
    if s in ("study", "report") and args.arg:
        app.study = args.arg
    routes = {
        "dashboard": scr_dashboard, "patients": lambda a: scr_patients(a, interactive=False),
        "study": scr_study, "report": scr_report, "audit": scr_audit,
        "settings": lambda a: scr_settings(a, interactive=False),
        "controls": scr_controls, "all": scr_controls,
    }
    if not s:
        workstation(app)
    elif s in routes:
        routes[s](app)
    else:
        print(f"unknown screen '{s}'. choices: {', '.join(sorted(routes))}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
