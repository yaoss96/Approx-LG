"""Logic synthesis / PPA characterisation back-end.

Flow (identical for every design point, proposed or baseline):

    Verilog -> Yosys ``synth -flatten`` -> ABC technology mapping onto the
    45 nm Nangate Open Cell Library -> area from ``stat -liberty`` and
    critical-path delay from ABC's ``stime``.  Power is obtained from OpenSTA
    on the mapped netlist with the operand pins annotated as uniformly random.

Both back-ends are optional: if OpenSTA is unavailable the analytical
activity-based estimator in :func:`estimate_power` is used instead, which is
calibrated on the cells' liberty capacitance and leakage values.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_LIB = os.path.join(ROOT, "lib", "NangateOpenCellLibrary_typical.lib")

VDD = 1.1          # V, Nangate45 typical corner
DEFAULT_FREQ = 2e8  # Hz (200 MHz, matching the reference works)


def _find(*names) -> Optional[str]:
    local = os.path.join(ROOT, "oss-cad-suite", "bin")
    for nm in names:
        p = os.path.join(local, nm + (".exe" if os.name == "nt" else ""))
        if os.path.exists(p):
            return p
        w = shutil.which(nm)
        if w:
            return w
    return None


@dataclass
class PPA:
    area: float = float("nan")        # um^2
    delay: float = float("nan")       # ns
    power: float = float("nan")       # uW
    leakage: float = float("nan")     # uW
    gates: int = 0
    ok: bool = False

    @property
    def adp(self) -> float:
        return self.area * self.delay

    @property
    def pdp(self) -> float:
        return self.power * self.delay          # uW * ns = fJ

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["ADP"] = self.adp
        d["PDP"] = self.pdp
        return d


# Delay-driven technology mapping with area recovery.  ``stime`` reports the
# area and the critical-path delay of the final mapped netlist.
# Mapping strategies used by the recipe sweep.  ABC's ``map -D`` (required
# time) has no effect in this build -- area and delay are bit-identical from
# 0.3ns to 5ns -- so the sweep varies the mapping strategy instead, which does
# move the trade-off: on the exact 16x16 multiplier these five span 1526 to
# 2305 um^2 and 0.94 to 1.44 ns.
MAP_RECIPES = {
    "default": "map,-B,0.9",     # delay-driven with area recovery
    "delay": "map,-B,0",         # delay-driven, no area recovery
    "balanced": "map,-B,0.5",
    "area": "map,-a",            # area mode
    "amap": "amap",              # area-oriented mapper
}

_ABC_SHELL = ("+strash;ifraig;scorr;dc2;dretime;strash;dch,-f;{m};"
              "topo;buffer,-c;upsize,-c;dnsize,-c;stime,-p")


def abc_script(recipe: str = "default") -> str:
    """The mapping recipe.

    Every number in the paper comes from ``default``.  Whether an *ordering*
    between two designs is an artefact of that single operating point can only
    be answered by sweeping it, which is what the other entries are for.
    """
    return _ABC_SHELL.replace("{m}", MAP_RECIPES[recipe])


YS_TEMPLATE = """
read_verilog {src}
hierarchy -check -top {top}
synth -top {top} -flatten
opt -purge
abc -liberty {lib} -script {abc}
setundef -zero
opt_clean -purge
stat -liberty {lib}
write_json {js}
write_verilog -noattr -noexpr -nohex -nodec {out}
"""


class Synthesizer:
    def __init__(self, lib: str = DEFAULT_LIB, workdir: Optional[str] = None,
                 use_sta: bool = False, freq: float = DEFAULT_FREQ,
                 cache: Optional[str] = None, n: int = 8,
                 power_vectors: int = 1 << 14, power: bool = True,
                 recipe: str = "default"):
        # "default" reproduces the recipe every number in the paper uses.
        self.recipe = recipe
        self.lib = os.path.abspath(lib).replace("\\", "/")
        self.yosys = _find("yosys")
        self.sta = _find("sta", "opensta") if use_sta else None
        self.workdir = workdir or tempfile.mkdtemp(prefix="approxlg_syn_")
        os.makedirs(self.workdir, exist_ok=True)
        self.freq = freq
        self.cache_path = cache
        self.cache: Dict[str, Dict] = {}
        if cache and os.path.exists(cache):
            try:
                with open(cache) as fh:
                    self.cache = json.load(fh)
            except Exception:
                self.cache = {}
        self.pest = None
        self.pstim = None
        if power:
            from .power import PowerEstimator, PowerStimulus
            self.pest = PowerEstimator(
                self.lib, cache=os.path.join(os.path.dirname(self.lib),
                                             "nangate45.json"), freq=freq)
            self.pstim = PowerStimulus(n, power_vectors)

    # ------------------------------------------------------------------
    def save_cache(self):
        # Best effort: concurrent runs may share the cache file on Windows,
        # where a replace can transiently fail; losing one snapshot only costs
        # a re-synthesis, so it must never abort a search.
        if not self.cache_path:
            return
        tmp = f"{self.cache_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(self.cache, fh)
            os.replace(tmp, self.cache_path)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def _recipe_tag(self) -> str:
        return "" if self.recipe == "default" else f"|{self.recipe}"

    def run(self, verilog: str, top: str = "mul", tag: str = "",
            a_port: str = "a", b_port: str = "b", retries: int = 3) -> PPA:
        h = hashlib.sha1((verilog + top + self._recipe_tag()).encode()
                         ).hexdigest()[:16]
        if h in self.cache:
            return PPA(**{k: v for k, v in self.cache[h].items()
                          if k in PPA.__dataclass_fields__})
        # Under heavy load a synthesis can fail for reasons that have nothing
        # to do with the design (process-spawn or timeout); retry before giving
        # up, and never cache a failure -- a poisoned entry would silently
        # degrade every later run that touches the same design point.
        for attempt in range(retries):
            ppa = self._synthesize(verilog, top, h, a_port, b_port)
            if ppa.ok:
                self.cache[h] = ppa.as_dict()
                return ppa
            time.sleep(0.5 * (attempt + 1))
        return ppa

    def run_files(self, files, top: str, incdirs=(), a_port: str = "a",
                  b_port: str = "b") -> PPA:
        """Synthesise a multi-file third-party design with the identical flow."""
        blob = []
        for f in files:
            with open(f, "r", errors="replace") as fh:
                blob.append(f"// ==== {os.path.basename(f)} ====\n" + fh.read())
        inc = " ".join(f"-I{os.path.abspath(d)}".replace("\\", "/")
                       for d in incdirs)
        txt = "\n".join(blob)
        h = hashlib.sha1((txt + top + inc).encode()).hexdigest()[:16]
        if h in self.cache:
            return PPA(**{k: v for k, v in self.cache[h].items()
                          if k in PPA.__dataclass_fields__})
        d = os.path.join(self.workdir, h)
        os.makedirs(d, exist_ok=True)
        reads = "\n".join("read_verilog " + inc + " " +
                          os.path.abspath(f).replace("\\", "/") for f in files)
        out = os.path.join(d, "map.v").replace("\\", "/")
        js = os.path.join(d, "map.json").replace("\\", "/")
        ys = os.path.join(d, "run.ys")
        with open(ys, "w") as fh:
            fh.write(f"""{reads}
hierarchy -check -top {top}
synth -top {top} -flatten
opt -purge
abc -liberty {self.lib} -script {abc_script(self.recipe)}
setundef -zero
opt_clean -purge
stat -liberty {self.lib}
write_json {js}
write_verilog -noattr -noexpr -nohex -nodec {out}
""")
        subprocess.run([self.yosys, "-q", "-l", os.path.join(d, "yosys.log"), ys],
                       capture_output=True, text=True, timeout=1800)
        logf = os.path.join(d, "yosys.log")
        log = open(logf).read() if os.path.exists(logf) else ""
        ppa = self._parse_log(log)
        if ppa.ok and self.pest is not None:
            rep = self.pest.estimate(js, top, self.pstim, a_port, b_port)
            if rep:
                ppa.power = rep.total_uw
                ppa.leakage = rep.leakage_uw
        if ppa.power != ppa.power and ppa.ok:
            ppa.power = estimate_power_from_log(log, ppa, self.freq)
        self.cache[h] = ppa.as_dict()
        return ppa

    # ------------------------------------------------------------------
    def _synthesize(self, verilog: str, top: str, h: str,
                    a_port: str = "a", b_port: str = "b") -> PPA:
        d = os.path.join(self.workdir, h)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "src.v").replace("\\", "/")
        out = os.path.join(d, "map.v").replace("\\", "/")
        js = os.path.join(d, "map.json").replace("\\", "/")
        ys = os.path.join(d, "run.ys")
        with open(src, "w") as fh:
            fh.write(verilog)
        with open(ys, "w") as fh:
            fh.write(YS_TEMPLATE.format(src=src, top=top, lib=self.lib,
                                        out=out, js=js, abc=abc_script(self.recipe)))
        try:
            r = subprocess.run([self.yosys, "-q", "-l",
                                os.path.join(d, "yosys.log"), ys],
                               capture_output=True, text=True, timeout=600)
        except Exception:                                     # pragma: no cover
            return PPA(ok=False)
        log = open(os.path.join(d, "yosys.log")).read() if \
            os.path.exists(os.path.join(d, "yosys.log")) else (r.stdout + r.stderr)
        ppa = self._parse_log(log)
        if ppa.ok and self.sta and os.path.exists(out):
            st = self._run_sta(d, out, top, ppa)
            if st:
                ppa.delay = st.get("delay", ppa.delay)
                ppa.power = st.get("power", float("nan"))
                ppa.leakage = st.get("leakage", float("nan"))
        if ppa.ok and self.pest is not None and ppa.power != ppa.power:
            rep = self.pest.estimate(js, top, self.pstim, a_port, b_port)
            if rep:
                ppa.power = rep.total_uw
                ppa.leakage = rep.leakage_uw
        if ppa.power != ppa.power and ppa.ok:
            ppa.power = estimate_power_from_log(log, ppa, self.freq)
        return ppa

    def _parse_log(self, log: str) -> PPA:
        ppa = PPA()
        m = re.findall(r"Chip area for module '.*?':\s*([0-9.]+)", log)
        if m:
            ppa.area = float(m[-1])
        m = re.findall(r"Delay\s*=\s*([0-9.]+)\s*ps", log)
        if m:
            ppa.delay = float(m[-1]) / 1000.0
        m = re.findall(r"Number of cells:\s*(\d+)", log)
        if m:
            ppa.gates = int(m[-1])
        ppa.ok = ppa.area == ppa.area and ppa.delay == ppa.delay
        return ppa

    # ------------------------------------------------------------------
    def _run_sta(self, d: str, netlist: str, top: str, ppa: PPA) -> Optional[Dict]:
        tcl = os.path.join(d, "sta.tcl")
        period = max(ppa.delay * 1.05, 0.05)
        with open(tcl, "w") as fh:
            fh.write(f"""
read_liberty {self.lib}
read_verilog {netlist}
link_design {top}
create_clock -name vclk -period {period:.4f}
set_input_delay 0 -clock vclk [all_inputs]
set_output_delay 0 -clock vclk [all_outputs]
set_load 0.005 [all_outputs]
set_power_activity -input -activity 0.5 -duty 0.5
report_checks -path_delay max -digits 6 -format summary
report_power -digits 8
exit
""")
        try:
            r = subprocess.run([self.sta, "-no_init", "-no_splash", "-exit", tcl],
                               capture_output=True, text=True, timeout=600)
        except Exception:                                     # pragma: no cover
            return None
        txt = r.stdout + r.stderr
        res = {}
        m = re.search(r"data arrival time\s+([0-9.\-]+)", txt)
        if m:
            res["delay"] = abs(float(m.group(1)))
        else:
            m = re.search(r"^\s*([0-9.]+)\s+slack", txt, re.M)
        m = re.search(r"^Total\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+"
                      r"([0-9.eE+\-]+)\s+([0-9.eE+\-]+)", txt, re.M)
        if m:
            internal, switching, leakage, total = (float(x) for x in m.groups())
            # OpenSTA reports Watts; scale to the target frequency
            scale = self.freq / (1.0 / (period * 1e-9))
            res["power"] = total * 1e6 * scale
            res["leakage"] = leakage * 1e6
        return res or None


# --------------------------------------------------------------------------
# Fallback power model
# --------------------------------------------------------------------------
_CELL_RE = re.compile(r"^\s+(\d+)\s+(\S+)\s*$", re.M)


def estimate_power_from_log(log: str, ppa: PPA, freq: float) -> float:
    """Activity-based fallback: P ~ 0.5 a f C V^2 with C proportional to area."""
    # Nangate45: ~1.2 fF of switched capacitance per um^2 of standard cell area
    cap = 1.2e-15 * ppa.area
    activity = 0.2
    dyn = 0.5 * activity * freq * cap * VDD ** 2
    leak = 2.0e-9 * ppa.area          # ~2 nW/um^2 typical leakage
    return (dyn + leak) * 1e6         # uW


if __name__ == "__main__":                                    # pragma: no cover
    s = Synthesizer()
    print("yosys:", s.yosys, "sta:", s.sta)
    v = ("module mul(input [7:0] a, input [7:0] b, output [15:0] p);\n"
         "  assign p = a * b;\nendmodule\n")
    print(s.run(v).as_dict())
