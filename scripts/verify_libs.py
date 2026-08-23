"""Cross-validate the gate-level evaluation of every third-party design
against Icarus Verilog RTL simulation.

A published design enters the comparison only if the product it produces in our
packed-bit gate-level simulator matches, bit for bit, the product produced by
an event-driven RTL simulation of the original sources on the same random
operands.  Designs that do not elaborate, that leave the product at X, or that
disagree are reported and excluded.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from approxlg.netjson import GateNetlist                      # noqa: E402
from approxlg.simulate import _unpack, make_stimulus          # noqa: E402
from run_libs import RESULTS, discover, pick_ports            # noqa: E402

IVERILOG = os.environ.get("IVERILOG", "iverilog")
VVP = os.environ.get("VVP", "vvp")
NVEC = 2048

TB = """
`timescale 1ns/1ps
module tb;
  reg [{nm1}:0] A, B;
  wire [{pm1}:0] P;
  integer i, f;
  reg [{nm1}:0] va [0:{nv1}];
  reg [{nm1}:0] vb [0:{nv1}];
  {top} dut(.{ap}(A), .{bp}(B), .{pp}(P));
  initial begin
    $readmemh("{af}", va);
    $readmemh("{bf}", vb);
    f = $fopen("{out}", "w");
    for (i = 0; i <= {nv1}; i = i + 1) begin
      A = va[i]; B = vb[i];
      #1;
      $fwrite(f, "%0d\\n", P);
    end
    $fclose(f);
    $finish;
  end
endmodule
"""


def main():
    import argparse
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n", type=int, default=16)
    n = ap_.parse_args().n
    stim = make_stimulus(n, NVEC, gpu=False, seed=99)
    wd = os.path.join(RESULTS, "libjson")
    tmp = tempfile.mkdtemp(prefix="approxlg_libv_")
    out = []
    prev = {}
    vp = os.path.join(RESULTS, "libs_verify.json" if n == 16
                      else f"libs_verify_n{n}.json")
    if os.path.exists(vp):
        prev = {r["name"]: r for r in json.load(open(vp))}
    for d in discover(n):
        if d["name"] in prev:
            out.append(prev[d["name"]])
            continue
        js = os.path.join(wd, f"{d['name']}.json")
        rec = {"name": d["name"], "lib": d["lib"]}
        if not os.path.exists(js):
            rec.update(status="no-netlist")
            out.append(rec)
            continue
        try:
            gn = GateNetlist(js, d["top"])
            ports = pick_ports(gn, n)
            if ports is None:
                rec.update(status="interface")
                out.append(rec)
                continue
            ap, bp, pp = ports
            zero = np.zeros(stim.nvec // 64, dtype=np.uint64)
            in_bits = {}
            for port, bits in ((ap, stim.abits), (bp, stim.bbits)):
                for j, net in enumerate(gn.ports[port]["bits"]):
                    in_bits[net] = bits[j] if j < n else zero
            env = gn.evaluate(in_bits, stim.nvec, np)
            ref = np.zeros(stim.nvec, dtype=np.int64)
            for j, net in enumerate(gn.ports[pp]["bits"]):
                if net in env:
                    ref += _unpack(env[net], stim.nvec, np).astype(np.int64) << j

            base = os.path.join(tmp, d["name"])
            af, bf = base + "_a.hex", base + "_b.hex"
            np.savetxt(af, np.asarray(stim.a), fmt="%x")
            np.savetxt(bf, np.asarray(stim.b), fmt="%x")
            tb = TB.format(nm1=n - 1, pm1=len(gn.ports[pp]["bits"]) - 1,
                           nv1=stim.nvec - 1, top=d["top"], ap=ap, bp=bp, pp=pp,
                           af=af.replace("\\", "/"), bf=bf.replace("\\", "/"),
                           out=(base + "_o.txt").replace("\\", "/"))
            tbf = base + "_tb.v"
            with open(tbf, "w") as fh:
                fh.write(tb)
            cmd = [IVERILOG, "-g2005", "-o", base + ".vvp"]
            for inc in d["incdirs"]:
                cmd += ["-I", os.path.abspath(inc).replace("\\", "/")]
            cmd += [os.path.abspath(f).replace("\\", "/") for f in d["files"]]
            cmd += [tbf]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                rec.update(status="elaboration", msg=r.stderr[-200:])
                out.append(rec)
                continue
            subprocess.run([VVP, base + ".vvp"], capture_output=True,
                           text=True, timeout=1800)
            of = base + "_o.txt"
            if not os.path.exists(of):
                rec.update(status="no-output")
                out.append(rec)
                continue
            txt = open(of).read().split()
            if any("x" in t.lower() or "z" in t.lower() for t in txt):
                rec.update(status="undefined-output")
                out.append(rec)
                continue
            rtl = np.array([int(t) for t in txt], dtype=np.int64)
            if rtl.shape != ref.shape:
                rec.update(status="length", msg=f"{rtl.shape} vs {ref.shape}")
                out.append(rec)
                continue
            bad = int((rtl != ref).sum())
            rec.update(status="match" if bad == 0 else "mismatch", bad=bad)
        except Exception as e:
            rec.update(status="error", msg=f"{type(e).__name__}: {e}")
        out.append(rec)
        print(f"{rec['lib']:10s} {rec['name']:18s} {rec['status']}"
              + (f" ({rec.get('bad')})" if rec.get("bad") else ""))
    with open(vp, "w") as fh:
        json.dump(out, fh, indent=1)
    ok = sum(1 for r in out if r["status"] == "match")
    print(f"\n{ok}/{len(out)} designs reproduce the RTL exactly")


if __name__ == "__main__":
    main()
