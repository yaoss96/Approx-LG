"""Cross-validate the bit-parallel functional model against RTL simulation.

For a set of random design points we emit Verilog, run Icarus Verilog over the
entire 8x8 input space, and compare every product against the value produced by
the packed-bit simulator used inside the search loop.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from approxlg.arch import Space  # noqa: E402
from approxlg.simulate import Evaluator, simulate  # noqa: E402

IVERILOG = os.environ.get("IVERILOG", "iverilog")
VVP = os.environ.get("VVP", "vvp")

TB = """
`timescale 1ns/1ps
module tb;
  reg [{nm1}:0] a, b;
  wire [{mm1}:0] p;
  integer i, j, f;
  mul dut(.a(a), .b(b), .p(p));
  initial begin
    f = $fopen("{out}", "w");
    for (i = 0; i < {np}; i = i + 1) begin
      for (j = 0; j < {np}; j = j + 1) begin
        a = i[{nm1}:0]; b = j[{nm1}:0];
        #1;
        $fwrite(f, "%0d\\n", p);
      end
    end
    $fclose(f);
    $finish;
  end
endmodule
"""

TB_RAND = """
`timescale 1ns/1ps
module tb;
  reg [{nm1}:0] a, b;
  wire [{mm1}:0] p;
  integer i, f;
  reg [{nm1}:0] va [0:{nv1}];
  reg [{nm1}:0] vb [0:{nv1}];
  mul dut(.a(a), .b(b), .p(p));
  initial begin
    $readmemh("{af}", va);
    $readmemh("{bf}", vb);
    f = $fopen("{out}", "w");
    for (i = 0; i <= {nv1}; i = i + 1) begin
      a = va[i]; b = vb[i]; #1;
      $fwrite(f, "%0d\\n", p);
    end
    $fclose(f);
    $finish;
  end
endmodule
"""


def check(n=8, ndesign=12, seed=1, signed=False, nvec=None):
    sp = Space(n, signed=signed)
    ev = Evaluator(sp, nvec=nvec)
    rng = np.random.default_rng(seed)
    tmp = tempfile.mkdtemp(prefix="approxlg_rtl_")
    ok = 0
    for d in range(ndesign):
        g = sp.random(rng) if d else np.zeros(sp.length, dtype=np.int32)
        nl = ev.builder.build(g)
        ref = simulate(nl, ev.stim)
        ref = np.asarray(ref).reshape(-1)
        vf = os.path.join(tmp, f"d{d}.v")
        of = os.path.join(tmp, f"d{d}.txt").replace("\\", "/")
        with open(vf, "w") as fh:
            fh.write(nl.to_verilog())
            if nvec is None:
                fh.write(TB.format(nm1=n - 1, mm1=2 * n - 1, np=1 << n, out=of))
            else:
                af = os.path.join(tmp, "a.hex").replace("\\", "/")
                bf = os.path.join(tmp, "b.hex").replace("\\", "/")
                np.savetxt(af, np.asarray(ev.stim.a), fmt="%x")
                np.savetxt(bf, np.asarray(ev.stim.b), fmt="%x")
                fh.write(TB_RAND.format(nm1=n - 1, mm1=2 * n - 1,
                                        nv1=ev.stim.nvec - 1, af=af, bf=bf,
                                        out=of))
        exe = os.path.join(tmp, f"d{d}.vvp")
        subprocess.run([IVERILOG, "-o", exe, vf], check=True,
                       capture_output=True)
        subprocess.run([VVP, exe], check=True, capture_output=True)
        rtl = np.loadtxt(of, dtype=np.int64)
        if rtl.shape != ref.shape:
            print(f"design {d}: SHAPE MISMATCH {rtl.shape} vs {ref.shape}")
            continue
        bad = int((rtl != ref).sum())
        status = "OK " if bad == 0 else f"FAIL({bad})"
        print(f"design {d:2d}: T={g[0]:2d} A={g[1]:2d} k={g[3]:2d} "
              f"cells={nl.n_cells:3d} -> {status}")
        ok += bad == 0
    print(f"\n{ok}/{ndesign} designs bit-exact against Icarus Verilog")
    return ok == ndesign


if __name__ == "__main__":
    nd = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nv = None if n <= 8 else 4096
    ok = True
    for sg in (False, True):
        print(f"--- {'signed' if sg else 'unsigned'} {n}x{n} ---")
        ok &= check(n=n, ndesign=nd, signed=sg, nvec=nv)
    sys.exit(0 if ok else 1)
