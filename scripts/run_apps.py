"""Application-level evaluation of selected 8x8 approximate multipliers.

Two workloads, both driven by a 256x256 product look-up table extracted from
the *exact* functional model of each design (so the application numbers refer
to the very same circuit that was synthesised):

  1. image blending and Gaussian smoothing  -> PSNR / SSIM
  2. LeNet-5 inference on MNIST (8-bit post-training quantisation, every
     multiplication in the convolution and fully-connected layers replaced by
     the approximate one)  -> top-1 accuracy

The CNN is trained once on the GPU with exact arithmetic and then evaluated
with each approximate multiplier, which is the standard protocol for
approximate-multiplier accuracy studies.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approxlg.arch import Space                          # noqa: E402
from approxlg.simulate import Evaluator, simulate        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


# --------------------------------------------------------------------------
def build_lut(space: Space, ev: Evaluator, g) -> np.ndarray:
    """lut[a, b] = approximate product of the 8-bit operands a and b."""
    nl = ev.builder.build(np.asarray(g, dtype=np.int32))
    prod = simulate(nl, ev.stim)
    prod = np.asarray(getattr(prod, "get", lambda: prod)())
    return prod.reshape(256, 256).astype(np.int32)


def build_lut_ref(js: str, top: str, stim) -> np.ndarray:
    """The same table for a published design, taken from its mapped netlist.

    The reference designs go through exactly the gate-level evaluator that
    produced their accuracy and power numbers, so the application results and
    the PPA results describe the same circuit here too."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from approxlg.netjson import GateNetlist
    from approxlg.simulate import _unpack
    from run_libs import pick_ports

    gn = GateNetlist(js, top)
    ports = pick_ports(gn, 8)
    if ports is None:
        raise ValueError(f"{top}: unusable interface")
    a_p, b_p, p_p = ports
    zero = np.zeros(stim.nvec // 64, dtype=np.uint64)
    in_bits = {}
    for port, bits in ((a_p, stim.abits), (b_p, stim.bbits)):
        for j, netid in enumerate(gn.ports[port]["bits"]):
            in_bits[netid] = bits[j] if j < 8 else zero
    env = gn.evaluate(in_bits, stim.nvec, np)
    prod = np.zeros(stim.nvec, dtype=np.int64)
    for j, netid in enumerate(gn.ports[p_p]["bits"]):
        if netid in env:
            prod += _unpack(env[netid], stim.nvec, np).astype(np.int64) << j
    return prod.reshape(256, 256).astype(np.int32)


def amul(lut, x, y):
    return lut[x.astype(np.int32), y.astype(np.int32)]


# --------------------------------------------------------------------------
# image workloads
# --------------------------------------------------------------------------
def load_images():
    from skimage import data, color, transform
    ims = {}
    for nm, fn in (("camera", data.camera), ("coins", data.coins),
                   ("moon", data.moon), ("page", data.page),
                   ("astronaut", lambda: color.rgb2gray(data.astronaut())),
                   ("chelsea", lambda: color.rgb2gray(data.chelsea()))):
        im = np.asarray(fn())
        if im.dtype != np.uint8:
            im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        im = transform.resize(im, (256, 256), preserve_range=True,
                              anti_aliasing=True).astype(np.uint8)
        ims[nm] = im
    return ims


def blend(lut, x, y, alpha=0.6):
    ka = int(round(alpha * 255))
    kb = 255 - ka
    a = amul(lut, x, np.full_like(x, ka))
    b = amul(lut, y, np.full_like(y, kb))
    return np.clip((a + b) // 255, 0, 255).astype(np.uint8)


def _gauss_kernel(sigma=1.1, r=2):
    ax = np.arange(-r, r + 1)
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    k = np.round(g / g.sum() * 255).astype(np.int32)   # 8-bit weights
    return k, int(k.sum())


GAUSS, GSUM = _gauss_kernel()


def smooth(lut, x):
    r = GAUSS.shape[0] // 2
    pad = np.pad(x, r, mode="edge").astype(np.int32)
    acc = np.zeros(x.shape, dtype=np.int64)
    for dy in range(GAUSS.shape[0]):
        for dx in range(GAUSS.shape[1]):
            w = np.full(x.shape, GAUSS[dy, dx], dtype=np.int32)
            acc += amul(lut, pad[dy:dy + x.shape[0], dx:dx + x.shape[1]], w)
    return np.clip(acc // GSUM, 0, 255).astype(np.uint8)


def immul(lut, x, y):
    """Pixel-wise image multiplication, the workload that stresses the
    multiplier most directly."""
    return np.clip(amul(lut, x, y) // 255, 0, 255).astype(np.uint8)


def image_quality(lut_ref, lut, images):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    keys = list(images)
    out = {}
    for task in ("blend", "smooth", "immul"):
        ps, ss = [], []
        for i, k in enumerate(keys):
            x = images[k]
            y = images[keys[(i + 1) % len(keys)]]
            if task == "blend":
                r, a = blend(lut_ref, x, y), blend(lut, x, y)
            elif task == "smooth":
                r, a = smooth(lut_ref, x), smooth(lut, x)
            else:
                r, a = immul(lut_ref, x, y), immul(lut, x, y)
            mse = float(((r.astype(np.float64) - a) ** 2).mean())
            ps.append(99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse))
            ss.append(ssim(r, a, data_range=255))
        out[f"{task}_psnr"] = float(np.mean(ps))
        out[f"{task}_ssim"] = float(np.mean(ss))
    return out


# --------------------------------------------------------------------------
# CNN workload
# --------------------------------------------------------------------------
def cnn_eval(lut_list, names, device="cuda", epochs=3, batch=256, seed=0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor()])
    root = os.path.join(RESULTS, "mnist")
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    trl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True)
    tel = torch.utils.data.DataLoader(te, batch_size=512)

    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 6, 5, padding=2)
            self.c2 = nn.Conv2d(6, 16, 5)
            self.f1 = nn.Linear(16 * 5 * 5, 120)
            self.f2 = nn.Linear(120, 84)
            self.f3 = nn.Linear(84, 10)

        def forward(self, x, mul=None):
            x = F.max_pool2d(F.relu(qconv(self.c1, x, mul)), 2)
            x = F.max_pool2d(F.relu(qconv(self.c2, x, mul)), 2)
            x = x.flatten(1)
            x = F.relu(qlin(self.f1, x, mul))
            x = F.relu(qlin(self.f2, x, mul))
            return qlin(self.f3, x, mul)

    def quant(t, bits=8):
        s = t.abs().amax().clamp(min=1e-8) / (2 ** (bits - 1) - 1)
        return torch.round(t / s).clamp(-(2 ** (bits - 1) - 1),
                                        2 ** (bits - 1) - 1), s

    def qconv(layer, x, mul):
        if mul is None:
            return layer(x)
        wq, ws_ = quant(layer.weight)
        xq, xs_ = quant(x)
        k = layer.kernel_size[0]
        patches = F.unfold(xq, k, padding=layer.padding[0])       # B, C*k*k, L
        W = wq.reshape(wq.shape[0], -1)                           # O, C*k*k
        out = _lut_matmul(patches.transpose(1, 2), W.t(), mul)    # B, L, O
        out = out.transpose(1, 2)
        H = x.shape[-1] + 2 * layer.padding[0] - k + 1
        out = out.reshape(x.shape[0], wq.shape[0], H, H) * (ws_ * xs_)
        return out + layer.bias.reshape(1, -1, 1, 1)

    def qlin(layer, x, mul):
        if mul is None:
            return layer(x)
        wq, ws_ = quant(layer.weight)
        xq, xs_ = quant(x)
        out = _lut_matmul(xq, wq.t(), mul) * (ws_ * xs_)
        return out + layer.bias

    def _lut_matmul(A, B, mul):
        """A: (..., K) int-valued, B: (K, N) int-valued -> approximate A @ B."""
        sh = A.shape
        A = A.reshape(-1, sh[-1])
        Aa = A.abs().to(torch.long)
        As = torch.sign(A)
        Ba = B.abs().to(torch.long)
        Bs = torch.sign(B)
        out = torch.zeros(A.shape[0], B.shape[1], device=A.device)
        chunk = max(1, 2 ** 22 // max(1, A.shape[0] * B.shape[1] // 64))
        for k0 in range(0, sh[-1], chunk):
            k1 = min(sh[-1], k0 + chunk)
            p = mul[Aa[:, k0:k1].unsqueeze(2), Ba[k0:k1].unsqueeze(0)]
            s = (As[:, k0:k1].unsqueeze(2) * Bs[k0:k1].unsqueeze(0))
            out += (p.to(torch.float32) * s).sum(1)
        return out.reshape(*sh[:-1], B.shape[1])

    net = LeNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        net.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            opt.step()
        print(f"  epoch {ep}: loss {loss.item():.4f}")

    net.eval()
    accs = {}
    with torch.no_grad():
        for nm, lut in zip(names, lut_list):
            mul = None if lut is None else torch.as_tensor(
                lut, dtype=torch.float32, device=device)
            ok = tot = 0
            for x, y in tel:
                x, y = x.to(device), y.to(device)
                pred = net(x, mul).argmax(1)
                ok += int((pred == y).sum())
                tot += y.numel()
            accs[nm] = ok / tot
            print(f"  {nm:28s} acc={accs[nm]*100:.2f}%")
    return accs


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default=os.path.join(RESULTS, "app_designs.json"))
    ap.add_argument("--refs", default=None,
                    help="json of matched-accuracy published designs")
    ap.add_argument("--out", default=os.path.join(RESULTS, "apps.json"))
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    with open(args.designs) as fh:
        designs = json.load(fh)
    if args.refs and os.path.exists(args.refs):
        designs = designs + json.load(open(args.refs))
    sp = Space(8)
    ev = Evaluator(sp)
    lut_ref = build_lut(sp, ev, np.zeros(sp.length, dtype=np.int32))

    images = load_images()
    out = []
    luts, names = [], []
    for d in designs:
        lut = (build_lut(sp, ev, d["g"]) if "g" in d
               else build_lut_ref(d["json"], d["top"], ev.stim))
        r = dict(d)
        r.update(image_quality(lut_ref, lut, images))
        out.append(r)
        luts.append(lut)
        names.append(d["name"])
        print(f"{d['name']:8s} blend {r['blend_psnr']:6.2f}dB/{r['blend_ssim']:.4f}  "
              f"smooth {r['smooth_psnr']:6.2f}dB/{r['smooth_ssim']:.4f}  "
              f"immul {r['immul_psnr']:6.2f}dB/{r['immul_ssim']:.4f}")

    if args.cnn:
        accs = cnn_eval([None] + luts, ["exact"] + names, epochs=args.epochs)
        for r in out:
            r["cnn_acc"] = accs.get(r["name"])
        exact_acc = accs.get("exact")
        for r in out:
            r["cnn_acc_drop"] = (exact_acc - r["cnn_acc"]) if r.get("cnn_acc") \
                else None
        with open(os.path.join(RESULTS, "cnn_acc.json"), "w") as fh:
            json.dump(accs, fh, indent=1)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
