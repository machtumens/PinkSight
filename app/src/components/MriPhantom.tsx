import { useEffect, useRef } from "react";

/*
  Synthetic DCE phantom — NOT patient data, NOT a real MRI. A dependency-free stand-in for the
  Cornerstone3D / VTK.js volume a production build would mount, so the viewport reads like an imaging
  workstation without shipping a heavy viewer against data the demo does not have. The lesion's
  apparent enhancement rises with `phase` purely as a DISPLAY axis — never a kinetics / growth-rate
  claim (claim ledger). The XAI overlay mimics a Grad-CAM heatmap; the real pipeline IoU-validates it
  on a trained encoder. ponytail: hand-drawn canvas, no imaging dep, ~1 file.
*/
export function MriPhantom({ phase, xai }: { phase: number; xai: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const W = cv.width;
    const H = cv.height;
    const cx = W / 2;
    const cy = H / 2;
    const enh = phase / 5; // enhancement fraction across the 6 DCE phases (display axis only)

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#05070d";
    ctx.fillRect(0, 0, W, H);

    // breast tissue — soft radial body
    const tissue = ctx.createRadialGradient(cx, cy, 20, cx, cy, W * 0.5);
    tissue.addColorStop(0, "#4a5262");
    tissue.addColorStop(0.6, "#2a303c");
    tissue.addColorStop(1, "#0a0d14");
    ctx.fillStyle = tissue;
    ctx.beginPath();
    ctx.arc(cx, cy, W * 0.42, 0, Math.PI * 2);
    ctx.fill();

    // deterministic speckle texture (fixed seed → identical every render, no flicker)
    let seed = 1337;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    ctx.globalAlpha = 0.06;
    for (let i = 0; i < 1400; i++) {
      const a = rnd() * Math.PI * 2;
      const r = rnd() * W * 0.42;
      ctx.fillStyle = rnd() > 0.5 ? "#ffffff" : "#000000";
      ctx.fillRect(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 1.5, 1.5);
    }
    ctx.globalAlpha = 1;

    // lesion — brighter Gaussian blob, off-centre; brightness tracks the display enhancement
    const lx = cx + W * 0.12;
    const ly = cy - H * 0.08;
    const bright = 120 + Math.round(enh * 110); // 120..230
    const lesion = ctx.createRadialGradient(lx, ly, 2, lx, ly, W * 0.13);
    lesion.addColorStop(0, `rgb(${bright},${bright},${bright})`);
    lesion.addColorStop(1, "rgba(120,120,120,0)");
    ctx.fillStyle = lesion;
    ctx.beginPath();
    ctx.arc(lx, ly, W * 0.13, 0, Math.PI * 2);
    ctx.fill();

    // XAI Grad-CAM-style overlay over the lesion (warm, semi-transparent)
    if (xai) {
      const cam = ctx.createRadialGradient(lx, ly, 2, lx, ly, W * 0.16);
      cam.addColorStop(0, "rgba(214,51,108,0.55)"); // rose (primary signal)
      cam.addColorStop(0.5, "rgba(240,140,40,0.35)"); // amber
      cam.addColorStop(1, "rgba(240,140,40,0)");
      ctx.fillStyle = cam;
      ctx.beginPath();
      ctx.arc(lx, ly, W * 0.16, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [phase, xai]);

  return (
    <canvas
      ref={ref}
      width={384}
      height={384}
      className="max-h-full max-w-full rounded-md ring-1 ring-white/10"
    />
  );
}
