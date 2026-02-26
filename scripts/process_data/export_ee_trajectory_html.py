#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path


def _load_states(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{path} is not a list")
    return data


def _extract_xyz(records):
    t, x, y, z = [], [], [], []
    for item in records:
        if not isinstance(item, dict):
            continue
        ts = item.get("timestamp")
        payload = item.get("data")
        if not isinstance(ts, (int, float)) or not isinstance(payload, dict):
            continue
        pos = payload.get("pos")
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 3):
            continue
        tsf = float(ts)
        xf = float(pos[0])
        yf = float(pos[1])
        zf = float(pos[2])
        if not (math.isfinite(tsf) and math.isfinite(xf) and math.isfinite(yf) and math.isfinite(zf)):
            continue
        t.append(tsf)
        x.append(xf)
        y.append(yf)
        z.append(zf)
    if not t:
        raise RuntimeError("No valid end-effector points in states.pkl")
    t0 = t[0]
    t = [v - t0 for v in t]
    return t, x, y, z


def _validate_series(t, x, y, z):
    errors = []
    warnings = []

    n = len(t)
    if n < 30:
        errors.append(f"Too few valid samples: {n} (<30)")
        return errors, warnings

    duration = t[-1] - t[0] if n > 1 else 0.0
    if duration <= 0.0:
        errors.append("Non-positive duration detected")
        return errors, warnings
    if duration < 1.0:
        warnings.append(f"Short duration: {duration:.3f}s")

    dts = [t[i] - t[i - 1] for i in range(1, n)]
    non_increasing = [dt for dt in dts if dt <= 0]
    if non_increasing:
        ratio = len(non_increasing) / max(1, len(dts))
        errors.append(f"Timestamp is not strictly increasing: {len(non_increasing)}/{len(dts)} ({ratio:.2%})")
        return errors, warnings

    fps_est = (n - 1) / duration
    if not math.isfinite(fps_est) or fps_est <= 0:
        errors.append("Invalid estimated FPS")
        return errors, warnings
    if fps_est < 5 or fps_est > 300:
        warnings.append(f"Estimated FPS is unusual: {fps_est:.2f}")

    x_span = max(x) - min(x)
    y_span = max(y) - min(y)
    z_span = max(z) - min(z)
    path_span = math.sqrt(x_span * x_span + y_span * y_span + z_span * z_span)
    if path_span < 1e-4:
        warnings.append("Trajectory span is very small; motion may be nearly static")

    max_step = 0.0
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dz = z[i] - z[i - 1]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > max_step:
            max_step = d
    if max_step > 0.25:
        warnings.append(f"Large single-step jump detected: {max_step:.4f}m")

    return errors, warnings


def _build_html(title: str, points_json: str, stats_json: str):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121a31;
      --line: #1f2a4a;
      --text: #e5ecff;
      --muted: #95a4cc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 20% 10%, #1a2754 0%, var(--bg) 45%);
      color: var(--text);
      font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    }}
    .wrap {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 20px;
    }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .title {{
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
      box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    }}
    canvas {{
      width: 100%;
      height: 70vh;
      min-height: 420px;
      display: block;
      cursor: grab;
      background:
        radial-gradient(circle at 70% 30%, rgba(57, 123, 255, 0.11) 0%, rgba(57,123,255,0.01) 35%, rgba(0,0,0,0) 65%),
        linear-gradient(180deg, #0e1630 0%, #0b1020 100%);
    }}
    canvas.dragging {{ cursor: grabbing; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px;
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .meta .item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: var(--panel);
    }}
    .hint {{
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="title">{title}</div>
      <div class="controls">
        <label><input id="autoRotate" type="checkbox" checked /> auto rotate</label>
        <button id="resetBtn">reset view</button>
      </div>
    </div>
    <div class="card">
      <canvas id="view"></canvas>
    </div>
    <div class="meta" id="meta"></div>
    <div class="hint">Drag: rotate | Wheel: zoom | Double click: reset</div>
  </div>
  <script>
    const points = {points_json};
    const stats = {stats_json};
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const autoRotateEl = document.getElementById("autoRotate");
    const resetBtn = document.getElementById("resetBtn");
    const metaEl = document.getElementById("meta");

    const metaItems = [
      ["samples", stats.samples],
      ["duration_s", stats.duration_s.toFixed(3)],
      ["fps_est", stats.fps_est.toFixed(2)],
      ["x_range", `[${{stats.x_min.toFixed(3)}}, ${{stats.x_max.toFixed(3)}}]`],
      ["y_range", `[${{stats.y_min.toFixed(3)}}, ${{stats.y_max.toFixed(3)}}]`],
      ["z_range", `[${{stats.z_min.toFixed(3)}}, ${{stats.z_max.toFixed(3)}}]`],
    ];
    metaEl.innerHTML = metaItems.map(([k, v]) => `<div class="item"><b>${{k}}</b><div>${{v}}</div></div>`).join("");

    function bounds(arr) {{
      let mn = Infinity, mx = -Infinity;
      for (const v of arr) {{ if (v < mn) mn = v; if (v > mx) mx = v; }}
      return [mn, mx];
    }}

    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const zs = points.map(p => p[2]);
    const [xMin, xMax] = bounds(xs);
    const [yMin, yMax] = bounds(ys);
    const [zMin, zMax] = bounds(zs);
    const cx = 0.5 * (xMin + xMax);
    const cy = 0.5 * (yMin + yMax);
    const cz = 0.5 * (zMin + zMax);
    const span = Math.max(xMax - xMin, yMax - yMin, zMax - zMin, 1e-6);
    const normalized = points.map((p, i) => [ (p[0]-cx)/span, (p[1]-cy)/span, (p[2]-cz)/span, i/(points.length-1||1) ]);

    let yaw = 0.85;
    let pitch = 0.35;
    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function resetView() {{
      yaw = 0.85;
      pitch = 0.35;
      zoom = 1.0;
      panX = 0;
      panY = 0;
    }}
    resetBtn.onclick = () => resetView();

    function resize() {{
      const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function rotate(p) {{
      const [x, y, z] = p;
      const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);

      const x1 = cyaw * x + syaw * z;
      const z1 = -syaw * x + cyaw * z;
      const y2 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;
      return [x1, y2, z2];
    }}

    function project(p, w, h) {{
      const [x, y, z] = rotate(p);
      const focal = 1.35;
      const k = focal / (focal + z * 0.9 + 1.3);
      const scale = 0.78 * Math.min(w, h) * zoom;
      return [w * 0.5 + x * scale * k + panX, h * 0.52 - y * scale * k + panY, z];
    }}

    function drawAxes(w, h) {{
      const axis = 0.65;
      const origin = project([0, 0, 0], w, h);
      const xEnd = project([axis, 0, 0], w, h);
      const yEnd = project([0, axis, 0], w, h);
      const zEnd = project([0, 0, axis], w, h);
      const draw = (to, color, label) => {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(origin[0], origin[1]);
        ctx.lineTo(to[0], to[1]);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillText(label, to[0] + 4, to[1] - 4);
      }};
      draw(xEnd, "#ef4444", "X");
      draw(yEnd, "#22c55e", "Y");
      draw(zEnd, "#3b82f6", "Z");
    }}

    function draw() {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = "rgba(255,255,255,0.03)";
      for (let i = 0; i < 12; i++) {{
        const y = (h / 12) * i;
        ctx.fillRect(0, y, w, 1);
      }}

      drawAxes(w, h);

      ctx.lineWidth = 2.0;
      for (let i = 0; i < normalized.length - 1; i++) {{
        const a = project(normalized[i], w, h);
        const b = project(normalized[i + 1], w, h);
        const hue = 210 + normalized[i][3] * 120;
        ctx.strokeStyle = `hsl(${{hue}}, 90%, 62%)`;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }}

      const s = project(normalized[0], w, h);
      const e = project(normalized[normalized.length - 1], w, h);
      ctx.fillStyle = "#22c55e";
      ctx.beginPath(); ctx.arc(s[0], s[1], 4.2, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#ef4444";
      ctx.beginPath(); ctx.arc(e[0], e[1], 4.2, 0, Math.PI * 2); ctx.fill();

      ctx.fillStyle = "#dbe7ff";
      ctx.font = "12px monospace";
      ctx.fillText("start", s[0] + 8, s[1] - 6);
      ctx.fillText("end", e[0] + 8, e[1] - 6);
    }}

    canvas.addEventListener("mousedown", (ev) => {{
      dragging = true;
      lastX = ev.clientX;
      lastY = ev.clientY;
      canvas.classList.add("dragging");
    }});
    window.addEventListener("mouseup", () => {{
      dragging = false;
      canvas.classList.remove("dragging");
    }});
    window.addEventListener("mousemove", (ev) => {{
      if (!dragging) return;
      const dx = ev.clientX - lastX;
      const dy = ev.clientY - lastY;
      yaw += dx * 0.008;
      pitch += dy * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch));
      lastX = ev.clientX;
      lastY = ev.clientY;
      draw();
    }});
    canvas.addEventListener("wheel", (ev) => {{
      ev.preventDefault();
      const scale = ev.deltaY < 0 ? 1.08 : 0.92;
      zoom *= scale;
      zoom = Math.max(0.3, Math.min(4.0, zoom));
      draw();
    }}, {{ passive: false }});
    canvas.addEventListener("dblclick", () => {{
      resetView();
      draw();
    }});

    window.addEventListener("resize", () => {{
      resize();
      draw();
    }});
    resize();
    draw();

    function loop() {{
      if (autoRotateEl.checked && !dragging) {{
        yaw += 0.0025;
        draw();
      }}
      requestAnimationFrame(loop);
    }}
    loop();
  </script>
</body>
</html>
"""


def build_html_for_demo(demo_dir: Path, output: Path):
    states_path = demo_dir / "states.pkl"
    if not states_path.exists():
        raise FileNotFoundError(f"states.pkl not found under {demo_dir}")

    t, x, y, z = _extract_xyz(_load_states(states_path))
    errors, warnings = _validate_series(t, x, y, z)
    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors))

    duration = t[-1] - t[0] if len(t) > 1 else 0.0
    fps_est = (len(t) - 1) / duration if duration > 1e-9 else 0.0

    points = [[x[i], y[i], z[i], t[i]] for i in range(len(t))]
    stats = {
        "samples": len(t),
        "duration_s": duration,
        "fps_est": fps_est,
        "x_min": min(x),
        "x_max": max(x),
        "y_min": min(y),
        "y_max": max(y),
        "z_min": min(z),
        "z_max": max(z),
        "validation_ok": True,
        "validation_warnings": warnings,
    }

    title = f"EE Trajectory 3D - {demo_dir.name}"
    html = _build_html(
        title=title,
        points_json=json.dumps(points, separators=(",", ":")),
        stats_json=json.dumps(stats, separators=(",", ":")),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    if warnings:
        print("[VALIDATION][WARN] " + " | ".join(warnings))
    else:
        print("[VALIDATION] OK")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Export end-effector 3D trajectory to an interactive HTML.")
    parser.add_argument("--demo-dir", required=True, help="Path to demo directory that contains states.pkl")
    parser.add_argument("--output", default=None, help="Output html path. Default: <demo-dir>/ee_trajectory_3d.html")
    return parser.parse_args()


def main():
    args = parse_args()
    demo_dir = Path(args.demo_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else demo_dir / "ee_trajectory_3d.html"
    out = build_html_for_demo(demo_dir, output)
    print(out)


if __name__ == "__main__":
    main()
