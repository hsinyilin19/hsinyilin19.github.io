// Hero: tracer particles advected by a = grad^perp psi on the torus T^2,
// psi = random Fourier series (the setting of thesis Ch. 3, "unmixing on T^2").
(function () {
  const canvas = document.getElementById('flow');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const btn = document.getElementById('flow-toggle');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const TAU = 2 * Math.PI;

  // seeded RNG so the "figure" is the same on every visit
  let seed = 20200709;
  const rand = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(TAU * rand());

  // psi(x,y) = sum a_k cos(kx x + ky y + p_k), |k|_inf <= 4, a_k ~ N(0,1)/|k|^2.2
  const modes = [];
  for (let kx = -4; kx <= 4; kx++) for (let ky = -4; ky <= 4; ky++) {
    if (kx === 0 && ky === 0) continue;
    modes.push([kx, ky, gauss() / Math.pow(kx * kx + ky * ky, 1.1), TAU * rand()]);
  }

  // tabulate psi and the velocity (u,w) = (d_y psi, -d_x psi) on a G x G grid of the torus
  const G = 192, PSI = new Float32Array(G * G), U = new Float32Array(G * G), W = new Float32Array(G * G);
  let pmax = 0, vmax = 0;
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const x = TAU * i / G, y = TAU * j / G;
    let p = 0, u = 0, w = 0;
    for (const [kx, ky, a, ph] of modes) {
      const s = Math.sin(kx * x + ky * y + ph);
      p += a * Math.cos(kx * x + ky * y + ph); u += -a * ky * s; w += a * kx * s;
    }
    const n = j * G + i; PSI[n] = p; U[n] = u; W[n] = w;
    pmax = Math.max(pmax, Math.abs(p)); vmax = Math.max(vmax, Math.hypot(u, w));
  }
  const sample = (A, x, y) => {           // bilinear, periodic; x,y in [0,2pi)
    const gx = (x / TAU) * G, gy = (y / TAU) * G;
    const i0 = Math.floor(gx), j0 = Math.floor(gy), fx = gx - i0, fy = gy - j0;
    const i = ((i0 % G) + G) % G, j = ((j0 % G) + G) % G, i1 = (i + 1) % G, j1 = (j + 1) % G;
    return (A[j * G + i] * (1 - fx) + A[j * G + i1] * fx) * (1 - fy) + (A[j1 * G + i] * (1 - fx) + A[j1 * G + i1] * fx) * fy;
  };

  // plasma-like colour ramp keyed to psi
  const stops = [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]];
  const color = (t, a) => {
    t = Math.min(0.999, Math.max(0, t)) * (stops.length - 1);
    const k = Math.floor(t), f = t - k, c0 = stops[k], c1 = stops[k + 1];
    return `rgba(${c0.map((c, m) => Math.round(c + (c1[m] - c) * f)).join(',')},${a})`;
  };

  let W_, H_, dpr, L, parts = [], running = !reduce, raf = null, visible = true;
  const N = () => Math.round(Math.min(2600, (W_ * H_) / 420));

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W_ = canvas.clientWidth; H_ = canvas.clientHeight;
    canvas.width = W_ * dpr; canvas.height = H_ * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    L = H_ / (1.35 * TAU);                      // pixels per unit length; ~1.35 periods tall
    parts = Array.from({ length: N() }, spawn);
    ctx.fillStyle = '#120f26'; ctx.fillRect(0, 0, W_, H_);
    drawContours();
    if (!running) still();
  }
  function spawn() { return { x: rand() * W_ / L, y: rand() * H_ / L, age: Math.floor(rand() * 400) }; }

  function drawContours() {                   // faint level sets of psi, drawn once
    const step = 6;
    ctx.lineWidth = 1;
    for (let py = 0; py < H_; py += step) for (let px = 0; px < W_; px += step) {
      const v = sample(PSI, ((px / L) % TAU + TAU) % TAU, ((py / L) % TAU + TAU) % TAU) / pmax;
      const band = Math.abs((v * 9) - Math.round(v * 9));
      if (band < 0.06) { ctx.fillStyle = color(0.5 + 0.5 * v, 0.16); ctx.fillRect(px, py, 1.4, 1.4); }
    }
  }

  function advance(p, dt) {                   // midpoint (RK2) step of dX/dt = a(X)
    const wrap = z => ((z % TAU) + TAU) % TAU;
    const x = wrap(p.x), y = wrap(p.y);
    const u1 = sample(U, x, y) / vmax, w1 = sample(W, x, y) / vmax;
    const xm = wrap(x + 0.5 * dt * u1), ym = wrap(y + 0.5 * dt * w1);
    p.x += dt * sample(U, xm, ym) / vmax; p.y += dt * sample(W, xm, ym) / vmax;
  }

  function frame() {
    ctx.fillStyle = 'rgba(18,15,38,0.06)'; ctx.fillRect(0, 0, W_, H_);
    ctx.lineWidth = 1.35;
    for (const p of parts) {
      const x0 = p.x * L, y0 = p.y * L;
      advance(p, 0.018);
      p.age++;
      const x1 = p.x * L, y1 = p.y * L;
      if (p.age > 520 || x1 < -4 || y1 < -4 || x1 > W_ + 4 || y1 > H_ + 4) { Object.assign(p, spawn(), { age: 0 }); continue; }
      const v = sample(PSI, ((p.x % TAU) + TAU) % TAU, ((p.y % TAU) + TAU) % TAU) / pmax;
      ctx.strokeStyle = color(0.5 + 0.5 * v, 0.85);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    raf = running && visible ? requestAnimationFrame(frame) : null;
  }
  function still() {                          // static figure for reduced motion / paused
    ctx.lineWidth = 0.8;
    for (let n = 0; n < Math.min(900, parts.length); n++) {
      const p = spawn(); ctx.beginPath(); ctx.moveTo(p.x * L, p.y * L);
      const v = sample(PSI, ((p.x % TAU) + TAU) % TAU, ((p.y % TAU) + TAU) % TAU) / pmax;
      for (let s = 0; s < 70; s++) { advance(p, 0.03); ctx.lineTo(p.x * L, p.y * L); }
      ctx.strokeStyle = color(0.5 + 0.5 * v, 0.35); ctx.stroke();
    }
  }
  function setRunning(on) {
    running = on;
    if (btn) { btn.textContent = on ? 'Pause flow' : 'Play flow'; btn.setAttribute('aria-pressed', String(!on)); }
    if (on && !raf && visible) raf = requestAnimationFrame(frame);
  }
  if (btn) btn.addEventListener('click', () => setRunning(!running));
  new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && running && !raf) raf = requestAnimationFrame(frame); }).observe(canvas);
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(resize, 150); });
  resize(); setRunning(running);
})();

// Publications filter
(function () {
  const bar = document.querySelector('.filters'); if (!bar) return;
  const items = [...document.querySelectorAll('.pub')], count = document.getElementById('pub-count');
  bar.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    bar.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    const f = b.dataset.f; let n = 0;
    items.forEach(li => { const show = f === 'all' || li.dataset.t.split(' ').includes(f); li.hidden = !show; n += show; });
    if (count) count.textContent = `${n} of ${items.length}`;
  });
})();

// Theme toggle (remembered per browser) and current-section highlight in the nav
(function () {
  const root = document.documentElement, btn = document.getElementById('theme');
  try { const t = localStorage.getItem('theme'); if (t) root.dataset.theme = t; } catch (e) {}
  const isDark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  const label = () => { if (btn) btn.textContent = isDark() ? 'Light' : 'Dark'; };
  label();
  if (btn) btn.addEventListener('click', () => { root.dataset.theme = isDark() ? 'light' : 'dark'; try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {} label(); });

  const links = [...document.querySelectorAll('.nav a[href^="#"]')];
  const map = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const navObs = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => a.setAttribute('aria-current', String(a === map.get(e.target.id))));
  }), { rootMargin: '-45% 0px -50% 0px' });
  map.forEach((a, id) => { const el = document.getElementById(id); if (el) navObs.observe(el); });
})();
