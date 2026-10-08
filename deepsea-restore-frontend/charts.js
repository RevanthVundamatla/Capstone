/* Lightweight inline-SVG line charts: no external libraries needed. */
(function () {
  const COLORS = ['#3FD8C8', '#E8935E', '#9db8ff', '#e6d36b'];

  function niceRange(vals, hline) {
    let lo = Math.min(...vals, hline ?? Infinity);
    let hi = Math.max(...vals, hline ?? -Infinity);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.1;
    return [lo - pad, hi + pad];
  }

  function lineChart(title, o) {
    const W = 520, H = 280, L = 52, R = 14, T = 12, B = 44;
    const all = o.series.flatMap((s) => s.ys);
    const [y0, y1] = niceRange(all, o.hline);
    const xs = o.xs, x0 = Math.min(...xs), x1 = Math.max(...xs);
    const X = (x) => L + (x1 === x0 ? (W - L - R) / 2 : ((x - x0) / (x1 - x0)) * (W - L - R));
    const Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
    let g = '';
    for (let i = 0; i <= 4; i++) {
      const v = y0 + ((y1 - y0) * i) / 4, y = Y(v);
      g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="rgba(234,243,241,.1)"/>` +
           `<text x="${L - 6}" y="${y + 4}" text-anchor="end" font-size="11" fill="#8FA9A6">${v.toFixed(o.dec ?? 2)}</text>`;
    }
    const tickXs = xs.length > 6 ? xs.filter((_, i) => i % Math.ceil(xs.length / 6) === 0) : xs;
    tickXs.forEach((x) => {
      g += `<text x="${X(x)}" y="${H - B + 16}" text-anchor="middle" font-size="11" fill="#8FA9A6">${o.xfmt ? o.xfmt(x) : x}</text>`;
    });
    if (o.hline != null) {
      g += `<line x1="${L}" x2="${W - R}" y1="${Y(o.hline)}" y2="${Y(o.hline)}" stroke="#E8935E" stroke-dasharray="5 4"/>` +
           `<text x="${W - R}" y="${Y(o.hline) - 5}" text-anchor="end" font-size="11" fill="#E8935E">${o.hlineLabel || o.hline}</text>`;
    }
    o.series.forEach((s, i) => {
      const c = COLORS[i % COLORS.length];
      const pts = xs.map((x, k) => `${X(x)},${Y(s.ys[k])}`).join(' ');
      g += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="2"/>`;
      xs.forEach((x, k) => { g += `<circle cx="${X(x)}" cy="${Y(s.ys[k])}" r="${xs.length > 40 ? 0 : 3}" fill="${c}"/>`; });
    });
    g += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="#8FA9A6">${o.xlabel}</text>`;
    g += `<text transform="translate(13 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#8FA9A6">${o.ylabel}</text>`;
    const legend = o.series.length > 1
      ? `<p class="legend" style="float:none;margin:4px 0 0">${o.series.map((s, i) => `<span style="color:${COLORS[i % 4]}">● ${s.name}</span>`).join(' &nbsp; ')}</p>` : '';
    return `<div class="chart-card"><h3>${title}</h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">${g}</svg>${legend}</div>`;
  }

  /* ---- per-image graphs: input -> pass 1 -> pass 2 -> pass 3 ---- */
  function renderPasses(result) {
    const el = document.getElementById('passCharts');
    if (!el || !result.steps || !result.steps.length) return;
    const init = result.initial_metrics || {};
    const steps = result.steps;
    const xs = [0, ...steps.map((s) => s.step)];
    const defs = [['psnr', 'PSNR (dB)', 2], ['ssim', 'SSIM', 3], ['uiqm', 'UIQM', 2], ['uciqe', 'UCIQE', 3]];
    let html = '';
    defs.forEach(([k, label, dec]) => {
      const ys = [init[k], ...steps.map((s) => s.metrics && s.metrics[k])];
      if (ys.some((v) => v == null || Number.isNaN(Number(v)))) return;   // e.g. PSNR without a reference
      html += lineChart(label, { xs, series: [{ name: label, ys: ys.map(Number) }], dec,
        xlabel: 'Pass (0 = input)', ylabel: label, xfmt: (x) => (x === 0 ? 'in' : x) });
    });
    el.innerHTML = html || '<p class="empty">No per-pass metrics were returned.</p>';
  }

  /* ---- training graphs from training_history.json ---- */
  async function loadHistory() {
    const el = document.getElementById('trainCharts');
    if (!el) return;
    try {
      const r = await fetch('training_history.json', { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      const h = await r.json();
      const v = h.validation || [], rw = h.reward || [];
      if (!v.length && !rw.length) throw new Error('empty');
      const epe = h.episodes_per_epoch;
      const ep = (a) => a.map((d) => d.episode);
      const xlabel = epe ? `Training episode (1 epoch ≈ ${epe} episodes)` : 'Training episode';
      let html = '';
      if (v.length) {
        html += lineChart('Validation PSNR vs episode', { xs: ep(v), series: [{ name: 'PSNR', ys: v.map((d) => d.psnr) }],
          xlabel, ylabel: 'PSNR (dB)', hline: 30, hlineLabel: '30 dB target', dec: 1 });
        html += lineChart('Validation SSIM vs episode', { xs: ep(v), series: [{ name: 'SSIM', ys: v.map((d) => d.ssim) }],
          xlabel, ylabel: 'SSIM', dec: 3 });
        html += lineChart('Validation UIQM & UCIQE vs episode', { xs: ep(v),
          series: [{ name: 'UIQM', ys: v.map((d) => d.uiqm) }, { name: 'UCIQE', ys: v.map((d) => d.uciqe) }],
          xlabel, ylabel: 'Score', dec: 2 });
      }
      if (rw.length) {
        html += lineChart('Average training reward vs episode', { xs: ep(rw), series: [{ name: 'Reward', ys: rw.map((d) => d.value) }],
          xlabel, ylabel: 'Avg episode reward', dec: 3 });
      }
      el.innerHTML = html;
    } catch (e) {
      el.innerHTML = '<p class="empty">No training history found yet. After training, run ' +
        '<code>python plot_results.py train_log.txt</code> (or just train with the updated train.py) ' +
        'to create <code>training_history.json</code> next to index.html.</p>';
    }
  }

  window.DeepSeaCharts = { renderPasses, loadHistory };
  document.addEventListener('DOMContentLoaded', loadHistory);
})();
