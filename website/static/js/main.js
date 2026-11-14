(function () {
  'use strict';

  // ---------- Occupancy-constrained allocation (Sec. 3.3-3.4) ----------
  var EPS = 1e-6;
  var ITERS = 24;

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  function meanSigmoid(scores, b) {
    var sum = 0;
    for (var i = 0; i < scores.length; i++) sum += sigmoid(scores[i] + b);
    return sum / scores.length;
  }

  // Bisection on b with the paper's bracket [-32 - max s, 32 - min s].
  function solveOffset(scores, p) {
    var target = Math.min(Math.max(p, EPS), 1 - EPS);
    var lo = -32 - Math.max.apply(null, scores);
    var hi = 32 - Math.min.apply(null, scores);
    for (var k = 0; k < ITERS; k++) {
      var mid = 0.5 * (lo + hi);
      if (meanSigmoid(scores, mid) < target) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // Magma-like colour map for probabilities in [0, 1].
  var STOPS = [
    [0.0, [0, 0, 4]], [0.2, [59, 15, 112]], [0.4, [140, 41, 129]],
    [0.6, [222, 73, 104]], [0.8, [254, 159, 109]], [1.0, [252, 253, 191]]
  ];
  function colour(v) {
    v = Math.min(Math.max(v, 0), 1);
    for (var i = 1; i < STOPS.length; i++) {
      if (v <= STOPS[i][0]) {
        var a = STOPS[i - 1], b = STOPS[i];
        var t = (v - a[0]) / (b[0] - a[0]);
        var c = a[1].map(function (x, k) { return Math.round(x + t * (b[1][k] - x)); });
        return 'rgb(' + c.join(',') + ')';
      }
    }
    return 'rgb(252,253,191)';
  }
  function paint(el, v) {
    el.style.backgroundColor = colour(v);
    el.style.color = v > 0.62 ? '#1f2328' : '#ffffff';
  }

  function fmt(x, d) { return x.toFixed(d); }
  function fmtErr(x) {
    if (x === 0) return '0';
    if (x < 1e-3) return x.toExponential(1).replace('e', ' × 10^').replace(/\^\+?(-?\d+)/, function (_, e) { return toSup(e); });
    return x.toFixed(4);
  }
  function toSup(e) {
    var map = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
    return e.split('').map(function (ch) { return map[ch] || ch; }).join('');
  }

  function initDemo() {
    var ids = ['p', 's1', 's2', 's3', 's4', 'ghat', 'tau'];
    var inputs = {};
    for (var i = 0; i < ids.length; i++) {
      inputs[ids[i]] = document.getElementById(ids[i]);
      if (!inputs[ids[i]]) return;
    }
    var routing = document.getElementById('routing');
    var parentCell = document.getElementById('parent-cell');
    var parentVal = document.getElementById('parent-val');
    var children = [1, 2, 3, 4].map(function (k) { return document.getElementById('c' + k); });
    var status = document.getElementById('status');
    var out = {
      u: document.getElementById('r-u'), g: document.getElementById('r-g'),
      b: document.getElementById('r-b'), mean: document.getElementById('r-mean'),
      err: document.getElementById('r-err'), unc: document.getElementById('r-unc')
    };

    function update() {
      var p = parseFloat(inputs.p.value);
      var s = ['s1', 's2', 's3', 's4'].map(function (k) { return parseFloat(inputs[k].value); });
      var ghat = parseFloat(inputs.ghat.value);
      var tau = parseFloat(inputs.tau.value);

      ids.forEach(function (k) {
        var o = document.getElementById(k + '-out');
        if (o) o.textContent = parseFloat(inputs[k].value).toFixed(k.charAt(0) === 's' ? 1 : 2);
      });

      var u = 4 * p * (1 - p);
      var g = u * ghat;
      var active = !routing.checked || u >= tau;
      var b = null, r;
      if (active) {
        b = solveOffset(s, p);
        r = s.map(function (x) { return sigmoid(x + b); });
      } else {
        r = [p, p, p, p];
      }
      var q = r.map(function (rj) { return (1 - g) * p + g * rj; });
      var mean = (q[0] + q[1] + q[2] + q[3]) / 4;
      var err = Math.abs(mean - p);
      var unc = s.reduce(function (acc, x) { return acc + sigmoid(x); }, 0) / 4;

      paint(parentCell, p);
      parentVal.textContent = fmt(p, 2);
      children.forEach(function (el, j) {
        paint(el, q[j]);
        el.firstElementChild.textContent = fmt(q[j], 3);
      });

      if (active) {
        status.className = 'status active';
        status.textContent = routing.checked
          ? 'u ≥ τr: solver runs (24 bisection steps)'
          : 'Dense inference: solver runs for every cell';
      } else {
        status.className = 'status bypass';
        status.textContent = 'u < τr: solver bypassed, children copy the parent';
      }

      out.u.textContent = fmt(u, 3);
      out.g.textContent = fmt(g, 3);
      out.b.textContent = b === null ? '— (bypassed)' : fmt(b, 4);
      out.mean.textContent = fmt(mean, 6);
      out.err.textContent = fmtErr(err);
      out.err.className = err < 1e-4 ? 'good' : 'bad';
      out.unc.textContent = fmt(unc, 3) + (Math.abs(unc - p) > 0.005 ? '  (≠ p)' : '');
    }

    var presets = {
      boundary: { p: 0.5, s1: 2.0, s2: 0.5, s3: 0.0, s4: -2.0, ghat: 0.9 },
      background: { p: 0.02, s1: 1.5, s2: -0.5, s3: -1.0, s4: -2.5, ghat: 0.9 },
      polyp: { p: 0.97, s1: 2.5, s2: 1.0, s3: 0.5, s4: -1.0, ghat: 0.9 }
    };
    Array.prototype.forEach.call(document.querySelectorAll('[data-preset]'), function (btn) {
      btn.addEventListener('click', function () {
        var preset = presets[btn.getAttribute('data-preset')];
        Object.keys(preset).forEach(function (k) { inputs[k].value = preset[k]; });
        update();
      });
    });

    ids.forEach(function (k) { inputs[k].addEventListener('input', update); });
    routing.addEventListener('change', update);
    update();
  }

  // ---------- BibTeX copy ----------
  function initCopy() {
    var btn = document.getElementById('copy-bib');
    var text = document.getElementById('bib-text');
    if (!btn || !text) return;
    btn.addEventListener('click', function () {
      var done = function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1600);
      };
      try {
        navigator.clipboard.writeText(text.textContent).then(done, function () { selectText(text); });
      } catch (e) {
        selectText(text);
      }
    });
  }
  function selectText(el) {
    var range = document.createRange();
    range.selectNodeContents(el);
    var sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { initDemo(); initCopy(); });
  } else {
    initDemo(); initCopy();
  }
})();
