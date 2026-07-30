(function () {
  var TRIES = 0;
  function boot() {
    var cv = document.getElementById('kg-canvas');
    if (!cv) { if (TRIES++ < 60) setTimeout(boot, 100); return; }
    if (cv.dataset.ready) return;
    cv.dataset.ready = '1';

    var PALETTE = ['#6ea8fe','#7ee0c0','#f6c177','#eb6f92','#c4a7e7','#9ccfd8','#f2b8b5','#a6da95',
                   '#8bd5ca','#ee99a0','#b7bdf8','#f5a97f'];
    var TYPE_COLORS = {
      'research-notes': '#6ea8fe', 'papers': '#eb6f92', 'topics': '#7ee0c0',
      'tech-blogs': '#f6c177', 'weekly': '#c4a7e7', 'reddit-digests': '#f5a97f'
    };

    var st = { nodes: [], edges: [], scale: 1, tx: 0, ty: 0, hover: null, colorBy: 'type',
               minW: 0, hideIso: false, q: '' };
    var tip = document.getElementById('kg-tip');
    var ctx = cv.getContext('2d');

    fetch('assets/knowledge-graph.json?_=' + Date.now())
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(init)
      .catch(function (e) {
        document.getElementById('kg-canvas-wrap').innerHTML =
          '<p style="padding:16px;color:#c00">图数据加载失败（' + e.message +
          '）。请先运行 <code>uv run --with networkx python3 scripts/wiki_graph.py</code> 生成 assets/knowledge-graph.json。</p>';
      });

    function init(data) {
      var byId = {};
      st.nodes = data.nodes.map(function (n, i) {
        var a = (i / data.nodes.length) * Math.PI * 2;
        var r = 180 + 150 * Math.sqrt((i % 7) / 7);
        var o = Object.assign({}, n, { x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0 });
        byId[n.id] = o; return o;
      });
      st.edges = data.edges.map(function (e) {
        return { s: byId[e.source], t: byId[e.target], w: e.weight, shared: e.shared, link: e.link };
      }).filter(function (e) { return e.s && e.t; });
      document.getElementById('kg-stats').textContent =
        data.stats.nodes + ' 节点 · ' + data.stats.edges + ' 边 · ' + data.stats.communities + ' 社区';
      layout(); resize(); buildLegend(); draw();
    }

    // --- force-directed layout (run offline, then render statically) ---
    function layout() {
      var N = st.nodes, E = st.edges, i, j, it;
      for (it = 0; it < 320; it++) {
        var k = 1 - it / 320;
        for (i = 0; i < N.length; i++) { N[i].vx = 0; N[i].vy = 0; }
        for (i = 0; i < N.length; i++) {          // repulsion
          for (j = i + 1; j < N.length; j++) {
            var dx = N[i].x - N[j].x, dy = N[i].y - N[j].y;
            var d2 = dx * dx + dy * dy || 0.01, d = Math.sqrt(d2);
            var f = 5200 / d2;
            N[i].vx += (dx / d) * f; N[i].vy += (dy / d) * f;
            N[j].vx -= (dx / d) * f; N[j].vy -= (dy / d) * f;
          }
        }
        for (i = 0; i < E.length; i++) {          // spring on edges
          var e = E[i], ex = e.t.x - e.s.x, ey = e.t.y - e.s.y;
          var ed = Math.sqrt(ex * ex + ey * ey) || 0.01;
          var want = 90 + 220 / (1 + e.w);
          var fs = (ed - want) * 0.02 * Math.min(1, e.w / 3);
          e.s.vx += (ex / ed) * fs; e.s.vy += (ey / ed) * fs;
          e.t.vx -= (ex / ed) * fs; e.t.vy -= (ey / ed) * fs;
        }
        for (i = 0; i < N.length; i++) {          // gravity + integrate
          N[i].vx -= N[i].x * 0.006; N[i].vy -= N[i].y * 0.006;
          N[i].x += Math.max(-24, Math.min(24, N[i].vx)) * k;
          N[i].y += Math.max(-24, Math.min(24, N[i].vy)) * k;
        }
      }
      fit();
    }

    function fit() {
      var xs = st.nodes.map(function (n) { return n.x; }), ys = st.nodes.map(function (n) { return n.y; });
      var minx = Math.min.apply(null, xs), maxx = Math.max.apply(null, xs);
      var miny = Math.min.apply(null, ys), maxy = Math.max.apply(null, ys);
      var w = cv.clientWidth || 800, h = cv.clientHeight || 600;
      st.scale = Math.min(w / (maxx - minx + 140), h / (maxy - miny + 140), 2.2);
      st.tx = w / 2 - ((minx + maxx) / 2) * st.scale;
      st.ty = h / 2 - ((miny + maxy) / 2) * st.scale;
    }

    function colorOf(n) {
      return st.colorBy === 'type' ? (TYPE_COLORS[n.type] || '#9aa4b6')
                                   : PALETTE[n.community % PALETTE.length];
    }
    function radius(n) { return 4 + Math.sqrt(n.degree) * 2.6; }
    function visible(n) {
      if (st.hideIso && n.degree <= 1) return false;
      return true;
    }
    function edgeVisible(e) { return e.w >= st.minW && visible(e.s) && visible(e.t); }
    function matches(n) { return st.q && n.label.toLowerCase().indexOf(st.q) >= 0; }

    function resize() {
      var dpr = window.devicePixelRatio || 1;
      cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw() {
      var w = cv.clientWidth, h = cv.clientHeight;
      ctx.clearRect(0, 0, w, h);
      var nbrs = {};
      if (st.hover) { st.edges.forEach(function (e) {
        if (e.s === st.hover) nbrs[e.t.id] = 1; if (e.t === st.hover) nbrs[e.s.id] = 1; }); }
      var P = function (n) { return [n.x * st.scale + st.tx, n.y * st.scale + st.ty]; };

      st.edges.forEach(function (e) {
        if (!edgeVisible(e)) return;
        var dim = st.hover && e.s !== st.hover && e.t !== st.hover;
        var a = P(e.s), b = P(e.t);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.lineWidth = Math.max(0.5, Math.min(3.4, e.w / 2.6));
        ctx.strokeStyle = dim ? 'rgba(180,186,196,.14)'
          : (e.w >= 5 ? 'rgba(46,164,120,.62)' : e.w >= 3 ? 'rgba(110,168,254,.5)' : 'rgba(150,158,170,.34)');
        ctx.stroke();
      });

      st.nodes.forEach(function (n) {
        if (!visible(n)) return;
        var p = P(n), r = radius(n) * Math.min(1.6, st.scale > 1 ? 1 : 1);
        var dim = st.hover && n !== st.hover && !nbrs[n.id];
        ctx.globalAlpha = dim ? 0.22 : 1;
        ctx.beginPath();
        if (n.bridge) { ctx.rect(p[0] - r, p[1] - r, r * 2, r * 2); }
        else { ctx.arc(p[0], p[1], r, 0, Math.PI * 2); }
        ctx.fillStyle = colorOf(n); ctx.fill();
        if (matches(n)) { ctx.lineWidth = 3; ctx.strokeStyle = '#e11d48'; ctx.stroke(); }
        else if (n === st.hover) { ctx.lineWidth = 2.4; ctx.strokeStyle = '#111'; ctx.stroke(); }
        ctx.globalAlpha = 1;
        if (!dim && (st.scale > 0.85 || n.degree >= 6 || matches(n))) {
          ctx.font = '11px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif';
          ctx.fillStyle = 'rgba(40,46,56,.9)'; ctx.textAlign = 'center';
          var lbl = n.label.length > 26 ? n.label.slice(0, 25) + '…' : n.label;
          ctx.fillText(lbl, p[0], p[1] - r - 4);
        }
      });
    }

    function buildLegend() {
      var el = document.getElementById('kg-legend'), h = '';
      if (st.colorBy === 'type') {
        var cnt = {}; st.nodes.forEach(function (n) { cnt[n.type] = (cnt[n.type] || 0) + 1; });
        Object.keys(cnt).sort(function (a, b) { return cnt[b] - cnt[a]; }).forEach(function (t) {
          h += '<div><i style="background:' + (TYPE_COLORS[t] || '#9aa4b6') + '"></i>' + t + ' (' + cnt[t] + ')</div>';
        });
      } else {
        var c2 = {}; st.nodes.forEach(function (n) { c2[n.community] = (c2[n.community] || 0) + 1; });
        Object.keys(c2).sort(function (a, b) { return c2[b] - c2[a]; }).slice(0, 10).forEach(function (c) {
          h += '<div><i style="background:' + PALETTE[c % PALETTE.length] + '"></i>社区 ' + c + ' (' + c2[c] + ')</div>';
        });
      }
      h += '<div style="margin-top:5px;color:#888">■ = 桥节点</div>';
      el.innerHTML = h;
    }

    function pick(mx, my) {
      var best = null, bd = 1e9;
      st.nodes.forEach(function (n) {
        if (!visible(n)) return;
        var px = n.x * st.scale + st.tx, py = n.y * st.scale + st.ty;
        var d = Math.hypot(px - mx, py - my), r = radius(n) + 5;
        if (d < r && d < bd) { bd = d; best = n; }
      });
      return best;
    }

    // --- interaction ---
    var drag = null, moved = false;
    cv.addEventListener('mousedown', function (ev) {
      drag = { x: ev.offsetX, y: ev.offsetY, tx: st.tx, ty: st.ty }; moved = false;
      cv.classList.add('dragging');
    });
    window.addEventListener('mouseup', function () { drag = null; cv.classList.remove('dragging'); });
    cv.addEventListener('mousemove', function (ev) {
      if (drag) {
        if (Math.abs(ev.offsetX - drag.x) + Math.abs(ev.offsetY - drag.y) > 3) moved = true;
        st.tx = drag.tx + (ev.offsetX - drag.x); st.ty = drag.ty + (ev.offsetY - drag.y);
        tip.hidden = true; draw(); return;
      }
      var n = pick(ev.offsetX, ev.offsetY);
      if (n !== st.hover) { st.hover = n; draw(); }
      if (n) {
        tip.hidden = false;
        tip.innerHTML = '<b>' + n.label + '</b><br>' + n.type + ' · 连接 ' + n.degree +
          ' · 社区 ' + n.community + (n.bridge ? ' · 桥节点' : '') + '<br><span style="color:#9aa4b6">单击打开</span>';
        var wrapW = cv.clientWidth;
        tip.style.left = Math.min(ev.offsetX + 14, wrapW - 330) + 'px';
        tip.style.top = (ev.offsetY + 14) + 'px';
        cv.style.cursor = 'pointer';
      } else { tip.hidden = true; cv.style.cursor = drag ? 'grabbing' : 'grab'; }
    });
    cv.addEventListener('mouseleave', function () { tip.hidden = true; st.hover = null; draw(); });
    cv.addEventListener('click', function (ev) {
      if (moved) return;
      var n = pick(ev.offsetX, ev.offsetY);
      if (n) { window.location.hash = n.route.replace(/^#/, ''); }   // ← 点击进入对应笔记
    });
    cv.addEventListener('dblclick', function () { fit(); draw(); });
    cv.addEventListener('wheel', function (ev) {
      ev.preventDefault();
      var f = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
      var mx = ev.offsetX, my = ev.offsetY;
      st.tx = mx - (mx - st.tx) * f; st.ty = my - (my - st.ty) * f;
      st.scale *= f; draw();
    }, { passive: false });

    document.getElementById('kg-color').addEventListener('change', function (e) {
      st.colorBy = e.target.value; buildLegend(); draw();
    });
    document.getElementById('kg-minw').addEventListener('input', function (e) {
      st.minW = parseFloat(e.target.value);
      document.getElementById('kg-minw-val').textContent = e.target.value; draw();
    });
    document.getElementById('kg-hide-iso').addEventListener('change', function (e) {
      st.hideIso = e.target.checked; draw();
    });
    document.getElementById('kg-search').addEventListener('input', function (e) {
      st.q = e.target.value.trim().toLowerCase(); draw();
    });
    window.addEventListener('resize', function () { resize(); draw(); });
  }
  window.__kgBoot = function () { TRIES = 0; boot(); };
  boot();
})();
