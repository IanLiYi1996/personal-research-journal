/* sidebar.js — 参考 game-ai-qa AppSidebar：品牌 + 分组导航（纯文字分组标题）+ 当前页高亮 */
(function () {
  var GROUPS = window.WB_GROUPS || {
    '00': '结论', '01': '背景与协议', '02': '客户画像', '03': '客户关切', '04': '方案与架构',
    '05': '场景方案', '07': '区域与合规', '08': '竞争定位', '09': 'BD 作战手册',
    '10': 'SA 落地与多轨', '11': '风险与待确认', 'Demo': 'Demo 验证'
  };
  function groupOf(slide, i) {
    var k = slide.querySelector('.kicker');
    var t = k ? k.textContent.trim() : '';
    var key = t.split('·')[0].trim();
    if (GROUPS[key]) return GROUPS[key];
    if (i === 0) return '开始';
    return '收尾';
  }
  function build() {
    var deck = document.querySelector('.deck');
    var slides = Array.prototype.slice.call(document.querySelectorAll('.deck > .slide'));
    if (!deck || !slides.length || document.querySelector('.wb-sidebar')) return;
    var aside = document.createElement('aside');
    aside.className = 'wb-sidebar';
    aside.setAttribute('aria-label', '目录');
    var html = '<div class="wb-brand"><div class="logo">' + ((window.WB_BRAND||{}).logo||'AP') + '</div><div><div class="name">' + ((window.WB_BRAND||{}).name||'Agentic Payment on AWS') + '</div>' +
      '<div class="sub">' + ((window.WB_BRAND||{}).sub||'AWS FSI · Internal') + '</div></div></div><nav class="wb-nav">';
    var last = null;
    slides.forEach(function (s, i) {
      var g = groupOf(s, i);
      if (g !== last) { if (last !== null) html += '</div>'; html += '<div class="wb-group"><h2>' + g + '</h2>'; last = g; }
      var title = s.getAttribute('data-title') || ('第 ' + (i + 1) + ' 页');
      html += '<button class="wb-item" data-idx="' + (i + 1) + '"><span class="no">' + (i + 1) + '</span><span>' + title + '</span></button>';
    });
    html += '</div></nav><div class="wb-foot">← → 翻页 · S 演讲者视图 · T 主题</div>';
    aside.innerHTML = html;
    deck.insertBefore(aside, deck.firstChild);
    aside.addEventListener('click', function (e) {
      var b = e.target.closest('.wb-item'); if (!b) return;
      location.hash = '#/' + b.getAttribute('data-idx');
    });
    function sync() {
      var idx = slides.findIndex(function (s) { return s.classList.contains('is-active'); });
      var items = aside.querySelectorAll('.wb-item');
      items.forEach(function (it, j) { it.classList.toggle('is-active', j === idx); });
      var cur = items[idx]; if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: 'nearest' });
    }
    var mo = new MutationObserver(sync);
    slides.forEach(function (s) { mo.observe(s, { attributes: true, attributeFilter: ['class'] }); });
    sync();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build); else build();
})();
