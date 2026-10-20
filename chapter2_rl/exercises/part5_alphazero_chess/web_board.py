"""A tiny dependency-free chess board renderer (CSS + JS strings) shared by `play.py` and the game viewer.

JS API:  cbRender(el, fen, {flipped, lastMove: ['e2','e4'], arrows: [{from, to, w (0..1), color}],
                            selected: 'e2', targets: ['e3','e4'], check: 'e1', onClick: sq => ...})
"""

BOARD_CSS = r"""
.cb-wrap{position:relative;width:var(--cb-size,480px);height:var(--cb-size,480px);user-select:none;flex:none}
.cb-grid{display:grid;grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);width:100%;height:100%;
  border-radius:4px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.35)}
.cb-sq{position:relative;display:flex;align-items:center;justify-content:center;line-height:1;cursor:pointer;
  font-size:calc(var(--cb-size,480px) / 8 * .8)}
.cb-l{background:#f0d9b5}.cb-d{background:#b58863}
.cb-last.cb-l{background:#cdd26a}.cb-last.cb-d{background:#aaa23a}
.cb-sel{box-shadow:inset 0 0 0 4px rgba(20,85,30,.75)}
.cb-check{box-shadow:inset 0 0 18px 6px rgba(230,20,20,.85)}
.cb-tgt::after{content:"";position:absolute;width:30%;height:30%;border-radius:50%;background:rgba(20,85,30,.45)}
.cb-tgt.cb-occ::after{width:88%;height:88%;background:none;border:5px solid rgba(20,85,30,.45);box-sizing:border-box}
.cb-p{font-family:"DejaVu Sans","Segoe UI Symbol","Noto Sans Symbols 2","Apple Symbols",sans-serif;z-index:1}
.cb-w{color:#fafafa;-webkit-text-stroke:1.4px #1b1b1b;paint-order:stroke fill;text-shadow:0 2px 3px rgba(0,0,0,.35)}
.cb-b{color:#141414;text-shadow:0 1px 2px rgba(255,255,255,.2)}
.cb-coord{position:absolute;font:600 calc(var(--cb-size,480px) / 42) sans-serif;opacity:.8;z-index:0}
.cb-l .cb-coord{color:#b58863}.cb-d .cb-coord{color:#f0d9b5}
.cb-file{bottom:2px;right:4px}.cb-rank{top:2px;left:3px}
.cb-arrows{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2}
"""

BOARD_JS = r"""
const CB_GLYPH = {k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
function cbParse(fen){
  const rows = fen.split(' ')[0].split('/'), pos = {};
  for(let r=0;r<8;r++){ let f=0;
    for(const ch of rows[r]){ if(/\d/.test(ch)) f+= +ch; else { pos['abcdefgh'[f]+(8-r)] = ch; f++; } } }
  return pos;
}
function cbCenter(sq, flipped){
  const f = 'abcdefgh'.indexOf(sq[0]), r = +sq[1];
  return flipped ? [7-f+.5, r-1+.5] : [f+.5, 8-r+.5];
}
function cbRender(el, fen, o={}){
  const pos = cbParse(fen);
  el.innerHTML=''; el.classList.add('cb-wrap');
  const grid = document.createElement('div'); grid.className='cb-grid'; el.appendChild(grid);
  const last = o.lastMove || [], tg = new Set(o.targets || []);
  for(let i=0;i<64;i++){
    const r=Math.floor(i/8), c=i%8;
    const file = o.flipped ? 7-c : c, rank = o.flipped ? r+1 : 8-r, name='abcdefgh'[file]+rank;
    const d = document.createElement('div');
    d.className = 'cb-sq ' + ((file+rank)%2===0 ? 'cb-d' : 'cb-l');
    if(last.includes(name)) d.classList.add('cb-last');
    if(o.selected===name) d.classList.add('cb-sel');
    if(o.check===name) d.classList.add('cb-check');
    if(tg.has(name)){ d.classList.add('cb-tgt'); if(pos[name]) d.classList.add('cb-occ'); }
    const p = pos[name];
    if(p){ const s=document.createElement('span');
      s.className='cb-p '+(p===p.toUpperCase()?'cb-w':'cb-b'); s.textContent=CB_GLYPH[p.toLowerCase()]+'︎'; d.appendChild(s); }
    if(r===7){ const s=document.createElement('span'); s.className='cb-coord cb-file'; s.textContent='abcdefgh'[file]; d.appendChild(s); }
    if(c===0){ const s=document.createElement('span'); s.className='cb-coord cb-rank'; s.textContent=rank; d.appendChild(s); }
    if(o.onClick) d.onclick = () => o.onClick(name);
    grid.appendChild(d);
  }
  const arrows = o.arrows || [];
  if(arrows.length){
    const NS='http://www.w3.org/2000/svg', svg=document.createElementNS(NS,'svg');
    svg.setAttribute('viewBox','0 0 8 8'); svg.setAttribute('class','cb-arrows');
    arrows.forEach((a,k)=>{
      const col = a.color || '#15781b', w = Math.max(0, Math.min(1, a.w ?? 1));
      const [x1,y1]=cbCenter(a.from,o.flipped), [x2,y2]=cbCenter(a.to,o.flipped);
      const dx=x2-x1, dy=y2-y1, L=Math.hypot(dx,dy), ux=dx/L, uy=dy/L, sw=.07+.16*w, head=.22+.18*w;
      const ex=x2-ux*head, ey=y2-uy*head;
      const line=document.createElementNS(NS,'line');
      line.setAttribute('x1',x1+ux*.15); line.setAttribute('y1',y1+uy*.15); line.setAttribute('x2',ex); line.setAttribute('y2',ey);
      line.setAttribute('stroke',col); line.setAttribute('stroke-width',sw); line.setAttribute('stroke-linecap','round');
      line.setAttribute('opacity', .35+.5*w); svg.appendChild(line);
      const px=-uy, py=ux, hw=head*.62;
      const tri=document.createElementNS(NS,'polygon');
      tri.setAttribute('points', `${x2-ux*.05},${y2-uy*.05} ${ex+px*hw},${ey+py*hw} ${ex-px*hw},${ey-py*hw}`);
      tri.setAttribute('fill',col); tri.setAttribute('opacity', .35+.5*w); svg.appendChild(tri);
    });
    el.appendChild(svg);
  }
}
"""
