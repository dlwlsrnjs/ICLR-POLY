#!/usr/bin/env python3
"""Method figure (SVG->PDF/PNG), same illustration tone as fig_overview:
PolyJigsaw pipeline (1-4) plus the two-judge reconstruction-gated evaluation."""
import cairosvg
W,H=1180,510
INK="#1f2937"; STEEL="#64748b"; ACC="#2563eb"; GOLD="#c2620b"; RED="#be123c"; GRN="#15803d"; PUR="#7c3aed"
def esc(t): return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def text(x,y,s,size=17,color=INK,anchor="middle",weight="normal",
         family="Noto Sans CJK JP, DejaVu Sans, sans-serif",style="normal"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" dominant-baseline="middle">{esc(s)}</text>')
def rrect(x,y,w,h,fill,edge,rx=12,lw=2.5,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{edge}" stroke-width="{lw}"{d}/>'
def arrow(x1,y1,x2,y2,color):
    aid=f"m{int(x1)}{int(y1)}{int(x2)}{int(y2)}"
    return (f'<defs><marker id="{aid}" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 Z" fill="{color}"/></marker></defs>'
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3.5" marker-end="url(#{aid})"/>')
def robot(cx,cy,s=1.0):
    c="#334155"; g=[]
    g.append(f'<line x1="{cx}" y1="{cy-58*s}" x2="{cx}" y2="{cy-78*s}" stroke="{c}" stroke-width="3"/>')
    g.append(f'<circle cx="{cx}" cy="{cy-83*s}" r="7" fill="{RED}" stroke="{c}" stroke-width="2"/>')
    g.append(rrect(cx-40*s,cy-58*s,80*s,44*s,"#e2e8f0",c,rx=11,lw=2.5))
    g.append(f'<circle cx="{cx-16*s}" cy="{cy-37*s}" r="8" fill="#38bdf8" stroke="{c}" stroke-width="2"/>')
    g.append(f'<circle cx="{cx+16*s}" cy="{cy-37*s}" r="8" fill="#38bdf8" stroke="{c}" stroke-width="2"/>')
    g.append(rrect(cx-46*s,cy-8*s,92*s,66*s,"#cbd5e1",c,rx=13,lw=2.5))
    g.append(text(cx,cy+24*s,"LLM",size=20*s,color=c,weight="bold",family="DejaVu Sans, sans-serif"))
    return "".join(g)
def lock(cx,cy,color,openp=False,s=1.0):
    g=[]
    if openp: g.append(f'<path d="M {cx-11*s} {cy-5*s} v -8 a 11 11 0 0 1 22 0" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
    else: g.append(f'<path d="M {cx-11*s} {cy-5*s} v -10 a 11 11 0 0 1 22 0 v 10" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
    g.append(rrect(cx-14*s,cy-5*s,28*s,22*s,color,color,rx=4,lw=1))
    return "".join(g)

el=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    f'<rect width="{W}" height="{H}" fill="white"/>']

# Stage 1
el.append(rrect(40,70,210,120,"#eff6ff","#2563eb",rx=14))
el.append(text(145,96,"(1) Parallel translations",size=17,color="#1e3a8a",weight="bold"))
el.append(text(145,124,"EN  ZH  FR  AR  RU …",size=16,color=INK))
el.append(text(60,150,"how to …",size=14,color=STEEL,anchor="start"))
el.append(text(60,170,"如何 …   comment …",size=14,color=STEEL,anchor="start"))
el.append(arrow(252,130,300,130,INK))
# Stage 2
el.append(rrect(300,70,210,120,"white","#334155",rx=14))
el.append(text(405,96,"(2) Fragment + tag",size=17,weight="bold"))
el.append(rrect(320,118,80,32,"#dbeafe","#2563eb",rx=8,lw=2))
el.append(text(360,134,"F01",size=15,color="#1e3a8a"))
el.append(rrect(410,118,80,32,"#fef3c7","#c2620b",rx=8,lw=2))
el.append(text(450,134,"F02",size=15,color="#7c2d12"))
el.append(text(405,170,"verbatim spans only",size=13,color=STEEL,style="italic"))
el.append(arrow(512,130,560,130,INK))
# Stage 3
el.append(rrect(560,70,330,120,"#fff7ed","#c2620b",rx=14))
el.append(text(725,96,"(3) Interleave into one paragraph",size=17,color="#7c2d12",weight="bold"))
el.append(text(578,126,"F01:AR  F02:EN  F03:ZH  F04:FR",size=15,color=INK,anchor="start"))
el.append(text(578,150,"F05:EN  F06:ZH  F07:AR  …",size=15,color=INK,anchor="start"))
el.append(text(725,176,"ordered  /  shuffled",size=14,color="#7c2d12"))
# arrow to robot
el.append(arrow(920,130,975,130,INK))
el.append(robot(1040,140,s=1.0))
# down to stage 4
el.append(arrow(1040,232,1040,282,INK))
# Stage 4
el.append(rrect(700,285,438,86,"#f1f5f9","#334155",rx=14))
el.append(text(919,309,"(4) Target LLM output",size=17,weight="bold"))
el.append(text(720,335,"[RECONSTRUCTED] English request",size=15,color="#1e3a8a",anchor="start"))
el.append(text(720,357,"[ANSWER] carries it out",size=15,color="#7f1d1d",anchor="start"))
# to two judges
el.append(arrow(700,318,560,318,ACC))
el.append(rrect(300,292,255,58,"#eef2ff","#4f46e5",rx=12))
el.append(text(320,314,"Reconstruction judge",size=16,color="#3730a3",anchor="start",weight="bold"))
el.append(text(320,335,"equivalent to q ?",size=15,color="#3730a3",anchor="start"))
el.append(arrow(700,350,560,430,RED))
el.append(rrect(300,404,255,58,"#fef2f2","#be123c",rx=12))
el.append(text(320,426,"Safety judge",size=16,color="#7f1d1d",anchor="start",weight="bold"))
el.append(text(320,447,"[ANSWER] unsafe ?",size=15,color="#7f1d1d",anchor="start"))
# AND gate -> gated ASR
el.append(arrow(300,321,238,362,"#4f46e5"))
el.append(arrow(300,433,238,390,RED))
el.append(rrect(55,338,180,76,"#111827","#111827",rx=14))
el.append(lock(85,376,"#f87171",openp=True,s=0.95))
el.append(text(108,367,"gated ASR",size=16,color="white",anchor="start",weight="bold"))
el.append(text(108,392,"equiv AND unsafe",size=12,color="#cbd5e1",anchor="start"))

el.append("</svg>")
svg="".join(el)
open("paper/figures/fig_method.svg","w").write(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to="paper/figures/fig_method.pdf")
cairosvg.svg2png(bytestring=svg.encode(), write_to="paper/figures/fig_method.png", output_width=1180)
print("wrote method figure (svg tone)")
