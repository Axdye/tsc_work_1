#!/usr/bin/env python3
"""Editable draw.io source; vector/PNG exports are rendered from that XML.

The deliberately small renderer supports only the primitives used in this file.
No network access or generated bitmap assets are used.
"""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
W, H = 2000, 960
COLORS = {
    "input": ("#E8F2F5", "#58727D"),
    "standard": ("#EAF0F6", "#63758A"),
    "proposed": ("#F1D7D4", "#B44948"),
    "update": ("#F4EEDC", "#9A7B3F"),
    "decision": ("#E5F1E3", "#5A8A55"),
}
DARK, GRAY = "#263238", "#6B7280"


def make_xml():
    doc = ET.Element("mxfile", host="app.diagrams.net", version="26.0.9")
    page = ET.SubElement(doc, "diagram", id="paired-cf-grpo", name="Method overview")
    model = ET.SubElement(page, "mxGraphModel", page="1", pageWidth=str(W), pageHeight=str(H), grid="1", gridSize="8")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def node(id, x, y, w, h, value="", kind="standard", fs=26, bold=False, text=False, ellipse=False, dashed=False, rich=False, align="center"):
        fill, stroke = COLORS[kind]
        style = f"fontFamily=Arial;fontSize={fs};fontColor={DARK};fontStyle={int(bold)};whiteSpace=wrap;html={int(rich)};align={align};verticalAlign=middle;spacing=8;"
        if text:
            style += "shape=text;strokeColor=none;fillColor=none;"
        else:
            style += f"fillColor={fill};strokeColor={stroke};strokeWidth={2.3 if kind == 'proposed' else 1.7};"
            style += "ellipse;shape=ellipse;" if ellipse else "rounded=1;arcSize=12;"
        if dashed:
            style += "fillColor=none;dashed=1;dashPattern=8 6;"
        cell = ET.SubElement(root, "mxCell", id=id, value=value, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return id

    def edge(id, source, target, points, dashed=False, color=DARK, arrow=True, width=1.8):
        style = f"edgeStyle=none;rounded=0;html=0;strokeColor={color};strokeWidth={width};endArrow={'block' if arrow else 'none'};endFill=1;endSize=10;"
        if dashed:
            style += "dashed=1;dashPattern=7 5;"
        cell = ET.SubElement(root, "mxCell", id=id, value="", style=style, edge="1", parent="1", source=source, target=target)
        g = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(g, "mxPoint", x=str(points[0][0]), y=str(points[0][1]), **{"as": "sourcePoint"})
        ET.SubElement(g, "mxPoint", x=str(points[-1][0]), y=str(points[-1][1]), **{"as": "targetPoint"})
        if len(points) > 2:
            arr = ET.SubElement(g, "Array", **{"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))

    node("title", 48, 28, 1904, 64, "Paired-CF-GRPO: counterfactual credit for traffic signal control", fs=38, bold=True, text=True)
    for id, x, width, text in [
        ("a", 24, 312, "(a) Traffic state"),
        ("b", 350, 318, "(b) Shared policy"),
        ("c", 704, 730, "(c) Paired evaluator (training only)"),
        ("d", 1536, 428, "(d) Credit & update"),
    ]:
        node("stage_" + id, x, 116, width, 48, text, fs=29, bold=True, text=True)

    node("cityflow", 52, 200, 260, 236, "", kind="input", dashed=True)
    node("cityflow_title", 70, 208, 224, 38, "CityFlow", fs=26, bold=True, text=True)
    centers = {}
    for row in range(4):
        for col in range(4):
            id = f"junction_{row}_{col}"
            x, y = 106 + col * 50, 272 + row * 44
            target = (row, col) == (1, 2)
            size = 28 if target else 14
            node(id, x - size/2, y - size/2, size, size, "i" if target else "",
                 kind="proposed" if target else "input", fs=18, bold=target, ellipse=True)
            centers[row, col] = (id, x, y, size / 2)
    for (r, c), (id, x, y, rad) in centers.items():
        for rr, cc in ((r + 1, c), (r, c + 1)):
            if (rr, cc) not in centers:
                continue
            other, xx, yy, r2 = centers[rr, cc]
            dx, dy = (xx-x)/max(abs(xx-x), abs(yy-y)), (yy-y)/max(abs(xx-x), abs(yy-y))
            edge(f"road_{r}_{c}_{rr}_{cc}", id, other,
                 [(x + rad*dx, y + rad*dy), (xx - r2*dx, yy - r2*dy)],
                 color=COLORS["input"][1], arrow=False, width=1.6)

    node("observation", 52, 470, 260, 92, "Queues / phase\nDownstream traffic", kind="input", fs=25)
    edge("observe", "cityflow", "observation", [(182,436),(182,470)])
    node("actor_note", 370, 282, 270, 40, "Old policy frozen", fs=22, text=True)
    node("actor", 370, 350, 270, 160, "<b>Shared LLM policy</b><br>Qwen3.5-0.8B<br>LoRA adapters", fs=28, rich=True)
    edge("policy_input", "observation", "actor", [(312,516),(340,516),(340,400),(370,400)])
    node("shared", 382, 628, 246, 76, "Same parameters\nat every junction", fs=24, text=True)

    node("evaluator", 700, 200, 750, 600, "", kind="proposed", dashed=True)
    node("pair_context", 720, 216, 710, 92,
         "Restore the same snapshot <i>ξ</i><sub>t</sub><br>Reuse other junctions' current actions <i>b</i><sub>−i</sub>", kind="proposed", fs=26, rich=True)
    edge("snapshot", "cityflow", "pair_context", [(312,294),(340,294),(340,180),(1060,180),(1060,216)],
         dashed=True, color=GRAY)
    node("candidate_head", 708, 308, 190, 34, "Replies (G)", fs=22, text=True)
    node("joint_head", 910, 308, 186, 34, "Current action", fs=22, text=True)
    node("rollout_head", 1100, 308, 300, 34, "Frozen-policy rollout", fs=22, text=True)

    for row, idx in enumerate(("1", "2", "G")):
        y, mid = 350 + row * 150, 400 + row * 150
        node("reply_"+idx, 720, y, 165, 100, f"Reply <i>y</i><sub>{idx}</sub><br>Phase <i>a</i><sub>i</sub><sup>({idx})</sup>", kind="decision", fs=25, rich=True)
        edge("sample_"+idx, "actor", "reply_"+idx, [(640,425),(680,425),(680,mid),(720,mid)])
        node("joint_"+idx, 930, y, 150, 100, f"Target <i>a</i><sub>i</sub><sup>({idx})</sup><br>Other <i>b</i><sub>−i</sub>", kind="proposed", fs=24, rich=True)
        edge("parse_"+idx, "reply_"+idx, "joint_"+idx, [(885,mid),(930,mid)])
        for h, text in enumerate(("t", "t+1", "t+2")):
            x = 1125 + h * 92
            node(f"state_{idx}_{h}", x, y+18, 64, 64, text, kind="proposed" if h == 0 else "standard", fs=26)
            if h == 0:
                edge("execute_"+idx, "joint_"+idx, f"state_{idx}_0", [(1080,mid),(1125,mid)])
            else:
                edge(f"transition_{idx}_{h}", f"state_{idx}_{h-1}", f"state_{idx}_{h}", [(x-28,mid),(x,mid)])
        node("return_"+idx, 1392, y+18, 48, 64, f"<i>Z</i><sub>{idx}</sub>", fs=26, text=True, rich=True)
        edge("return_edge_"+idx, f"state_{idx}_2", "return_"+idx, [(1373,mid),(1392,mid)])
        rail = 1465 + row * 25
        dest = 355 + row * 30
        edge("score_"+idx, "return_"+idx, "pool", [(1440,mid),(rail,mid),(rail,dest),(1540,dest)])
    node("omitted_candidates", 910, 606, 160, 40, "…", fs=28, text=True)

    node("pool", 1540, 320, 420, 120, "Pool same-phase returns\nSame phase → same q(a)", kind="update", fs=26)
    node("advantage", 1540, 490, 420, 105, "Group-relative advantage<br><i>A</i><sub>k</sub> = (<i>q</i><sub>k</sub> − μ) / (σ + ε)", kind="update", fs=26, rich=True)
    node("grpo", 1540, 630, 420, 125, "GRPO + KL\nOne advantage for all reply tokens\nUpdate LoRA only", kind="update", fs=24)
    edge("normalize", "pool", "advantage", [(1750,440),(1750,490)])
    edge("optimize", "advantage", "grpo", [(1750,595),(1750,630)])
    edge("parameter_update", "grpo", "actor", [(1750,755),(1750,830),(650,830),(650,495),(640,495)],
         dashed=True, color=GRAY, width=2.0)
    node("return_note", 720, 758, 710, 36, "<i>Z</i><sub>k</sub>: discounted return (−network queue)", fs=22, text=True, rich=True)
    node("update_note", 850, 836, 680, 36, "LoRA update after each collection window", fs=22, text=True)

    node("visit", 52, 630, 260, 120, "Advance environment\nSampled policy action\n(no rollout argmax)", kind="decision", fs=23)
    edge("visit_action", "actor", "visit", [(505,510),(505,586),(340,586),(340,690),(312,690)], color=COLORS["decision"][1])
    edge("next_state", "visit", "cityflow", [(52,690),(18,690),(18,318),(52,318)], color=COLORS["decision"][1])

    # Legend gives each used fill a meaning; there are no fabricated traffic scores.
    for j, (kind, label) in enumerate((("input","Traffic input"),("standard","Shared policy"),("proposed","Paired comparison"),("update","Scoring / update"),("decision","Action / visit"))):
        x = 60 + j * 285
        node("legend_dot_"+kind, x, 910, 18, 18, kind=kind, ellipse=True)
        node("legend_"+kind, x+26, 902, 245, 36, label, fs=22, text=True, align="left")
    node("defaults", 1500, 894, 460, 48, "Defaults: G=4 · H=3 · M=1", fs=23, text=True)
    path = OUT / "overview.drawio"
    ET.indent(doc, space="  ")
    ET.ElementTree(doc).write(path, encoding="utf-8", xml_declaration=True)
    return path


def render(path, pass_number):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
    from matplotlib.path import Path as MPath
    from matplotlib import font_manager
    from html.parser import HTMLParser
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family":"Arial", "svg.fonttype":"none", "pdf.fonttype":42})
    xml=ET.parse(path)
    fig=plt.figure(figsize=(W/100,H/100), dpi=100, facecolor="white")
    ax=fig.add_axes((0,0,1,1));ax.set(xlim=(0,W),ylim=(H,0));ax.set_axis_off()
    renderer=fig.canvas.get_renderer()
    class Runs(HTMLParser):
        def __init__(self):
            super().__init__();self.lines=[[]];self.active=set()
        def handle_starttag(self,tag,attrs):
            if tag=="br":self.lines.append([])
            else:self.active.add(tag)
        def handle_endtag(self,tag):self.active.discard(tag)
        def handle_data(self,data):self.lines[-1].append((data,set(self.active)))
    def rich_text(value,x,y,w,h,p):
        parser=Runs();parser.feed(value)
        size=float(p.get("fontSize",26));lineheight=size*1.35
        for li,runs in enumerate(parser.lines):
            pieces=[]
            for text,tags in runs:
                sub,sup="sub" in tags,"sup" in tags
                scale=.70 if sub or sup else 1
                fp=font_manager.FontProperties(family="Arial",size=size*.72*scale,
                    style="italic" if "i" in tags else "normal",
                    weight="bold" if "b" in tags or p.get("fontStyle")=="1" else "normal")
                width=renderer.get_text_width_height_descent(text,fp,False)[0]
                pieces.append((text,fp,width,sub,sup))
            xpos=x+(w-sum(v[2] for v in pieces))/2
            baseline=y+h/2+(li-(len(parser.lines)-1)/2)*lineheight+size*.32
            for text,fp,width,sub,sup in pieces:
                shift=size*.21 if sub else -size*.31 if sup else 0
                ax.text(xpos,baseline+shift,text,fontproperties=fp,color=p.get("fontColor",DARK),
                        ha="left",va="baseline",zorder=4)
                xpos+=width
    def props(cell):
        return dict(chunk.split("=",1) for chunk in cell.get("style", "").split(";") if "=" in chunk)
    for cell in xml.findall(".//mxCell[@vertex='1']"):
        p=props(cell);g=cell.find("mxGeometry")
        x,y,w,h=[float(g.get(k,0)) for k in ("x","y","width","height")]
        fill,stroke=p.get("fillColor","none"),p.get("strokeColor","none")
        if p.get("shape")!="text":
            kw=dict(facecolor=fill,edgecolor=stroke,linewidth=float(p.get("strokeWidth",1.7))*.72,zorder=1)
            if p.get("dashed")=="1":kw["linestyle"]=(0,(8,6))
            shape=Ellipse((x+w/2,y+h/2),w,h,**kw) if p.get("shape")=="ellipse" else FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=12",**kw)
            ax.add_patch(shape)
        if cell.get("value"):
            if p.get("html")=="1":rich_text(cell.get("value"),x,y,w,h,p)
            else:
                left=p.get("align")=="left"
                ax.text(x+8 if left else x+w/2,y+h/2,cell.get("value"),ha="left" if left else "center",va="center",fontsize=float(p.get("fontSize",26))*.72,
                        fontweight="bold" if p.get("fontStyle")=="1" else "normal",color=p.get("fontColor",DARK),linespacing=1.35,zorder=4)
    for cell in xml.findall(".//mxCell[@edge='1']"):
        p=props(cell);g=cell.find("mxGeometry")
        start=g.find("mxPoint[@as='sourcePoint']");end=g.find("mxPoint[@as='targetPoint']")
        points=[(float(start.get("x")),float(start.get("y")))]
        points += [(float(pt.get("x")),float(pt.get("y"))) for pt in g.findall("Array/mxPoint")]
        points += [(float(end.get("x")),float(end.get("y")))]
        path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
        patch=FancyArrowPatch(path=path,arrowstyle="-|>" if p.get("endArrow")!="none" else "-",mutation_scale=15,
                              linewidth=float(p.get("strokeWidth",1.8))*.72,color=p.get("strokeColor",DARK),
                              linestyle=(0,(7,5)) if p.get("dashed")=="1" else "-",zorder=2)
        ax.add_patch(patch)
    fig.savefig(OUT/f"review-{pass_number}.png",dpi=100)
    fig.savefig(OUT/"overview.png",dpi=200)
    fig.savefig(OUT/"overview.svg")
    fig.savefig(OUT/"overview.pdf")
    # 1000px version is used to inspect label hierarchy at reduced width.
    fig.savefig(OUT/"paper-scale.png",dpi=50)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--pass-number",type=int,default=1)
    args=parser.parse_args()
    if args.render:
        render(OUT/"overview.drawio",args.pass_number)
    else:
        print(make_xml())
