"""Draw matched regular-gather panel tradeoffs; each point uses its own same-GPU control."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
two=json.loads((root/"panels/verified-summary.json").read_text())["timing"]
multi=json.loads((root/"multi-panel/results.json").read_text())
fig, ax=plt.subplots(figsize=(9,5), layout="constrained")
for chunks,color,marker in [(2,"#0a5770","o"),(8,"#b04421","s")]:
    two_arms=two[f"regular-{chunks}"]["arms"]
    many=multi[f"c{chunks}"]
    points=[(0,0)]
    for name in ("32768/95488","65536/62720"):
        d=two_arms[name]
        points.append((d["delta_wall_ms"],-d["delta_total_gib"]))
    for n in ("4","8"):
        points.append((many[n]["median_ms"]-many["1"]["median_ms"],
                       many["1"]["peak_total_gib"]-many[n]["peak_total_gib"]))
    ax.plot(*zip(*points),color=color,marker=marker,lw=2,ms=6,label=f"{chunks} outer loss chunks")
    for (x,y),label in zip(points[1:],["2 unequal","2 balanced","4","8"]):
        offset=10 if chunks==2 else -15
        ax.annotate(label,(x,y),xytext=(0,offset),textcoords="offset points",ha="center",color=color)
ax.set(xlabel="Extra isolated head time per microbatch (ms)",ylabel="Peak GIFT total memory saved (GiB)",
       xlim=(-1,31),ylim=(-0.1,1.9),title="Vocabulary panels on H100 with regular all-gather")
ax.spines[["top","right"]].set_visible(False)
ax.grid(True,axis="y",alpha=0.22)
ax.legend(loc="upper left",frameon=False)
fig.savefig(root/"memory-tradeoff.png",dpi=160)
