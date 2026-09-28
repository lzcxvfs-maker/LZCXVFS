"""Generate figures solely from this run's saved experimental results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
v=json.loads((OUT/"explore.json").read_text(encoding="utf-8"))
f=json.loads((OUT/"final.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.size":10,"figure.dpi":150,"savefig.dpi":180})

ks=np.array([1,3,5,7,9])
fig,ax=plt.subplots(figsize=(7,3.1))
for key,label in [("k_scan","Pixels L2"),("weighted","Pixels L2 weighted")]:
    ax.plot(ks,[v[key][str(k)]["top1"]*100 for k in ks],marker="o",label=label)
for key,label in [("pca","PCA-50 L2"),("hog","HOG L2"),("pca_l1","PCA-50 L1")]:
    ax.plot(ks,[v[key]["k_scan"][str(k)]["top1"]*100 for k in ks],marker="o",label=label)
ax.set(xlabel="k",ylabel="Validation Top-1 (%)",xticks=ks,ylim=(95,98))
ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2); fig.tight_layout()
fig.savefig(OUT/"validation_k.png");plt.close(fig)

fig,ax=plt.subplots(figsize=(6.5,3))
names=["Pixels","PCA-50","HOG","PCA-50 L1"]
times=[v["raw_timing"]["neighbors_s"],v["pca"]["timing"]["neighbors_s"],
       v["hog"]["timing"]["neighbors_s"],v["pca_l1"]["timing"]["neighbors_s"]]
bars=ax.bar(names,times,color=["#4d78a8","#60a886","#d5a547","#be6d6d"])
ax.bar_label(bars,fmt="%.3f s",padding=3)
ax.set(ylabel="Neighbor search on 12k x 2k (s)",ylim=(0,max(times)*1.2))
fig.tight_layout();fig.savefig(OUT/"timing.png");plt.close(fig)

cm=np.asarray(f["cm"])
fig,ax=plt.subplots(figsize=(5.4,4.8))
ax.imshow(cm,cmap="Blues")
ax.set(xticks=np.arange(10),yticks=np.arange(10),xlabel="Predicted",ylabel="True")
for i in range(10):
    for j in range(10):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=6.5,
                color="white" if cm[i,j]>600 else "#263238")
fig.tight_layout();fig.savefig(OUT/"confusion_matrix.png");plt.close(fig)

sample=np.load(OUT/"sample_images.npz")
images,truth,pred=sample["images"],sample["truth"],sample["pred"]
fig,axs=plt.subplots(10,10,figsize=(8,8.2))
for i,(ax,img,t,p) in enumerate(zip(axs.flat,images,truth,pred)):
    ax.imshow(img,cmap="gray_r",vmin=0,vmax=255)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(f"{t}/{p}",fontsize=6,color="#ba2626" if t!=p else "#245236",pad=1)
for row in axs:
    for ax in row:
        for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.5,.005,"Each row: one true class (0-9); title = true/predicted; red = error",
         ha="center",fontsize=8)
fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.035,wspace=.12,hspace=.24)
fig.savefig(OUT/"samples_100.png");plt.close(fig)
print("figures generated")
