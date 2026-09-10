"""Draw the observed evaluation stages; no inferred internal model operations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(10,3.2));ax.set(xlim=(0,10),ylim=(0,3.2));ax.axis('off')
def box(x,y,w,h,text,color='#e9f0f5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor=color,edgecolor='#425466',linewidth=1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#425466','lw':1.3})
box(.12,1.1,1.65,1.05,'Released parallel\ntranslations')
box(2.12,1.1,1.65,1.05,'Fragment and\ninterleave')
box(4.12,1.1,1.4,1.05,'Target model\n(two output sections)')
box(6.02,1.9,1.75,.85,'Reconstruction\nequivalence: R')
box(6.02,.5,1.75,.85,'Answer\nunsafe label: U')
box(8.3,1.1,1.5,1.05,'Joint event\nR AND U','#e2f1eb')
arrow((1.82,1.625),(2.06,1.625));arrow((3.82,1.625),(4.06,1.625))
arrow((5.57,1.85),(5.96,2.3));arrow((5.57,1.4),(5.96,.95))
arrow((7.82,2.3),(8.24,1.85));arrow((7.82,.95),(8.24,1.4))
ax.text(5,.1,'Separate judging calls measure visible outputs; internal reasoning is not observed.',ha='center',fontsize=10,color='#425466')
p=Path(__file__).resolve().parents[1]/'paper/figures/fig_reviewed_evaluation.pdf'
fig.savefig(p,bbox_inches='tight',pad_inches=.05);plt.close(fig)
