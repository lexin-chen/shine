import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from math import ceil

num = 2
sample_type = 'rep'
data = np.loadtxt(f'{sample_type}_c{num}.csv', delimiter=',')
counts, edges = np.histogram(data, bins=80)

plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
font_size = 12
texts = []

for i in range(len(counts)):
    if counts[i] == 0:
        continue
    plt.bar(edges[i], counts[i], width=edges[i+1]-edges[i], color='lightskyblue')
    texts.append(plt.text(edges[i], counts[i], str(ceil(edges[i])), fontsize=font_size, fontweight='bold'))

adjust_text(texts)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(1.25)
plt.ylabel('Frequency', fontsize=font_size, fontweight='bold')
plt.title(f'Medoid of Cluster {num}', fontsize=font_size, fontweight='bold')
plt.savefig(f'hist_{sample_type}_c{num}.png', dpi=500, bbox_inches='tight', pad_inches=0.1, transparent=True)