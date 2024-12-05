import glob
import matplotlib.pyplot as plt

import numpy as np

from shine_class import Shine
    


farthest = False
rep_sampling = False
div_sample = True

trajs = {}
st = ''
frames_all = []
for file in glob.glob('data/*.csv'):
    traj = file.split('_')[-1].split('.')[0]
    frame = np.genfromtxt(file, delimiter=',')
    frames_all.append((traj, frame))

for frac in [0.5, 1]:
    print(int(frac * 100))

    #trajs = gen_trajs(frames_all, trajs, sampling='diversity', frac=frac, frame_cutoff=50)
    #mod = Shine(frames_all, 'MSD', 1, 'ward', 2, 'maxclust', merge_scheme=None, sampling='diversity', frac=frac)
    #trajs = mod.process_trajs()
    links = ['ward']
    merge_schemes = ['intra', 'semi_sum', 'min']
    for merge_scheme in merge_schemes:
        #print(merge_scheme)
        #s = ''
        for li in links:
            st = ''
            #s += '{}\n'.format(link)
            mod = Shine(frames_all, 'MSD', N_atoms=1 , t=2, criterion='maxclust', 
                        link='ward', merge_scheme=merge_scheme, 
                        sampling='diversity', frac=frac)
            """
            link, clusters = mod.run()
            
            # New code
            colors = ['black', 'royalblue', '#FF5349', '#00a86b']
            plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.weight'] = 'bold'
            font_size = 14
            
            for axis in ['top','bottom','left','right']:
                plt.gca().spines[axis].set_linewidth(1.25)
            
            label_indices = []
            for i, cluster in enumerate(np.unique(clusters)):
                label_indices.append(np.where(clusters == cluster)[0])
            custom_labels = [f"{group_consecutive_indices(cluster)}" for i, cluster in enumerate(label_indices)]
            
            num_ticks = len(custom_labels)
            tick_positions = np.linspace(100, 500, num_ticks, dtype=int)
            tick_labels = [custom_label for custom_label in custom_labels]
            ax = plt.gca()
            
            for i, label in enumerate(custom_labels):
                ax.text(tick_positions[i], -0.03, label, color=colors[i+1], ha='center', 
                        va='top', transform=ax.get_xaxis_transform(), fontsize=10, fontweight='bold')
            
            # New code
            dendrogram(link, no_labels=True)
            """
            link, clusters = mod.run()
            ax = mod.plot()
            plt.savefig('div_dendro_{}_{}.png'.format(merge_scheme, int(frac*100)), bbox_inches='tight', dpi=500, pad_inches=0.1)
            plt.close()
            
            #label1 = np.where(clusters == 1)[0]
            #label2 = np.where(clusters == 2)[0]
            #
            ##if len(label1) < len(label2):
            ##    labs = label1
            ##elif len(label1) > len(label2):
            ##    labs = label2
            ##else:
            ##    labs = ['tie']
            #condensed_distances = np.sum(distances, axis=0)
            #c1_med = label1[np.argmin(condensed_distances[label1])]
            #c1_out = label1[np.argmax(condensed_distances[label1])]
            #c2_med = label2[np.argmin(condensed_distances[label2])]
            #c2_out = label2[np.argmax(condensed_distances[label2])]
            #st += '\nFrames {:5}: Cluster1: MEDOID {:5} OUTLIER {:5}   Cluster2: MEDOID {:5} OUTLIER {:5}'.format(int(frac * 100), c1_med, c1_out, c2_med, c2_out)
            ##for l in labs:
            ##    st += '{:4}'.format(l)
            #
            #with open('paths_div_scan_{}_{}.txt'.format(merge_scheme, li), 'a') as outfile:
            #    outfile.write(st)

"""
25
['0-32, 48-79', '33-47']
['33-47', '0-32, 48-79']
['33-47, 53', '0-32, 48-52, 54-79']
50
['0-32, 48-52, 54-79', '33-47, 53']
['33-47, 53', '0-32, 48-52, 54-79']
['33-42, 47, 53', '0-32, 43-46, 48-52, 54-79']
100
['33-47, 66', '0-32, 48-65, 67-79']
['33-47', '0-32, 48-79']
['33-42, 47, 53', '0-32, 43-46, 48-52, 54-79']
"""