import sys
sys.path.insert(0, "../../../")
from shine.src.SHINE import shine, cluster_postprocess, gen_trajs
import numpy as np
import glob
import re

sampling = 'quota'
sampling_fraction = 0.25
metric = 'intra'

traj_files = sorted(glob.glob('../../../../data/ala/double-periodic/pathway_dih_*.csv'), 
                    key = lambda x: tuple(map(int, re.findall(r'\d+', x))))
all_traj = []
for file in traj_files:
    traj = np.loadtxt(file, delimiter = ',')
    all_traj.append(traj)
    
link = shine(all_traj)
#all_traj = gen_trajs(all_traj, sampling=sampling, sampling_fraction=sampling_fraction, frame_cutoff=50)
link = shine(all_traj, linkag='ward', metric=metric)
clusters = cluster_postprocess(link, file_base_name=f'{metric}_{sampling}_{sampling_fraction}')
print(clusters[:10])