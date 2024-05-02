import sys
sys.path.insert(0, "../../")
from shine.src.SHINE import shine, cluster_postprocess
import glob
import re
import numpy as np

traj_files = sorted(glob.glob('../traj_6d/traj_6d_*.csv'), 
                    key = lambda x: tuple(map(int, re.findall(r'\d+', x))))

all_traj = []
for file in traj_files:
    traj = np.loadtxt(file, delimiter = ',')
    all_traj.append(traj)
    
link = shine(all_traj)
cluster_postprocess(link)
