import sys
sys.path.insert(0, "../../")
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
from shine.src.SHINE import cluster_postprocess

linmat = np.load('linmat.npy')
link = linkage(linmat, 'ward')
cluster_postprocess(link)