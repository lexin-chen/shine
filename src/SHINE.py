import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from mdance.tools.bts import mean_sq_dev, calculate_comp_sim, get_new_index_n
import random
from cycler import cycler

def msd_div(total_data, select):
    start = 'medoid'
    # Numpy array with the data
    
    # total number of fingerprints
    fp_total = len(total_data)
    
    # indices of all the fingerprints
    total_indices = np.array(range(fp_total))
    
    # starting point
    if start =='medoid':
        comps = calculate_comp_sim(total_data)
        seed = np.argmin(comps)
    else:
        seed = random.randint(0, fp_total - 1)
    selected_n = [seed]
    
    # vector with the column sums of all the selected fingerprints
    selected_condensed = total_data[seed].copy()
    selectedsq_condensed = selected_condensed ** 2
    
    # number of fingerprints selected
    n = 1
    while len(selected_n) < select:
        # indices from which to select the new fingerprints
        select_from_n = np.delete(total_indices, selected_n)
        
        # new index selected
        new_index_n = get_new_index_n(total_data, selected_condensed, selectedsq_condensed, n, select_from_n, selected_n)
        
        # updating column sum vector
        selected_condensed += total_data[new_index_n]
        selectedsq_condensed += total_data[new_index_n]**2
        
        # updating selected indices
        selected_n.append(new_index_n)
        #print(selected_n)
        
        # updating n
        n = len(selected_n)
    return selected_n

# We should add this sampling to BTS as well
def quota_sample(data, metric = 'MSD', nbins=10, nsample=100, hard_cap=True):
    """Representative sampling according to comp_sim values.
    
    Divides the range of comp_sim values in nbins and then
    uniformly selects nsample molecules, consecutively
    taking one from each bin
    """
    n = len(data)
    if nsample < 1:
        nsample = int(n * nsample)
    cs = calculate_comp_sim(data, metric=metric, N_atoms=1)
    tups = []
    for i, comp in enumerate(cs):
        tups.append((i, comp))
    comp_sims = np.sort(cs)
    mi = np.min(comp_sims)
    ma = np.max(comp_sims)
    D = ma - mi
    step = D/nbins
    bins = []
    indices = np.array(list(range(n)))
    for i in range(nbins - 1):
        low = mi + i * step
        up = mi + (i + 1) * step
        bins.append(indices[(comp_sims >= low) * (comp_sims < up)])
    bins.append(indices[(comp_sims >= up) * (comp_sims <= ma)])
    order_sampled = []
    i = 0
    while len(order_sampled) < nsample:
        for b in bins:
            if len(b) > i:
                order_sampled.append(b[i])
                if hard_cap:
                    if len(order_sampled) >= nsample:
                        break
            else:
                pass
        i += 1
    tups.sort(key = lambda tups : tups[1])
    sampled_mols = []
    for i in order_sampled:
        sampled_mols.append(tups[i][0])
    return sampled_mols

def gen_msdmatrix(trajs, metric='intra'):
    """Generates matrix of distances between trajectories"""
    ntrajs = len(trajs)
    distances = []
    for i in range(ntrajs):
        distances.append([])
        for j in range(ntrajs):
            if i == j:
                distances[-1].append(0)
            else:
                combined = np.concatenate((trajs[i], trajs[j]), axis = 0)
                if metric == 'intra':
                    d = mean_sq_dev(combined, N_atoms=1)
                elif metric == 'inter':
                    d = (mean_sq_dev(combined) * len(combined)**2 - (mean_sq_dev(trajs[i])*len(trajs[i])**2 + mean_sq_dev(trajs[j])*len(trajs[j])**2))/(len(trajs[i]) * len(trajs[j]))
                elif metric == 'semi_sum':
                    d = mean_sq_dev(combined) - 0.5 * (mean_sq_dev(trajs[i]) + mean_sq_dev(trajs[j]))
                elif metric == 'max':
                    d = mean_sq_dev(combined) - max(mean_sq_dev(trajs[i]), mean_sq_dev(trajs[j]))
                elif metric == 'min':
                    d = mean_sq_dev(combined) - min(mean_sq_dev(trajs[i]), mean_sq_dev(trajs[j]))
                elif metric == 'haus':
                    d = max(directed_hausdorff(trajs[j], trajs[i])[0], directed_hausdorff(trajs[i], trajs[j])[0])
                distances[-1].append(d)
    distances = np.array(distances)
    # Making the distances well behaved while preserving the relative rankings
    distances += distances.T
    distances *= 0.5
    distances -= np.min(distances)
    np.fill_diagonal(distances, 0)
    return distances

def gen_trajs(data, sampling='diversity', sampling_fraction=0.2, frame_cutoff=50):
    """Generates the trajectory dictionary
    data: contains the frames of each trajectory
    sampling: {None, 'diversity', 'quota'}
    sampling_fraction: float in (0, 1] indicating fraction of data to be sampled from each trajectory
    frame_cutoff: minimum number of frames to perform sampling
    """
    trajs = {}
    for i, frames in enumerate(data):
        nframes = len(frames)
        if sampling:
            if nframes >= frame_cutoff:
                nf = int(sampling_fraction * nframes)
                if sampling == 'diversity':
                    s = msd_div(frames, nf)
                elif sampling == 'quota':
                    s = quota_sample(frames, nbins = nf, nsample = nf)
                trajs[i] = frames[s]
            else:
                trajs[i] = frames
        else:
            trajs[i] = frames
    return trajs

def shine(trajs, linkag='ward', metric='intra'):
    """Performs the clustering between the trajectories"""
    distances = gen_msdmatrix(trajs, metric=metric)
    linmat = squareform(distances, force='no', checks=True)
    np.save('linmat.npy', linmat)
    link = linkage(linmat, linkag)
    return link

def cluster_postprocess(link, maxclusts=2, dendro=True, file_base_name='dendrogram'):
    """Post-processing the SHINE clustering"""
    colors = ['black', 'royalblue', 'orangered']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    font_size = 14
    clusters = fcluster(link, t=maxclusts, criterion='maxclust')
    dendrogram(link, no_labels=True)

    if dendro:
        plt.xlabel('Pathways', fontsize=font_size, fontweight='bold')
        plt.ylabel('Distance', fontsize=font_size, fontweight='bold')
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.25)
        plt.savefig(f'{file_base_name}.png', dpi=500, bbox_inches='tight', pad_inches=0.1, transparent=True)
    return clusters