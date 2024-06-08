import numpy as np
import glob
import time
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster


# These are the same functions you already have in MDANCE
def mean_sq_dev(matrix, N_atoms=1):
    """O(N) Mean square deviation (MSD) calculation for n-ary objects.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Data matrix.
    N_atoms : int
        Number of atoms in the system.
    
    Returns
    -------
    float
        normalized MSD value.
    """
    N = len(matrix)
    sq_data = matrix ** 2
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(sq_data, axis=0)
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N * N) # this part should call msd_condensed
    norm_msd = msd / N_atoms
    return norm_msd

def msd_condensed(c_sum, sq_sum, N, N_atoms=1):
    """Condensed version of 'mean_sq_dev'.

    Parameters
    ----------
    c_sum : array-like of shape (n_features,)
        Column sum of the data. 
    sq_sum : array-like of shape (n_features,)
        Column sum of the squared data.
    N : int
        Number of data points.
    N_atoms : int
        Number of atoms in the system.
    
    Returns
    -------
    float
        normalized MSD value.
    """
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N * N)
    norm_msd = msd / N_atoms
    return norm_msd

def get_new_index_n(total_data, selected_condensed, selectedsq_condensed, n, select_from_n, selected_n):
    """Select a diverse object using the ECS_MeDiv algorithm"""
    n_total = n + 1
    # min value that is guaranteed to be higher than all the comparisons
    max_value = -3.08
    
    # placeholder index
    indices = [len(total_data[0]) + 1]
    
    # for all indices that have not been selected
    for i in select_from_n:
        # column sum
        c_total = selected_condensed + total_data[i]
        sq_total = selectedsq_condensed + total_data[i]**2
        # calculating similarity
        msd_value = msd_condensed(c_total, sq_total, n_total)
        # if the sim of the set is less than the similarity of the previous diverse set, update min_value and index
        if msd_value > max_value:
            indices = [i]
            max_value = msd_value
    return indices[0]

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

def extended_comparison(matrix, data_type='full', metric='MSD', N=None, N_atoms=1, 
                        **kwargs):
    """Calculate the extended comparison of a dataset. 
    
    Parameters
    ----------
    matrix : {array-like of shape (n_samples, n_features), tuple/list of length 1 or 2}
        Input data matrix.
        For 'full', use numpy.ndarray of shape (n_samples, n_features).
        For 'condensed', use tuple/list of length 1 (c_sum) or 2 (c_sum, sq_sum).
    data_type : {'full', 'condensed'}, optional
        Type of data inputted. Defaults to 'full'.
        Options:
            - 'full': Use numpy.ndarray of shape (n_samples, n_features).
            - 'condensed': Use tuple/list of length 1 (c_sum) or 2 (c_sum, sq_sum).
    metric : {'MSD', 'BUB', 'Fai', 'Gle', 'Ja', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2'}
        Metric to use for the extended comparison. Defaults to 'MSD'.
        Available metrics:
        Mean square deviation (MSD), Bhattacharyya's U coefficient (BUB),
        Faiman's coefficient (Fai), Gleason's coefficient (Gle),
        Jaccard's coefficient (Ja), Jaccard-Tanimoto coefficient (JT),
        Rogers-Tanimoto coefficient (RT), Russell-Rao coefficient (RR),
        Simpson's coefficient (SM), Sokal-Sneath 1 coefficient (SS1),
        Sokal-Sneath 2 coefficient (SS2).
    N : int, optional
        Number of data points. Defaults to None.
    N_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    **kwargs
        c_threshold : int, optional
            Coincidence threshold. Defaults to None.
        w_factor : {'fraction', 'power_n'}, optional
            Type of weight function that will be used. Defaults to 'fraction'.
            See `esim_modules.calculate_counters` for more information.
    
    Raises
    ------
    TypeError
        If data is not a numpy.ndarray or tuple/list of length 2.
    
    Returns
    -------
    float
        Extended comparison value.
    """
    if data_type == 'full':
        if not isinstance(matrix, np.ndarray):
            raise TypeError('data must be a numpy.ndarray')
        c_sum = np.sum(matrix, axis=0)
        if not N:
            N = len(matrix)
        if metric == 'MSD':
            sq_data = matrix ** 2
            sq_sum = np.sum(sq_data, axis=0)
    elif data_type == 'condensed':
        if not isinstance(matrix, (tuple, list)):
            raise TypeError('data must be a tuple or list of length 1 or 2')
        c_sum = matrix[0]
        if metric == 'MSD':
            sq_sum = matrix[1]
    if metric == 'MSD':
        return msd_condensed(c_sum, sq_sum, N, N_atoms)
    else:
            if 'c_threshold' in kwargs:
                c_threshold = kwargs['c_threshold']
            else:
                c_threshold = None
            if 'w_factor' in kwargs:
                w_factor = kwargs['w_factor']
            else:
                w_factor = 'fraction'
            esim_dict = gen_sim_dict(c_sum, n_objects=N, c_threshold=c_threshold, w_factor=w_factor)
            return 1 - esim_dict[metric]

def calculate_comp_sim(matrix, metric='MSD', N_atoms=1):
    """Complementary similarity is calculates the similarity of a set 
    without one object or observation using metrics in the extended comparison.
    The greater the complementary similarity, the more representative the object is.
    
    Parameters
    ----------
    matrix : array-like
        Data matrix.
    metric : {'MSD', 'RR', 'JT', 'SM', etc}
        Metric used for extended comparisons. See `extended_comparison` for details.
    N_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    
    Returns
    -------
    numpy.ndarray
        Array of complementary similarities for each object.
    """
    # this might need to be separated in 2, fir msd and for esim
    # it seems like most of this should be within the if, because it is inefficient to calculate sq_sum for the esim indices
    #if metric == 'MSD' and N_atoms == 1:
    #    warnings.warn('N_atoms is being specified as 1. Please change if N_atoms is not 1.')
    N = len(matrix)
    sq_data_total = matrix ** 2
    c_sum_total = np.sum(matrix, axis = 0)
    sq_sum_total = np.sum(sq_data_total, axis=0)
    comp_sims = []
    for i, object in enumerate(matrix):
        object_square = object ** 2
        value = extended_comparison([c_sum_total - object, sq_sum_total - object_square],
                                    data_type='condensed', metric=metric, 
                                    N=N - 1, N_atoms=N_atoms)
        comp_sims.append(value)
    comp_sims = np.array(comp_sims)
    return comp_sims

# We should add this sampling to BTS as well
def qupta_sample(data, n_objects = None, metric = 'MSD', nbins = 10, nsample = 100, hard_cap = True):
    """Representative sampling according to comp_sim values.
    
    Divides the range of comp_sim values in nbins and then
    uniformly selects nsample molecules, consecutively
    taking one from each bin
    """
    n = len(data)
    if nsample < 1:
        nsample = int(n * nsample)
    cs = calculate_comp_sim(data, metric='MSD', N_atoms=1)
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
                    d = mean_sq_dev(combined)
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

def shine(trajs, linkage='ward', metric='intra'):
    """Performs the clustering between the trajectories"""
    distances = gen_msdmatrix(trajs, metric=metric)
    linmat = squareform(distances, force='no', checks=True)
    link = linkage(linmat, linkage)
    return link

def cluster_postprocess(link, maxclusts=2, dendrogram=True)
    """Post-processing the SHINE clustering"""
    clusters = fcluster(link, t=maxclusts, criterion='maxclust')
    dendrogram(link)
    if dendrogram:
        plt.savefig('dendrogram.png')
    return clusters
                    
farthest = False
rep_sampling = True
div_sample = False
trim_init = False
trajs = {}
st = ''
for frac in np.linspace(0.05, 1, 96):
    print(int(frac * 100))
    for file in glob.glob("*.csv"):
        traj = file.split('_')[-1].split('.')[0]
        frames = np.genfromtxt(file, delimiter=',')
        if rep_sampling:
            if len(frames) >= 50:
                #frac = 0.30
                nf = int(frac * len(frames))
                s = quota_sample(frames, nbins = nf, nsample = nf)
                trajs[int(traj)] = frames[s]
            else:
                trajs[int(traj)] = frames
        elif div_sample:
            if len(frames) >= 50:
                #frac = 0.25
                nf = int(frac * len(frames))
                s = msd_div(frames, nf)
                trajs[int(traj)] = frames[s]
            else:
                trajs[int(traj)] = frames
        else:
            trajs[int(traj)] = frames
    links = ['ward']
    metrics = ['intra', 'inter', 'semi_sum', 'max', 'min', 'haus']#print(trajs)
    for metric in metrics:
        #print(metric)
        #s = ''
        for li in links:
            st = ''
            #s += '{}\n'.format(link)
            distances = gen_msdmatrix(trajs, metric=metric)
            distances += distances.T
            distances *= 0.5
            distances -= np.min(distances)
            np.fill_diagonal(distances, 0)
            linmat = squareform(distances, force='no', checks=True)
            link = linkage(linmat, "ward")
            clusters = fcluster(link, t=2, criterion='maxclust')
            #print(clusters)
            #dendrogram(link)
            #plt.show()
            #plt.savefig('{}_{}_{}.png'.format(l, metric, int(100 * frac)))
            #plt.close()
            
            
            #farthest_idx = np.unravel_index(distances.argmax(), (80, 80))
            #node1_dist, node2_dist = dist[(farthest_idx,)]
            
            #clustering = AgglomerativeClustering(n_clusters=2, metric='precomputed', memory=None, connectivity=None, compute_full_tree='auto', linkage=link)
            #clustering.fit(distances)
            #labels = clustering.labels_
            #labels = np.array(labels)
            label1 = np.where(clusters == 1)[0]
            label2 = np.where(clusters == 2)[0]
            #print(label1)
            if len(label1) < len(label2):
                labs = label1
            elif len(label1) > len(label2):
                labs = label2
            else:
                labs = ['tie']
            st += '\nLabels {}: '.format(int(frac * 100))
            for l in labs:
                st += '{:4}'.format(l)
            
            with open('ward_rep_scan_{}_{}.txt'.format(metric, li), 'a') as outfile:
                outfile.write(st)