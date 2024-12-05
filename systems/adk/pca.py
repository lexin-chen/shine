import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(1.25)

all_frames = []
labels = []
for file in glob.glob("data/*.npy"):
    frames = np.load(file)
    file = file.split('/')[-1]
    if file.split('_')[0] == 'd':
        traj = int(file.split('_')[-1].split('.')[0]) + 200 - 1
        labels.extend([0 for _ in range(len(frames))])
    elif file.split('_')[0] == 'f':
        traj = int(file.split('_')[-1].split('.')[0]) - 1
        # append 0 for all frames in traj
        labels.extend([1 for _ in range(len(frames))])
    all_frames.append(frames)

pca_frames = np.concatenate(all_frames)
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(pca_frames)

# Plot
for i, unique_label in enumerate(np.unique(labels)):
    if unique_label == 0:
        plt.scatter(transformed_data[labels == unique_label, 0], 
                    transformed_data[labels == unique_label, 1], 
                    label=f'DIMS', s=10)
    elif unique_label == 1:
        plt.scatter(transformed_data[labels == unique_label, 0], 
                    transformed_data[labels == unique_label, 1], 
                    label=f'FRODA', s=10)
plt.xlabel('PC1', fontsize=12, fontweight='bold')
plt.ylabel('PC2', fontsize=12, fontweight='bold')
plt.legend(fontsize=12)
plt.savefig('pca.png', dpi=500, bbox_inches='tight', pad_inches=0.1, transparent=True)
