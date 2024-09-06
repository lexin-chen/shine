import sys
sys.path.insert(0, "../../../")
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from shine.src.SHINE import cluster_postprocess

linmat = np.load('linmat.npy')
link = linkage(linmat, 'ward')
clusters = cluster_postprocess(link)
# print all the indices of the clusters with each unique label
label_indices = []
for i, cluster in enumerate(np.unique(clusters)):
    label_indices.append(np.where(clusters == cluster)[0])

def group_consecutive_indices(indices):

    # Sort the indices to ensure they are in order
    indices = sorted(indices)

    # Initialize the result list and the first group
    result = []
    start = indices[0]
    end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            # If the current index is consecutive, update the end
            end = indices[i]
        else:
            # If the current index is not consecutive, add the current group to the result
            if start == end:
                result.append(f"{start}")
            else:
                result.append(f"{start}-{end}")
            # Start a new group
            start = indices[i]
            end = indices[i]

    # Add the last group to the result
    if start == end:
        result.append(f"{start}")
    else:
        result.append(f"{start}-{end}")

    return ", ".join(result)

# Create custom labels for the clusters
custom_labels = [f"{group_consecutive_indices(cluster)}" for i, cluster in enumerate(label_indices)]
print(custom_labels)
# put this on the x-axis of the dendrogram
dendrogram(link)
num_ticks = 2  
# Calculate the positions for the ticks
tick_positions = np.linspace(100, 500, num_ticks, dtype=int)
tick_labels = [custom_labels[0], custom_labels[-1]]  # First and last labels

ax = plt.gca()
ax.set_xticks(tick_positions)
ax.set_xticklabels(["", ""])  # Clear existing labels

# Add colored text at the tick positions
ax.text(tick_positions[0], -0.03, tick_labels[0], color='blue', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=8, fontweight='bold')
ax.text(tick_positions[1], -0.03, tick_labels[1], color='red', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=8, fontweight='bold')

# Set the x-ticks with the specified positions and labels
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=0, fontsize=8, fontweight='bold', color='white')
plt.savefig('dendrogram.png')