import matplotlib.pyplot as plt
import numpy as np

delta_vaihingen = np.array([0.14, 0.73, 0.12, 0.80, 0.22, 0.44, 0.61, 0.33, 0.07])
delta_potsdam = np.array([-0.03, 0.21, 0.04, 0.15, 0.01, 0.08, 0.08, 0.26, 0.04])
delta_loveda = np.array([0.44, 0.07, 0.47, 0.04, 0.01, 0.03, 0.20, 0.62, 0.18])

dataset_sizes_gb = [0.47, 2.22, 8.93]
dataset_labels = ['Vaihingen', 'Potsdam', 'LoveDA']
box_data = [delta_vaihingen, delta_potsdam, delta_loveda]
colors = ['pink', 'skyblue', 'lightgreen']

fig, ax = plt.subplots(figsize=(9, 3.5))
positions = dataset_sizes_gb
box = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True)

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

for i, (data, x_pos) in enumerate(zip(box_data, positions)):
    max_val = np.max(data)
    mean_val = np.mean(data)
    # min_val = np.min(data)
    median_val = np.median(data)

    # std_val = np.std(data)
    # y_text = mean_val + std_val + 0.02  

    max_mag = '+' if max_val > 0 else ''
    mean_mag = '+' if mean_val > 0 else ''
    # min_mag = '+' if min_val > 0 else ''
    median_mag = '+' if median_val > 0 else ''

    ax.text(x_pos + 0.2, max_val,
            f"Max: {max_mag}{max_val:.2f}\nMean: {mean_mag}{mean_val:.2f}\nMedian: {median_mag}{median_val:.2f}",
            fontsize=12, weight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='none'))

ax.set_xlabel("Dataset Size", fontsize=12)
ax.set_ylabel("Performance Gain (mIoU)", fontsize=12)
ax.set_xticks(dataset_sizes_gb)
ax.set_xticklabels([f"{label}\n({size} GB)\n" for label, size in zip(dataset_labels, dataset_sizes_gb)])

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()