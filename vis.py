import matplotlib.pyplot as plt
import numpy as np

delta_vaihingen = np.array([77.77 - 77.63, 78.26 - 77.53, 79.71 - 79.59, 77.76 - 76.96, 79.27 - 79.05, 74.76 - 74.32])
delta_potsdam = np.array([83.03 - 83.06, 83.16 - 82.95, 84.51 - 84.60, 83.23 - 83.08, 84.43 - 84.42, 81.42 - 81.34])
delta_loveda = np.array([43.55 - 43.11, 43.36 - 43.29, 43.82 - 43.35, 46.49 - 46.45, 47.92 - 47.91, 40.60 - 40.57])

dataset_sizes_gb = [0.47, 2.22, 8.93]
dataset_labels = ['Vaihingen', 'Potsdam', 'LoveDA']
box_data = [delta_vaihingen, delta_potsdam, delta_loveda]
colors = ['orange', 'skyblue', 'lightgreen']

fig, ax = plt.subplots(figsize=(9, 4.5))
positions = dataset_sizes_gb
box = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True)

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

for i, (data, x_pos) in enumerate(zip(box_data, positions)):
    max_val = np.max(data)
    mean_val = np.mean(data)
    min_val = np.min(data)
    # std_val = np.std(data)
    # y_text = mean_val + std_val + 0.02  

    max_mag = '+' if max_val > 0 else ''
    min_mag = '+' if min_val > 0 else ''
    mean_mag = '+' if mean_val > 0 else ''

    ax.text(x_pos + 0.2, max_val,
            f"Max: {max_mag}{max_val:.2f}\nMean: {mean_mag}{mean_val:.2f}\nMin: {min_mag}{min_val:.2f}",
            fontsize=12, weight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='none'))

ax.set_xlabel("Dataset Size", fontsize=12)
ax.set_ylabel("Performance Gain (mIoU)", fontsize=12)
ax.set_xticks(dataset_sizes_gb)
ax.set_xticklabels([f"{label}\n({size} GB)" for label, size in zip(dataset_labels, dataset_sizes_gb)])

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()