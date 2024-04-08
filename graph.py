import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Data
models = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']
metrics = ['MSE', 'PCP', 'PCK', 'MPJPE', 'Čas (sekundy)', 'Chýbajúce \n landmarky']
values = {
    'yolov8n-pose': [750.71, 100.0, 92.54, 29.84, 12.60, 0.75],
    'yolov8s-pose': [669.96, 100.0, 92.62, 28.44, 19.82, 0.8],
    'yolov8m-pose': [572.34, 100.0, 93.38, 25.71, 35.16, 0.75],
    'yolov8l-pose': [727.32, 100.0, 93.15, 29.21, 60.97, 0.77],
    'yolov8x-pose': [588.22, 100.0, 92.69, 26.51, 90.14, 0.84]
}

# Define green-blue color palette
green_blue_colors =[ '#00ccff', '#34b393','#0099cc','#187d6e', '#0077b3', '#004080']
cmap = ListedColormap(green_blue_colors)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.15
index = np.arange(len(models))

for i, metric in enumerate(metrics):
    ax.bar(index + i * bar_width, [values[model][i] for model in models], bar_width, label=metric, color=cmap(i))

ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Výkon', fontsize=14)
ax.set_title('Porovnanie výkonu modelov YOLOv8-pose', fontsize=16)
ax.set_xticks(index + 2 * bar_width)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12)
ax.set_yscale('log')  # Set y-axis scale to logarithmic

plt.tight_layout()
plt.show()
