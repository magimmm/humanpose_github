import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_neural_network():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Nakreslenie vrstiev
    layers = ['Input Layer', 'Dense(16, SELU)', 'Dense(8, ReLU)', 'Dense(1, Sigmoid)', 'Output Layer']
    colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral', 'lightgray']
    heights = [0.6, 0.4, 0.2, 0.4, 0.6]

    for i, (layer, color, height) in enumerate(zip(layers, colors, heights)):
        ax.add_patch(Rectangle((0.1, height - 0.05), 0.8, 0.1, edgecolor='black', facecolor=color))
        ax.text(0.5, height, layer, ha='center', va='center', fontsize=12)

    # Nakreslenie spojov
    connections = [(0.5, 0.6), (0.5, 0.4), (0.5, 0.2), (0.5, 0.4), (0.5, 0.6)]
    for start, end in zip(connections[:-1], connections[1:]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black')

    # Nastavenie osí
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Zobrazenie grafu
    plt.show()


# Vygenerovanie grafu
plot_neural_network()
