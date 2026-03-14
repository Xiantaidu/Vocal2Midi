import matplotlib.pyplot as plt


def draw_distribution(title, x_label, y_label, items: list, values: list, zoom=0.8):
    plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    plt.grid()
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
