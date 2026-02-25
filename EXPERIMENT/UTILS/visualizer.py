import matplotlib.pyplot as plt


def comparison_curve(
    histories: list[list[float]],
    labels: list[str],
    criterion: str,
    title: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    for hist, label in zip(histories, labels):
        plt.plot(hist, label=label)
    plt.xlabel('EPOCH')
    plt.ylabel(criterion)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def criterion_curve(
    history: list[float],
    label: str,
    criterion: str,
    title: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(history, label=label)
    plt.xlabel('EPOCH')
    plt.ylabel(criterion)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()