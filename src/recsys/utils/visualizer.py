import matplotlib.pyplot as plt


def main(
    record: list[float],
    label: str,
    criterion: str,
    title: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(record, label=label)
    plt.xlabel('EPOCH')
    plt.ylabel(criterion)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()