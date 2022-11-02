from config import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = read_results()

    for datum in data:
        field = datum["losses"]
        plt.plot(range(len(field)), field, label=datum["name"])
    plt.xlabel("Rounds (Global Epoch)")
    plt.ylabel("Loss")
    plt.title("Validation Loss vs. Global Epoch")
    plt.legend()
    plt.show()

    for datum in data:
        field = datum["accuracies"]
        plt.plot(range(len(field)), field, label=datum["name"])
    plt.xlabel("Rounds (Global Epoch)")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs. Global Epoch")
    plt.legend()
    plt.show()
