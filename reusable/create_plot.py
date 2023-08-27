import matplotlib.pyplot as plt
import pandas as pd


def plot_results(path_in: str):
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()

    results = pd.read_csv(path_in)

    results.plot(x="date", y=["a2c", "dji"],
                 title="Account Value", xlabel="Date", ylabel="Account Value ($)", grid=True, rot=45)
    plt.show()


if __name__ == '__main__':
    plot_results('results/a2c/results.csv')