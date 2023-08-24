import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (15,5)
plt.figure()

results = pd.read_csv("../datasets/result.csv")

results.plot(x="date", y=["a2c",  "dji"],
             title="Account Value", xlabel="Date", ylabel="Account Value ($)", grid=True, rot=45)
plt.show()