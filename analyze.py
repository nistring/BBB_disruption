import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style="ticks", palette="pastel")

# df = pd.read_excel("histogram.xlsx", index_col=0)
# #df = df.loc[df["layer"] >= 12]
# g = sns.catplot(
#     data=df, kind="bar",
#     x="layer", y=0, hue="prognosis", palette="dark", alpha=.6, height=6
# )
# g.despine(left=True)
# g.set_axis_labels("", "Volume(%)")
# g.legend.set_title("")
# plt.savefig("histogram.png")


df = pd.read_excel("results.xlsx", index_col=0)
sns.boxplot(x= "1, good 2, poor", y="volume(%)", data=df)
plt.ylabel("mean enhancement (%)")
plt.savefig("results.png")