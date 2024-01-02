import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from statannotations.Annotator import Annotator
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

x = "1, good 2, poor"
for y in ["volume(%)", "z score"]:
    df = pd.read_excel("results/table.xlsx", index_col=0)
    print(df)
    ax = sns.boxplot(x=x, y=y, data=df)
    annot = Annotator(ax, [(1, 2)], data=df, x=x, y=y)
    annot.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
    annot.apply_test()
    ax, test_results = annot.annotate()
    plt.ylabel(y)
    plt.savefig(f"results/figure({y}).png")
    plt.clf()