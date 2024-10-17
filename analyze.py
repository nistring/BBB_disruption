import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from statannotations.Annotator import Annotator
sns.set_theme(style="ticks", palette="pastel")
from scipy.stats import pearsonr, spearmanr


x = "z score"
dfs = pd.read_excel(f"results/table.xlsx", sheet_name=None)
for key, df in dfs.items():
    print(df)
    df.columns = ["Prognosis", "Qa", "volume(%)", "z score", 'vmin']
    df = df.replace({1: "good", 2: "poor"})
    # for y in ["Qa", "volume(%)", "z score"]:
    #     ax = sns.boxplot(x=x, y=y, data=df)
    #     annot = Annotator(ax, [("good", "poor")], data=df, x=x, y=y)
    #     annot.configure(test='Mann-Whitney-ls', text_format='star', loc='outside', verbose=2)
    #     annot.apply_test()
    #     ax, test_results = annot.annotate()
    #     plt.ylabel(y)
    #     plt.savefig(f"results/figure({y}).png")
    #     plt.clf()

    lm = sns.lmplot(data=df, x=x, y="Qa", hue="Prognosis", fit_reg=False)
    sns.regplot(data=df, x=x, y="Qa", scatter=False, ax=lm.axes[0, 0], robust=True)
    lm.set(xlabel=x, ylabel="log₁₀Qa", title='')
    # lm.axes[0, 0].set_xlim(left=0.02, right=0.11)

    r, p = pearsonr(df["Qa"], df[x], alternative="greater")
    lm.axes[0, 0].text(0.01, -1.5, "R² = {:.3f}, p = {:.4f}".format(r, p))
    plt.savefig(f"results/correlation_{key}.png")
    plt.clf()