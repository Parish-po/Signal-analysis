"""viz.py — Plot helpers (box/violin/scatter)."""
from typing import Sequence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def box_violin(df: pd.DataFrame, x: str, y: str, hue: str | None = None, kind: str = "violin", save_path: str | None = None):
    plt.figure()
    if kind == "violin":
        sns.violinplot(data=df, x=x, y=y, hue=hue, cut=0)
    else:
        sns.boxplot(data=df, x=x, y=y, hue=hue)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    return plt.gcf()

def pair_scatter(df: pd.DataFrame, cols: Sequence[str], save_path: str | None = None):
    g = sns.pairplot(df[cols], corner=True)
    if save_path:
        g.savefig(save_path, dpi=200)
    return g
