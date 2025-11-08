from django.db import models

# Create your models here.
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

def risk_parity(df):
    # Step 1: Compute correlation and distance matrices
    corr = df.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    
    # Step 2: Hierarchical clustering
    link = linkage(squareform(dist), method='single')
    dendro = dendrogram(link, no_plot=True)
    ordered_indices = leaves_list(link)
    ordered_columns = df.columns[ordered_indices]

    # Step 3: Quasi-diagonalization
    quasi_corr = corr.loc[ordered_columns, ordered_columns]

    # Step 4: Generate visuals (dendrogram + heatmap)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    dendrogram(link, labels=df.columns, ax=ax1)
    plt.title("Dendrogram")
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    dendro_b64 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    im = ax2.imshow(quasi_corr, cmap='viridis')
    plt.xticks(ticks=np.arange(len(ordered_columns)), labels=ordered_columns, rotation=90)
    plt.yticks(ticks=np.arange(len(ordered_columns)), labels=ordered_columns)
    plt.colorbar(im)
    plt.title("Quasi-Diagonalized Correlation Matrix")
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    heatmap_b64 = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)

    # Step 5: Compute equal weights (dummy HRP output for now)
    weights = pd.Series(1 / len(df.columns), index=ordered_columns)
    
    # Step 6: Convert weights to HTML
    results_html = weights.to_frame(name='Weight').to_html(classes='min-w-full text-sm text-left border')

    return results_html, dendro_b64, heatmap_b64



