from pca import pca, calc_factor_loadings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tsne import tsne
from tsne_orig import tsne_orig
from sklearn.manifold import TSNE
import matplotlib.animation as animation

class Plotter:
  def __init__(self, sample_labels):
    self.label_to_colors = {
      'Days0-3': 'red',
      'Days6-9': 'orange',
      'Days12-15': 'yellow',
      'Days18-21': 'green',
      'Days24-27': 'blue',
    }
    self.sample_labels = sample_labels

  def plot(self, embeded):
    fig, ax = plt.subplots()
    ax.scatter(embeded[:,0], embeded[:,1], alpha=0.5)
    plt.show()
    plt.close()

  def plot_with_labels(self, embeded):
    fig, ax = plt.subplots()
    for label in self.label_to_colors.keys():
      mask = np.array([x==label for x in self.sample_labels])
      x = embeded[mask]
      ax.scatter(x[:,0],x[:,1],color=self.label_to_colors[label],label=label,alpha=0.5, cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='upper left')
    plt.show()
    plt.close()
    
  def plot_with_expression(self, embeded, gene_expression):
    fig, ax = plt.subplots()
    sc = ax.scatter(embeded[:,0], embeded[:,1], alpha=0.5, c=gene_expression)
    plt.colorbar(sc)
    plt.show()
    plt.close()

  def plot_animation(self, embededs):
    fig, ax = plt.subplots()
    ims = []
    for embeded in embededs:
      scatters = []
      for label in self.label_to_colors.keys():
        mask = np.array([x==label for x in self.sample_labels])
        x = embeded[mask]
        p = ax.scatter(x[:,0],x[:,1],color=self.label_to_colors[label],label=label,alpha=0.5, cmap='viridis')
        scatters.append(p)
      ims.append(scatters)
    #ax.set_xlabel('PC1')
    #ax.set_ylabel('PC2')
    #ax.legend(loc='lower right')
    ani = animation.ArtistAnimation(fig, ims, interval=1)
    plt.show()
    ani.save("../dataset/tsne_animation.gif", writer="imagemagick")

def main_pca():
  scaled_df_path = '../dataset/df_scaled.gz'
  df_scaled = pd.read_pickle(scaled_df_path)
  
  X = df_scaled.values.T
  transformed = pca(X, n_components=2)

  #plot results
  sample_labels = list(df_scaled.columns.get_level_values(0))
  plotter = Plotter(sample_labels)
  plotter.plot(transformed)
  plotter.plot_with_labels(transformed)

  #calculate factor loadings
  gene_labels = list(df_scaled.index)
  gene_pc1 = calc_factor_loadings(X, transformed, gene_labels, nth_factor=0)
  gene_pc2 = calc_factor_loadings(X, transformed, gene_labels, nth_factor=1)

  #plot gene expression
  plotter.plot_with_expression(transformed, df_scaled.loc[gene_pc1,:])
  plotter.plot_with_expression(transformed, df_scaled.loc[gene_pc2,:])

def main_tsne():
  scaled_df_path = '../dataset/df_scaled.gz'
  df_scaled = pd.read_pickle(scaled_df_path)
  
  X = df_scaled.values.T
  transformed = tsne(X, perplexity=30, n_components=2, niter=100, eta=100, return_all_iter=True, initialize_pca=True)
  #transformed = tsne_orig(X, perplexity=30, d=2, niter=100, eta_init=100)
  #np.save('../dataset/tsne.npy', transformed)

  #plot results
  sample_labels = list(df_scaled.columns.get_level_values(0))
  plotter = Plotter(sample_labels)
  #for i in range(len(transformed)):
  #  #plotter.plot(transformed[i,:,:])
  #  plotter.plot_with_labels(transformed[i])
  plotter.plot_with_labels(transformed[-1])
  plotter.plot_animation(transformed)

def main_tsne_sk():
  scaled_df_path = '../dataset/df_scaled.gz'
  df_scaled = pd.read_pickle(scaled_df_path)
  
  X = df_scaled.values.T
  transformed =TSNE(n_components=2, perplexity=100, init="random").fit_transform(X)

  #plot results
  sample_labels = list(df_scaled.columns.get_level_values(0))
  plotter = Plotter(sample_labels)
  plotter.plot(transformed)
  plotter.plot_with_labels(transformed)

if __name__=='__main__':
  #main_pca()
  main_tsne()
  #main_tsne_sk()
