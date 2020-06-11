import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot(zs, ys, labels, show=False, save_path=None, plot_by_class=True, **tsne_params):
    # TODO : MAKE PLOT PRETTIER : CUSTOMIZE COLORS,  ADD DIFFERENT SHAPES, ...
    # reducing zs to 2 dimensions using tsne. tsne parameters might need to be fine tuned
    # to achieve better results
    zs2 = TSNE(**tsne_params).fit_transform(zs)

    if not show:
        import matplotlib
        matplotlib.use('Agg')

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    cmap = plt.get_cmap('gist_rainbow', len(labels))
    colors = [cmap(i) for i in range(len(labels))]
    for k in range(len(labels)):
        m = ys == k
        ax.scatter(zs2[m, 0], zs2[m, 1], label='y=%s' % labels[k], s=6, c=cmap(k), alpha=0.5)
    ax.legend()
    fig.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'tsne.png'))

    if plot_by_class:
        fig = plt.figure(figsize=(10, 5))
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        for k in range(len(labels)):
            ax = plt.subplot(4, 5, k + 1)
            m = ys == k
            ax.scatter(zs2[m, 0], zs2[m, 1], s=2, c=cmap(k), alpha=0.5)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_title('y=%s'%labels[k])
        fig.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, 'tsne_by_class.png'))


def correlation_plot(zs, show=False, save_path=None):
    cov_matrix = np.corrcoef(zs.T)
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    plot_cov = cov_matrix
    for i in range(len(cov_matrix)):
        plot_cov[i, i] = np.nan
    plot_cov[abs(plot_cov) < 0.075] = 0.0
    plt.imshow(cov_matrix, cmap='inferno')
    # plt.imshow(cov_matrix, interpolation='None', cmap='hot')
    plt.title('z-correlation matrix')
    plt.colorbar()

    if save_path:
        plt.savefig(os.path.join(save_path, 'z-correlations.png'))