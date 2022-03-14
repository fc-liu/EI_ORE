# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch


def tsne_cpp():
    from tsnecuda import TSNE

    x = np.random.rand(200, 128)
    y = np.random.randint(0, 50, 200)
    y_g = np.random.randint(0, 50, 200)

    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(20, 20))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
    plt.savefig('figures/test.jpg')
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_g)
    plt.savefig('figures/test2.jpg')
    # plt.show()


def tsne_pytorch():
    from tsne import tsne

    x = torch.rand(200, 128)
    y = np.random.randint(0, 50, 200)
    y_g = np.random.randint(0, 50, 200)

    X_tsne = tsne(x, initial_dims=128)

    X_tsne = X_tsne.detach().cpu().numpy()
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(20, 20))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
    plt.savefig('figures/test.jpg')
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_g)
    plt.savefig('figures/test2.jpg')


def tsne_multi_core():
    from sklearn.datasets import load_digits
    from MulticoreTSNE import MulticoreTSNE as TSNE
    from matplotlib import pyplot as plt

    digits = load_digits()
    embeddings = TSNE(n_jobs=4).fit_transform(digits.data)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.figure(figsize=(80, 80))
    plt.scatter(vis_x, vis_y, s=40, c=digits.target,
                cmap=plt.cm.get_cmap("jet", 250), marker='.')
    plt.colorbar(ticks=range(250))
    # plt.clim(-0.5, 9.5)
    plt.savefig('figures/mnist_1.jpg')

    plt.figure(figsize=(80, 80))
    plt.scatter(vis_x, vis_y,  s=40, c=digits.target,
                cmap=plt.cm.get_cmap("jet", 250), marker='o')
    plt.colorbar(ticks=range(250))
    # plt.clim(-0.5, 9.5)
    plt.savefig('figures/mnist_2.jpg')
    # plt.show()


if __name__ == "__main__":
    # tsne_pytorch()
    tsne_multi_core()
