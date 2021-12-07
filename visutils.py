from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visu_pca_2d(df, labels, name):
    # principal component analysis reduced to 2 axis for 2 visualization
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(df.drop(columns=labels))
    df['pca-1'] = pca_res[:,0]
    df['pca-2'] = pca_res[:,1]
    # plot clustering results in 2D
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-1", y="pca-2",
        hue=labels,
        data=df,
        legend="full",
        alpha=0.3)
    plt.title(name + "_2D visu with PCA")
    plt.savefig(name+'_visu_pca_2d.png')

def visu_pca_3d(df, labels, name) -> None:
    # principal component analysis reduced to 3 axis for 3 visualization
    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(df.drop(columns=labels))
    df['pca-1'] = pca_res[:,0]
    df['pca-2'] = pca_res[:,1]
    df['pca-3'] = pca_res[:,2]
    plt.figure(figsize=(16,10))
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-1"],
        ys=df["pca-2"],
        zs=df["pca-3"],
        c=df[labels])
    # plot clustering results in 2D
    ax.set_xlabel('pca-1')
    ax.set_ylabel('pca-2')
    ax.set_zlabel('pca-3')
    plt.title(name + "_3D visu with PCA")
    plt.savefig(name+'_visu_pca_3d.png')

def visu_tsne_2d(df, labels, name) -> None:
    # TSNE reduced to 2 axis for 2 visualization
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state = 42)
    tsne_results = tsne.fit_transform(df.drop(columns=labels))
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=df[labels],
        data=df,
        legend="full",
        alpha=0.3)
    # plot clustering results in 2D
    plt.title(name + "_2D visu with t-SNE")
    plt.savefig(name+'_visu_tsne_2d.png')



