import numpy as np
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces

from basic_ml import rpca


def test_decomposition(steps=50, plot_results=False):
    rng = RandomState(0)

    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    faces = faces[:100]
    n_samples, n_features = faces.shape
    faces_centered = faces - faces.mean(axis=0)
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    decomp = rpca.RPCA(maxCount = int(100), thresholdScale = 2.5e-8)
    X = np.asfortranarray(faces_centered.astype(np.float64))
    decomp.run(X)
    L = np.asarray(decomp.getL())
    S = np.asarray(decomp.getS())
    if plot_results:
        import matplotlib.pyplot as plt
        n_row, n_col = 2, 3
        image_shape = (64, 64)
        def plot_gallery(title, L, S, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
            """
            Helper function to plot a gallery of portraits. Taken from sklearn and modified.
            """
            fig, axs = plt.subplots(
                nrows=n_row,
                ncols=n_col,
                figsize=(2.0 * n_col, 2.3 * n_row),
                facecolor="white",
                constrained_layout=True,
            )
            fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
            fig.set_edgecolor("black")
            fig.suptitle(title, size=16)
            for i in range(n_col):
                for j, mat in enumerate((L, S)):
                    vec = mat[i, :]
                    vmax = max(vec.max(), -vec.min())
                    im = axs[j, i].imshow(
                        vec.reshape(image_shape),
                        cmap=cmap,
                        interpolation="nearest",
                        vmin=-vmax,
                        vmax=vmax,
                    )
                    axs[j, i].axis("off")
            fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
        plot_gallery(
            "RPCA -- dense (top), sparse (bottom)", L, S
        )
        plt.show()


def test_rpca():
    decomp = rpca.RPCA()
    assert decomp is not None
    X = np.random.rand(3, 3)
    X = np.asfortranarray(X)
    print(decomp.getL())
    print(X.shape, X.flags.writeable, X.dtype, X.flags.f_contiguous)
    decomp.run(X)
    print(decomp.getL())


if __name__ == "__main__":
    # test_rpca()
    test_decomposition(plot_results=True)    
