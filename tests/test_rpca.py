import numpy as np

from basic_ml import rpca


def test_rpca():
    decomp = rpca.RPCA()
    assert decomp is not None
    X = np.random.rand(3, 3)
    X = np.asfortranarray(X)
    print(X.shape, X.flags.writeable, X.dtype, X.flags.f_contiguous)
    decomp.run(X)
    print(decomp.getL())
    assert False == True
