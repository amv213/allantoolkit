import allantoolkit
import numpy as np

minimum_ns = 1

taus = np.arange(1, 499)
afs = taus.copy()
vars = np.random.rand(len(taus))

ns = np.zeros(len(taus))
ns[0] = len(ns)
last_i = 0
for i in range(1, len(ns)):
    ns[i] = max(ns[i-1] // 2, 1.0)
    if ns[i] > minimum_ns:
        last_i = i + 1


def test_4params():
    afs2, taus2, ns2, vars2 = allantoolkit.utils.remove_small_ns(
        afs=afs, taus=taus, ns=ns, vars=vars)

    assert np.array_equal(afs2, taus[:last_i])
    assert np.array_equal(taus2, taus[:last_i])
    assert np.array_equal(ns2, ns[:last_i])
    assert np.array_equal(vars2, vars[:last_i])
