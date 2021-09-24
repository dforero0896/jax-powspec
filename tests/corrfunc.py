import numpy as np
from Corrfunc.theory.xi import xi as xi_corrfunc
import proplot as pplt
import os

bins = np.linspace(1e-5, 200, 41)
if not os.path.isfile("data/corrfunc_xi.npy"):
    data = np.load("data/lognormal_nobao_corrected.npy")
    box_size = 1000.
    results = xi_corrfunc(box_size, 8, bins, data[:,0], data[:,1], data[:,2], output_ravg=True, verbose=True)
    np.save("data/corrfunc_xi.npy", results)
else:
    results = np.load("data/corrfunc_xi.npy")

fig, ax = pplt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)

ax[0].plot(results['ravg'], results['ravg']**2 * results['xi'])

fig.savefig("plots.corrfunnc_xi.png", dpi=300)



