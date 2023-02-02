import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import scipy.constants as const
import matplotlib.pyplot as plt


class ReadOpac:
    def __init__(self, ls, lp, lt, lf, lg):
        """Construct the reader. Setup all arrays."""

        self.ls, self.lp, self.lt, self.lf, self.lg = ls, lp, lt, lf, lg
        
        assert self.ls>1, 'no files found'
        assert len(set(self.lf)) <= 1, 'frequency needs to match'
        assert len(set(self.lg)) <= 1, 'g grid needs to match'
        
        # initialize arrays:
        self.kcoeff = np.zeros((self.ls, self.lp.max(),self.lt.max(),self.lf[0],self.lg[0]), dtype=np.float64)
        self.bin_edges = np.zeros(self.lf[0]+1)
        self.bin_center = np.zeros(self.lf[0])
        self.weights = np.zeros(self.lg[0])
        self.T = np.zeros((self.ls, self.lt.max()))
        self.p = np.zeros((self.ls, self.lp.max()))
        self.spec = self.ls*[""]

        # Initialize reduced arrays (will only be set during interpolation)
        self.pr = np.empty(self.lp.max())
        self.Tr = np.empty(self.lt.max())
        self.interp_done = False
        self._read_done = False

    def read_opac(self):
        """read in the opacity, dependent on the opac IO model."""
        self._read_done = True
        return NotImplementedError('to be implemented in childclass')

    def setup_temp_and_pres(self, temp=None, pres=None):
        """Interpolate kcoeffs to different pressure and temperature values."""

        assert self._read_done, 'run read_opac first'
        if pres is None:
            pmin = min([min(self.p[i,:self.lp[i]]) for i in range(self.ls)])
            pres = np.logspace(np.log10(pmin),np.log10(self.p.max()),len(self.p[0]))
        else:
            pres = np.array(pres)

        if temp is None:
            tmin = min([min(self.T[i,:self.lt[i]]) for i in range(self.ls)])
            temp = np.logspace(np.log10(tmin),np.log10(self.T.max()),len(self.T[0]))
        else:
            temp = np.array(temp)

        lp_new = self.ls*[len(pres)]
        lt_new = self.ls*[len(temp)]
        kcoeff_new = np.zeros((self.ls, lp_new[0], lt_new[0], self.lf[0], self.lg[0]))

        x1, x2, x3, x4 = np.meshgrid(pres, temp, self.bin_center, self.weights.cumsum(),indexing='ij')
        eval_pts = np.array([x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()])

        for i in range(self.ls):
            interpolator = RegularGridInterpolator((self.p[i,:self.lp[i]],self.T[i,:self.lt[i]], self.bin_center, self.weights.cumsum()), self.kcoeff[i,:self.lp[i],:self.lt[i],:,:], method='linear', fill_value=0.0, bounds_error=False)

            # EDGES are important !!
            assert np.all(self.p[i, :self.lp[i]-1] <= self.p[i, 1:self.lp[i]]), 'pressure array not sorted correctly'
            assert np.all(self.T[i, :self.lt[i]-1] <= self.T[i, 1:self.lt[i]]), 'temperature array not sorted correctly'

            pltl = np.logical_and(x1 < self.p[i, :self.lp[i]].min(), x2 < self.T[i, :self.lt[i]].min())
            psts = np.logical_and(x1 > self.p[i, :self.lp[i]].max(), x2 > self.T[i, :self.lt[i]].max())
            pstl = np.logical_and(x1 > self.p[i, :self.lp[i]].max(), x2 < self.T[i, :self.lt[i]].min())
            plts = np.logical_and(x1 < self.p[i, :self.lp[i]].min(), x2 > self.T[i, :self.lt[i]].max())

            kcoeff_new[i, :, :, :, :] = interpolator(eval_pts.T).reshape(kcoeff_new[i, :, :, :, :].shape)
            kcoeff_new[i][pltl] = np.repeat(self.kcoeff[i, 0, 0, :, :], repeats=int(pltl.sum()/self.lf[0]/self.lg[0]))
            kcoeff_new[i][psts] = np.repeat(self.kcoeff[i, -1, -1, :, :], repeats=int(psts.sum()/self.lf[0]/self.lg[0]))
            kcoeff_new[i][pstl] = np.repeat(self.kcoeff[i, -1, 0, :, :], repeats=int(pstl.sum()/self.lf[0]/self.lg[0]))
            kcoeff_new[i][plts] = np.repeat(self.kcoeff[i, 0, -1, :, :], repeats=int(plts.sum()/self.lf[0]/self.lg[0]))

        self.kcoeff = kcoeff_new
        self.pr = pres
        self.Tr = temp
        self.T = np.ones((self.ls,lt_new[0]))*temp
        self.p = np.ones((self.ls,lp_new[0]))*pres
        self.lp = lp_new
        self.lt = lt_new
        self.interp_done = True

    def plot_opac(self, pres, temp, spec, ax=None, **plot_kwargs):
        """Simple pltting of the opacity."""
        if ax is None:
            ax = plt.gca()

        speci = self.spec.index(spec)
        pi = np.searchsorted(self.p[speci], pres)-1
        ti = np.searchsorted(self.T[speci], temp)-1
        print('p:',self.p[speci,pi])
        print('T:',self.T[speci,ti])

        for fi in range(self.lf[0]):
            x = self.bin_edges[fi]+self.weights.cumsum()*(self.bin_edges[fi+1]-self.bin_edges[fi])
            ax.loglog(x,self.kcoeff[speci,pi,ti,fi,:], **plot_kwargs)


class ReadOpacChubb(ReadOpac):
    def __init__(self, files) -> None:
        """Construct the chubb reader."""
        ls = len(files)
        self._files = files
        # read meta data:
        lp,lt,lf,lg = np.empty(ls, dtype=int),np.empty(ls, dtype=int),np.empty(ls, dtype=int),np.empty(ls, dtype=int)
        for i, file in enumerate(files):
            with h5py.File(file) as f:
                lp[i],lt[i],lf[i],lg[i] = f['kcoeff'].shape

        super().__init__(ls,lp,lt,lf,lg)

    def read_opac(self):
        """Read in the kcoeff from h5 file."""
        bin_edges = np.empty((self.ls, self.lf[0]+1))
        weights = np.empty((self.ls, self.lg[0]))
        for i, file in enumerate(self._files):
            with h5py.File(file) as f:
                bin_edges[i,:] = np.array(f['bin_edges'])
                weights[i,:] = np.array(f['weights'])
                self.spec[i] = f['mol_name'][0].decode('ascii')
                self.T[i,:self.lt[i]] = np.array(f['t'])
                self.p[i,:self.lp[i]] = np.array(f['p'])
                # from cm2/mol to cm2/g:
                conversion_factor = 1/(float(f['mol_mass'][0])*const.atomic_mass*1000)
                self.kcoeff[i,:self.lp[i],:self.lt[i],:,:] = np.array(f['kcoeff'], dtype=np.float64)*conversion_factor

        assert np.all(bin_edges[1:,:] == bin_edges[:-1,:]), 'frequency needs to match'
        assert np.all(weights[1:,:] == weights[:-1,:]), 'g grid needs to match'

        self.bin_edges = bin_edges[0,:]
        self.bin_center = .5*(self.bin_edges[1:]+self.bin_edges[:-1])
        self.weights = weights[0,:]
        self._read_done = True