import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import scipy.constants as const
import matplotlib.pyplot as plt

class ReadOpac:
    def __init__(self, ls, lp, lt, lf, lg):
        self.ls, self.lp, self.lt, self.lf, self.lg = ls, lp, lt, lf, lg
        
        assert self.ls>1, 'no files found'
        assert len(set(self.lf)) <= 1, 'frequency needs to match'
        assert len(set(self.lg)) <= 1, 'g grid needs to match'
        
        # initialize arrays:
        self.kcoeff = np.zeros((self.ls, self.lp.max(),self.lt.max(),self.lf[0],self.lg[0]))
        self.bin_edges = np.zeros((self.ls, self.lf[0]+1))
        self.bin_center = np.zeros((self.ls, self.lf[0]))
        self.weights = np.zeros((self.ls, self.lg[0]))
        self.T = np.zeros((self.ls, self.lt.max()))
        self.p = np.zeros((self.ls, self.lp.max()))
        self.spec = self.ls*[""]

    def setup_temp_and_pres(self, temp=None, pres=None):
        if pres is None:
            pmin = min([min(self.p[i,:self.lp[i]]) for i in range(self.ns)])
            pres = np.logspace(np.log10(pmin),np.log10(self.p.max()),len(self.p[0]))
        if temp is None:
            tmin = min([min(self.T[i,:self.lt[i]]) for i in range(self.ns)])
            temp = np.logspace(np.log10(tmin),np.log10(self.T.max()),len(self.T[0]))

        lp_new = len(pres)
        lt_new = len(temp)
        kcoeff_new = np.zeros((self.ns, lp_new, lt_new, self.lf[0], self.lg[0])) 
        for i in range(self.ns):
            interpolator = RegularGridInterpolator((self.p[i,:self.lp[i]],self.T[i,:self.lt[i]], self.bin_center, self.weights.cumsum()), self.kcoeff[i,:self.lp[i],:self.lt[i],:,:], method='linear', fill_value=0.0, bounds_error=False)
            x1,x2,x3,x4 = np.meshgrid(pres, temp, self.bin_center, self.weights.cumsum())
            eval_pts = np.array([x1.flatten(),x2.flatten(),x3.flatten(),x4.flatten()]).T
            kcoeff_new[i,:,:,:,:] = interpolator(eval_pts).reshape(kcoeff_new[i,:,:,:,:].shape)
        
        self.kcoeff = kcoeff_new
        self.T = temp
        self.p = pres
        self.lp = lp_new
        self.lt = lt_new

    def plot_opac(self, pres, temp, spec, ax=None, **plot_kwargs):
        if ax is None:
            ax = plt.gca()

        speci = self.spec.index(spec)
        if len(self.p.shape)>1:
            pi = np.searchsorted(self.p[speci], pres)
            ti = np.searchsorted(self.T[speci], temp)
            print('p:',self.p[speci,pi])
            print('T:',self.T[speci,ti])

        else:
            pi = np.searchsorted(self.p, pres)
            ti = np.searchsorted(self.T, temp)
            print('p:',self.p[pi])
            print('T:',self.T[ti])
        for fi in range(self.lf[0]):
            x = self.bin_edges[fi]+self.weights.cumsum()*(self.bin_edges[fi+1]-self.bin_edges[fi])
            ax.loglog(x,self.kcoeff[speci,pi,ti,fi,:], **plot_kwargs)



class ReadOpacChubb(ReadOpac):
    def __init__(self, files) -> None:
        ls = len(files)
        self._files = files
        # read meta data:
        lp,lt,lf,lg = np.empty(ls, dtype=int),np.empty(ls, dtype=int),np.empty(ls, dtype=int),np.empty(ls, dtype=int)
        for i, file in enumerate(files):
            with h5py.File(file) as f:
                lp[i],lt[i],lf[i],lg[i] = f['kcoeff'].shape

        super().__init__(ls,lp,lt,lf,lg)

    def read_opac(self):
        for i, file in enumerate(self._files):
            with h5py.File(file) as f:
                self.bin_edges[i,:] = np.array(f['bin_edges'])
                self.weights[i,:] = np.array(f['weights'])
                self.spec[i] = f['mol_name'][0].decode('ascii')
                self.T[i,:self.lt[i]] = np.array(f['t'])
                self.p[i,:self.lp[i]] = np.array(f['p'])
                # from cm2/mol to cm2/g:
                conversion_factor = float(1/(float(f['mol_mass'][0])*const.atomic_mass*1000))
                print(conversion_factor)
                self.kcoeff[i,:self.lp[i],:self.lt[i],:,:] = np.array(f['kcoeff'], dtype=float)*conversion_factor

        assert np.all(self.bin_edges[1:,:] == self.bin_edges[:-1,:]), 'frequency needs to match'
        assert np.all(self.weights[1:,:] == self.weights[:-1,:]), 'g grid needs to match'
        assert np.all(np.isclose(self.kcoeff,0.0)), 'only zeros loaded'
        
        self.bin_edges = self.bin_edges[0,:]
        self.bin_center = .5*(self.bin_edges[1:]+self.bin_edges[:-1])
        self.weights = self.weights[0,:]