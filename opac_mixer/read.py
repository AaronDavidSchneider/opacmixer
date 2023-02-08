import numpy as np
import numba
import h5py
import scipy.constants as const
import matplotlib.pyplot as plt


@numba.njit(nogil=True, fastmath=True, cache=True)
def interp_2d(temp_old, press_old, temp_new, press_new, kcoeff, ls, lf, lg, lt_old, lp_old, lt_new, lp_new):
    """Function that interpolates to correct pressure and temperature."""
    kcoeff_new = np.empty((ls, lp_new, lt_new, lf, lg), dtype=np.float64)

    for speci in numba.prange(ls):
        to_i = temp_old[speci, :lt_old[speci]]
        po_i = press_old[speci, :lp_old[speci]]
        kcoeff_i = kcoeff[speci, :lp_old[speci], :lt_old[speci]]

        for gi in numba.prange(lg):
            for freqi in numba.prange(lf):
                # reset temporary array
                p_interp = np.empty((lp_new, lt_old[speci]), dtype=np.float64)
                pt_interp = np.empty((lp_new, lt_new), dtype=np.float64)

                # interp to new pressure (for all temperatures)
                for ti in numba.prange(lt_old[speci]):
                    p_interp[:, ti] = np.interp(press_new, po_i, kcoeff_i[:, ti, freqi, gi])
                # interp to new temperature (for all -new- pressures)
                for pi in numba.prange(lp_new):
                    pt_interp[pi, :] = np.interp(temp_new, to_i, p_interp[pi, :])

                # Do edges
                for pi in numba.prange(lp_new):
                    for ti in numba.prange(lt_new):
                        if press_new[pi] < min(po_i) and temp_new[ti] < min(to_i):
                            pt_interp[pi, ti] = kcoeff[speci, np.argmin(po_i), np.argmin(to_i), freqi, gi]
                        elif press_new[pi] < min(po_i) and temp_new[ti] > max(to_i):
                            pt_interp[pi, ti] = kcoeff[speci, np.argmin(po_i), np.argmax(to_i), freqi, gi]
                        elif press_new[pi] > max(po_i) and temp_new[ti] < min(to_i):
                            pt_interp[pi, ti] = kcoeff[speci, np.argmax(po_i), np.argmin(to_i), freqi, gi]
                        elif press_new[pi] > max(po_i) and temp_new[ti] > max(to_i):
                            pt_interp[pi, ti] = kcoeff[speci, np.argmax(po_i), np.argmax(to_i), freqi, gi]

                kcoeff_new[speci, :, :, freqi, gi] = pt_interp

    return kcoeff_new


class ReadOpac:
    def __init__(self, ls, lp, lt, lf, lg):
        """Construct the reader. Setup all arrays."""

        self.ls, self.lp, self.lt, self.lf, self.lg = ls, lp, lt, lf, lg
        
        assert self.ls>1, 'no files found'
        assert len(set(self.lf)) <= 1, 'frequency needs to match'
        assert len(set(self.lg)) <= 1, 'g grid needs to match'
        
        # initialize arrays:
        self.kcoeff = np.zeros((self.ls, self.lp.max(),self.lt.max(),self.lf[0],self.lg[0]), dtype=np.float64)
        self.bin_edges = np.zeros(self.lf[0]+1, dtype=np.float64)
        self.bin_center = np.zeros(self.lf[0], dtype=np.float64)
        self.weights = np.zeros(self.lg[0], dtype=np.float64)
        self.T = np.zeros((self.ls, self.lt.max()), dtype=np.float64)
        self.p = np.zeros((self.ls, self.lp.max()), dtype=np.float64)
        self.spec = self.ls*[""]

        # Initialize reduced arrays (will only be set during interpolation)
        self.pr = np.empty(self.lp.max(), dtype=np.float64)
        self.Tr = np.empty(self.lt.max(), dtype=np.float64)
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

        self.kcoeff = interp_2d(self.T, self.p, temp, pres, self.kcoeff, self.ls, self.lf[0], self.lg[0], self.lt, self.lp, lt_new[0], lp_new[0])

        self.pr = pres
        self.Tr = temp
        self.T = np.ones((self.ls,lt_new[0]), dtype=np.float64)*temp
        self.p = np.ones((self.ls,lp_new[0]), dtype=np.float64)*pres
        self.lp = lp_new
        self.lt = lt_new
        self.interp_done = True

    def remove_sparse_frequencies(self):
        """Check for zeros in the opacity and remove them"""

        # Search for the zeros in every species
        nonzero_index = np.empty((self.ls, self.lf[0]))
        for i in range(self.ls):
            nonzero_index[i] = np.all(self.kcoeff[i, :self.lp[i], :self.lt[i], :, :], axis=(0, 1, 3))

        # Search for common zeros in every species
        nonzero_index = np.all(nonzero_index, axis=0)

        # Construct the array for the edges
        edges_nonzero = np.ones(self.lf[0] + 1)  # default case, no zeros
        if not nonzero_index[0] or not nonzero_index[-1]:
            # We need to add the outer borders to the edges
            edges_nonzero = np.append(nonzero_index, 1.0)
        else:
            if np.count_nonzero(nonzero_index) != self.lf[0]:
                # We want that the zeros start at the frequency edges
                # nonzero_index[-1] or nonzero_index[0] would then need to be zero
                raise ValueError('zeros in the middle. Cant handle that. It makes no sense.')

        # adapt the members accordingly
        self.lf = np.repeat(np.count_nonzero(nonzero_index), self.ls)
        self.bin_edges = self.bin_edges[np.asarray(edges_nonzero, dtype=bool)]
        self.bin_center = .5 * (self.bin_edges[1:] + self.bin_edges[:-1])
        self.kcoeff = self.kcoeff[:, :, :, np.asarray(nonzero_index, dtype=bool), :]

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
        bin_edges = np.empty((self.ls, self.lf[0]+1), dtype=np.float64)
        weights = np.empty((self.ls, self.lg[0]), dtype=np.float64)
        for i, file in enumerate(self._files):
            with h5py.File(file) as f:
                bin_edges[i,:] = np.array(f['bin_edges'], dtype=np.float64)
                weights[i,:] = np.array(f['weights'], dtype=np.float64)
                self.spec[i] = f['mol_name'][0].decode('ascii')
                self.T[i,:self.lt[i]] = np.array(f['t'], dtype=np.float64)
                self.p[i,:self.lp[i]] = np.array(f['p'], dtype=np.float64)
                # from cm2/mol to cm2/g:
                conversion_factor = 1/(float(f['mol_mass'][0])*const.atomic_mass*1000)
                kcoeff = np.array(f['kcoeff'], dtype=np.float64)*conversion_factor
                self.kcoeff[i,:self.lp[i],:self.lt[i],:,:] = kcoeff

        assert np.all(bin_edges[1:,:] == bin_edges[:-1,:]), 'frequency needs to match'
        assert np.all(weights[1:,:] == weights[:-1,:]), 'g grid needs to match'
        self.bin_edges = bin_edges[0,:]

        self.remove_sparse_frequencies()

        self.weights = weights[0,:]

        self._read_done = True