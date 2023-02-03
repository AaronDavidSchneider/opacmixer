import numba
import numpy as np
import time
from functools import partial
from multiprocessing.pool import Pool

DEFAULT_METHOD = 'RORR'


@numba.njit(nogil=True, fastmath=True, cache=True)
def resort_rebin_njit(ggrid, kout_conv, k1, k2, weights_conv, kout, Np, Nt, Nf, Ng):
    for pi in numba.prange(Np):
        for ti in numba.prange(Nt):
            for freqi in numba.prange(Nf):
                if (k1[pi, ti, freqi, -1] - k2[pi, ti, freqi, 0]) * (k2[pi, ti, freqi, -1] - k1[pi, ti, freqi, 0]) < 0.:
                    if k1[pi, ti, freqi,-1] > k2[pi, ti, freqi, -1]:
                        kout[pi, ti, freqi, :] = k1[pi, ti, freqi, :]
                    else:
                        kout[pi, ti, freqi, :] = k2[pi, ti, freqi, :]
                    continue
                index_sort = np.argsort(kout_conv[pi, ti, freqi])
                kout_conv_resorted = kout_conv[pi, ti, freqi][index_sort]
                weights_resorted = weights_conv[index_sort]
                g_resorted = weights_resorted.cumsum()
                kout[pi, ti, freqi, :] = np.interp(ggrid, g_resorted,
                                                   kout_conv_resorted)
    return kout


@numba.njit(nogil=True, fastmath=True, cache=True)
def _add_rorr_njit(ktable, weights, mmr, Ns, Np, Nt, Nf, Ng):
    """
    Add up ktables by random overlap with resorting and rebinning.
    - The complete nonpython version, slightly slower compared to CombineOpac._add_rorr -
    """
    kout = np.zeros((Np, Nt, Nf, Ng))
    k1 = np.zeros((Np, Nt, Nf, Ng))
    k2 = np.zeros((Np, Nt, Nf, Ng))
    kout_conv = np.zeros((Np, Nt, Nf, Ng*Ng))

    for freqi in numba.prange(Nf):
        for gi in numba.prange(Ng):
            kout[:, :, freqi, gi] = mmr[0, :, :] * ktable[0, :, :, freqi, gi]

    ggrid = weights.cumsum()
    weights_conv = np.outer(weights, weights).flatten()

    for speci in numba.prange(Ns):
        for freqi in numba.prange(Nf):
            for gi in numba.prange(Ng):
                k1[:, :, freqi, gi] = kout[:,:, freqi, gi]
                k2[:, :, freqi, gi] = ktable[speci + 1,:,:, freqi, gi] * mmr[speci + 1, :, :]

        for pi in numba.prange(Np):
            for ti in numba.prange(Nt):
                for freqi in numba.prange(Nf):
                    for gi in numba.prange(Ng):
                        for gj in numba.prange(Ng):
                            kout_conv[pi,ti,freqi,gj+Ng*gi] = k1[pi,ti,freqi,gi] + k2[pi,ti,freqi,gj]

        kout = resort_rebin_njit(ggrid, kout_conv, k1, k2, weights_conv, kout, Np, Nt, Nf, Ng)

    return kout


class CombineOpac:
    def __init__(self, opac):
        self.opac = opac
        assert opac.interp_done, 'yo, dude, you need to run setup_temp_and_pres on opac first'

    def _get_mix_func(self, method):
        """Reshapes the mass mixing ratios and check that they are in the correct shape."""
        if method == 'linear':
            return partial(self._add_linear, self.opac.kcoeff)
        elif method == 'AEE':
            return partial(self._add_aee, self.opac.kcoeff)
        elif method == 'RORR':
            return partial(self._add_rorr, self.opac.kcoeff, self.opac.weights)
        else:
            raise NotImplementedError('Method not implemented.')

    def _check_mmr_shape(self, mmr):
        """Reshapes the mass mixing ratios and check that they are in the correct shape."""
        if isinstance(mmr, dict):
            mmr = np.array([mmr[speci] for speci in self.opac.spec])
        assert mmr.shape == (self.opac.ls, self.opac.lp[0], self.opac.lt[0]), 'shape of mmr needs to be species, pressure, temperature'
        return mmr

    def add_single(self, mmr, method=DEFAULT_METHOD):
        """mix one kgrid"""
        mmr = self._check_mmr_shape(mmr)
        mix_func = self._get_mix_func(method)
        return mix_func(mmr)

    def add_batch(self, mmr, method=DEFAULT_METHOD):
        """mix multiple kgrids"""
        mmr = [self._check_mmr_shape(mmr_i) for mmr_i in mmr]
        mix_func = self._get_mix_func(method)
        return np.asarray([mix_func(mmr_i) for mmr_i in mmr])

    def add_batch_parallel(self, mmr, method=DEFAULT_METHOD, **pool_kwargs):
        """Parallel version of add_batch"""
        mmr = [self._check_mmr_shape(mmr_i) for mmr_i in mmr]
        mix_func = self._get_mix_func(method)
        with Pool(**pool_kwargs) as pool:
            return np.asarray(pool.map(mix_func, mmr))

    @staticmethod
    def _add_linear(ktable, mmr):
        """linear additive mixing of a kgrid."""
        return np.sum(ktable*mmr[:,:,:,np.newaxis,np.newaxis], axis=0)

    @staticmethod
    def _add_aee(ktable, mmr):
        """Adaptive equivalent extinction on a kgrid."""
        weighted_k = ktable*mmr[:,:,:,np.newaxis,np.newaxis]
        gray_k = np.sum(weighted_k,axis=-1)
        index_max = np.argsort(gray_k, axis=0)
        major = np.take_along_axis(weighted_k, index_max[..., np.newaxis], 0)[-1]
        rest = np.sum(np.take_along_axis(gray_k, index_max, 0)[:-1], axis=0)
        return major + rest[..., np.newaxis]

    @staticmethod
    def _add_rorr(ktable, weights, mmr):
        """Add up ktables by random overlap with resorting and rebinning."""
        kout = mmr[0,:,:,np.newaxis,np.newaxis]*ktable[0,:,:,:,:]
        ggrid = weights.cumsum()
        weights_conv = np.outer(weights, weights).flatten()

        for speci in range(ktable.shape[0]-1):
            k1 = kout
            k2 = ktable[speci + 1] * mmr[speci + 1, :, :, np.newaxis, np.newaxis]
            kout_conv = (k1[...,:,None] + k2[...,None,:]).reshape(*kout.shape[:-1], weights_conv.shape[0])
            kout = resort_rebin_njit(ggrid, kout_conv, k1, k2, weights_conv, kout, *kout.shape)

        return kout