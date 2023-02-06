import numba
import numpy as np
import time
from functools import partial
from multiprocessing.pool import Pool

DEFAULT_METHOD = 'RORR'


@numba.njit(nogil=True, fastmath=True, cache=True)
def resort_rebin_njit(kout_conv, k1, k2, weights_in, weights_conv, Np, Nt, Nf, Ng):
    """Resort and rebin the convoluted kappas. Note that this function works with g values calculated half integer."""

    # Initialize arrays
    kout = np.zeros((Np, Nt, Nf, Ng))
    len_resort = Ng*Ng
    kout_conv_resorted = np.zeros(len_resort+1)  # note: We add +1 for the right edge
    g_resorted = np.zeros(len_resort+1)  # note: We add +1 for the right edge
    ggrid = compute_ggrid(weights_in, Ng)

    # Start looping over p, t and freq, because we need to do the resorting and rebinning individually

    for pi in numba.prange(Np):
        for ti in numba.prange(Nt):
            for freqi in numba.prange(Nf):
                # Sort and resort:
                index_sort = np.argsort(kout_conv[pi, ti, freqi])
                kout_conv_resorted[:len_resort] = kout_conv[pi, ti, freqi][index_sort]
                weights_resorted = weights_conv[index_sort]
                # compute new g-grid:
                g_resorted[:len_resort] = compute_ggrid(weights_resorted, Ng*Ng)
                # edges:
                g_resorted[len_resort] = 1.0
                kout_conv_resorted[len_resort] = k1[pi, ti, freqi, -1] + k2[pi, ti, freqi, -1]
                kout_conv_resorted[0] = k1[pi, ti, freqi, 0] + k2[pi, ti, freqi, 0]
                # interpolate:
                kout[pi, ti, freqi, :] = np.interp(ggrid, g_resorted,
                                                   kout_conv_resorted)
    return kout


@numba.njit(nogil=True, fastmath=True, cache=True)
def compute_ggrid(w, Ng):
    """Helper function that calculates the ggrid for given weights. Works on a halfinteger grid."""
    cum_sum = 0.0
    gcomp = np.empty(Ng)

    w_weighted = w / np.sum(w)
    for i in range(Ng):
        gcomp[i] = cum_sum + 0.5 * w_weighted[i]
        cum_sum = cum_sum + w_weighted[i]

    return gcomp

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
        mixed_ktables = mmr[:, :, :, np.newaxis, np.newaxis]*ktable[:, :, :, :, :]
        kout = mixed_ktables[0, :, :, :, :]
        weights_conv = np.outer(weights, weights).flatten()

        for speci in range(1, ktable.shape[0]):
            k1 = kout
            k2 = mixed_ktables[speci]
            kout_conv = (k1[..., :, np.newaxis] + k2[..., np.newaxis, :]).reshape(*kout.shape[:-1], weights_conv.shape[0])
            kout = resort_rebin_njit(kout_conv, k1, k2, weights, weights_conv, *kout.shape)

        return kout