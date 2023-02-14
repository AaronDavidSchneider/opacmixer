import numba
import numpy as np
import tqdm
from .utils import interp_2d
from functools import partial
from multiprocessing.pool import Pool


DEFAULT_METHOD = 'RORR'

@numba.njit(nogil=True, fastmath=True, cache=True)
def resort_rebin_njit(kout_conv, k1, k2, weights_in, weights_conv, Np, Nt, Nf, Ng):
    """Resort and rebin the convoluted kappas. Note that this function works with g values calculated half integer."""

    # Initialize arrays
    kout = np.zeros((Np, Nt, Nf, Ng), dtype=np.float64)
    len_resort = Ng*Ng
    kout_conv_resorted = np.zeros(len_resort+1, dtype=np.float64)  # note: We add +1 for the right edge
    g_resorted = np.zeros(len_resort+1, dtype=np.float64)  # note: We add +1 for the right edge
    ggrid = compute_ggrid(weights_in, Ng)

    # Start looping over p, t and freq, because we need to do the resorting and rebinning individually

    for pi in range(Np):
        for ti in range(Nt):
            for freqi in range(Nf):
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


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=False)
def _rorr_single(ktable, weights, weights_conv, ls, lf, lg, temp_old, press_old, lt_old, lp_old, input_data):
    kout = np.empty((1, 1, lf, lg), dtype=np.float64)
    kout_conv = np.empty((1, 1, lf, lg*lg), dtype=np.float64)
    mixed_ktables = np.empty((ls, 1, 1, lf, lg), dtype=np.float64)

    temp = np.asarray([input_data[-1]])
    press = np.asarray([input_data[-2]])
    mmr = np.asarray(input_data[:-2])

    ki = interp_2d(temp_old, press_old, temp, press, ktable, ls, lf, lg, lt_old, lp_old, 1, 1)

    for speci in range(ls):
        mixed_ktables[speci, 0, 0, :, :] = mmr[speci]*ki[speci, 0, 0, :, :]

    kout[:, :, :, :] = mixed_ktables[0, :, :, :, :]

    for speci in range(1, ls):
        k1 = kout
        k2 = mixed_ktables[speci, :, :, :, :]
        for gi in range(lg):
            for gj in range(lg):
                kout_conv[0, 0, :, gi+lg*gj] = k1[0, 0, :, gj] + k2[0, 0, :, gi]

        kout = resort_rebin_njit(kout_conv, k1, k2, weights, weights_conv, 1, 1, lf, lg)

    return kout[0, 0, :, :]


class CombineOpac:
    def __init__(self, opac):
        self.opac = opac
        assert self.opac.interp_done, 'yo, dude, you need to run setup_temp_and_pres on opac first'


class CombineOpacIndividual(CombineOpac):
    def add_batch(self, input_data, method=DEFAULT_METHOD):
        """mix one kgrid"""
        input_data = self._check_input_shape(input_data)
        mix_func = self._get_mix_func(method, use_mult=False)
        return mix_func(input_data)

    def add_batch_parallel(self, input_data, method=DEFAULT_METHOD):
        """mix one kgrid"""
        input_data = self._check_input_shape(input_data)
        mix_func = self._get_mix_func(method, use_mult=True)
        return mix_func(input_data)

    def _get_mix_func(self, method, use_mult):
        """Reshapes the mass mixing ratios and check that they are in the correct shape."""
        if method == 'RORR':
            return partial(self._add_rorr, self.opac.kcoeff, self.opac.weights, self.opac.Tr, self.opac.pr, use_mult)
        else:
            raise NotImplementedError('Method not implemented.')

    def _check_input_shape(self, input_data):
        """Checks that they are in the correct shape."""
        assert len(input_data.shape) == 2
        assert input_data.shape[1] == self.opac.ls + 2
        return input_data

    @staticmethod
    def _add_rorr(ktable, weights, temp_old, press_old, use_mult, input_data):
        """Add up ktables by random overlap with resorting and rebinning."""
        Nsamples = input_data.shape[0]
        ls = input_data.shape[1] - 2
        lp_old = np.ones(ls, np.int8)*len(press_old)
        lt_old = np.ones(ls, np.int8)*len(temp_old)
        temp_old = np.ones((ls, lt_old[0]))*temp_old[np.newaxis,:]
        press_old = np.ones((ls, lp_old[0]))*press_old[np.newaxis,:]
        lf = ktable.shape[-2]
        lg = ktable.shape[-1]

        assert ls == ktable.shape[0]
        assert lp_old[0] == ktable.shape[1]
        assert lt_old[0] == ktable.shape[2]

        weights_conv = np.outer(weights,weights).flatten()

        func = partial(_rorr_single, ktable, weights, weights_conv, ls, lf, lg, temp_old, press_old, lt_old, lp_old)

        if use_mult:
            with Pool() as pool:
                return np.asarray(list(tqdm.tqdm(pool.imap(func, input_data, chunksize=100), total=Nsamples)), dtype=np.float64)
        else:
            return np.asarray(list(tqdm.tqdm(map(func, input_data), total=Nsamples)), dtype=np.float64)


class CombineOpacGrid(CombineOpac):
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
        return np.asarray([mix_func(mmr_i) for mmr_i in tqdm.tqdm(mmr)])

    def add_batch_parallel(self, mmr, method=DEFAULT_METHOD, **pool_kwargs):
        """Parallel version of add_batch"""
        mmr = [self._check_mmr_shape(mmr_i) for mmr_i in mmr]
        mix_func = self._get_mix_func(method)
        with Pool(**pool_kwargs) as pool:
            return np.asarray(list(tqdm.tqdm(pool.imap(mix_func, mmr), total=len(mmr))), dtype=np.float64)

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