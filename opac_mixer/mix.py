import numba
import numpy as np


@numba.njit(nogil=True, fastmath=True, cache=True)
def resort_rebin_njit(ggrid, kout_conv, weights_conv, kout, Np, Nt, Nf, Ng):
    for pi in numba.prange(Np):
        for ti in numba.prange(Nt):
            for freqi in numba.prange(Nf):
                index_sort = np.argsort(kout_conv[pi,ti,freqi])
                kout_conv_resorted = kout_conv[pi,ti,freqi][index_sort]
                weights_resorted = weights_conv[index_sort]
                g_resorted = weights_resorted.cumsum()
                kout[pi, ti, freqi, :] = np.interp(ggrid, g_resorted,
                                                   kout_conv_resorted)
    return kout


class CombineOpac:
    def __init__(self, opac):
        self.opac = opac
        assert opac.interp_done, 'yo, dude, you need to run setup_temp_and_pres on opac first'

    def add(self, mmr, method='RORR'):
        if isinstance(mmr, dict):
            mmr = np.array([mmr[speci] for speci in self.opac.spec])
        
        assert mmr.shape == (self.opac.ls, self.opac.lp[0], self.opac.lt[0]), 'shape of mmr needs to be species, pressure, temperature'

        if method == 'linear':
            return self._add_linear(self.opac.kcoeff, mmr)
        elif method == 'AEE':
            return self._add_aee(self.opac.kcoeff, mmr)
        elif method == 'RORR':
            return self._add_rorr(self.opac.kcoeff, self.opac.weights, mmr)
        else:
            raise NotImplementedError('Method not implemented.')

    @staticmethod
    def _add_linear(ktable, mmr):
        return np.sum(ktable*mmr[:,:,:,np.newaxis,np.newaxis], axis=0)

    @staticmethod
    def _add_aee(ktable, mmr):
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
            # t0 = time.time()
            kout_conv = (kout[...,:,None]+
                         (ktable[speci + 1] * mmr[speci + 1, :, :, np.newaxis, np.newaxis])[...,None,:])\
                .reshape(*kout.shape[:-1], weights_conv.shape[0])

            # t1 = time.time()
            kout = resort_rebin_njit(ggrid, kout_conv, weights_conv, kout, *kout.shape)
            # print('time (total, interp, relinterp):',time.time()-t0,time.time()-t1, (time.time()-t1)/(time.time()-t0))

        return kout
