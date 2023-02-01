import numpy as np


class CombineOpac:
    def __init__(self, opac):
        self.opac = opac
        assert len(opac.T.shape) > 1, 'yo, dude, you need to run setup_temp_and_pres on opac first'
        assert len(opac.p.shape) > 1, 'yo, dude, you need to run setup_temp_and_pres on opac first'

    def add(self, mmr, method='RORR'):
        if isinstance(mmr, dict):
            mmr = np.array([mmr[speci] for speci in self.opac.spec])
        
        assert mmr.shape == (self.opac.ns, self.opac.lp, self.opac.lt), 'shape of mmr needs to be species, pressure, temperature'

        if method == 'linear':
            return self._add_linear(self.opac.kcoeff, mmr)
        elif method == 'AEE':
            return self._add_aee(self.opac.kcoeff, mmr)
        elif method == 'RORR':
            return self._add_rorr(self.opac.kcoeff, self.opac.weights.cumsum(), mmr)
        else:
            raise NotImplementedError('Method not implemented.')
    

    @staticmethod
    def _add_linear(ktable, mmr):
        return np.sum(ktable*mmr[:,:,:,np.newaxis,np.newaxis], axis=0)

    @staticmethod
    def _add_aee(ktable, mmr):
        raise NotImplementedError('Method not implemented.')

    @staticmethod
    def _add_rorr(ktable, g, mmr):
        kout = mmr[0,:,:,np.newaxis,np.newaxis]*ktable[0]
        g_conv = np.convolve(g,g)
        sort_index = np.argsort(g_conv)
        g_conv_sorted = g_conv[sort_index]

        interp_j = np.searchsorted(g_conv_sorted, g) - 1
        interp_d = (g - g_conv_sorted[interp_j]) / (g_conv_sorted[interp_j + 1] - g_conv_sorted[interp_j])

        for speci in range(len(ktable.shape[0])-1):
            kout_conv = np.convolve(kout[speci], ktable[speci+1]*mmr[speci+1,:,:,np.newaxis,np.newaxis])[sort_index]
            kout = (1 - interp_d) * kout_conv[:,:,:,interp_j] + kout_conv[:,:,:,interp_j] * interp_d
        
        return kout