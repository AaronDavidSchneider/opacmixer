import numpy as np

from test_config import setup_reader, setup_interp_reader, opac_files
from opac_mixer.mix import CombineOpac


def test_mix_single(setup_interp_reader):
    """ Test that single molecule is 'mixed' identically with all mixing functions """
    opac, expected = setup_interp_reader

    mol_single_i = 0
    mmr = np.zeros((opac.ls, opac.lp[0], opac.lt[0]))
    mmr[mol_single_i,:,:] = 1.0

    mix = CombineOpac(opac)
    lm = mix.add(mmr=mmr, method='linear')
    am = mix.add(mmr=mmr, method='AEE')
    rm = mix.add(mmr=mmr, method='RORR')

    assert np.all(np.isclose(lm, am)), 'mixes are not the same, even though we just mix one species!'
    assert np.all(np.isclose(lm, rm)), 'mixes are not the same, even though we just mix one species!'
    assert np.all(np.isclose(lm, opac.kcoeff[mol_single_i,:,:])), 'we do not get the correct mix!'









