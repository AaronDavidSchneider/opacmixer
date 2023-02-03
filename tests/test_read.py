import pytest
from test_config import setup_reader, setup_interp_reader, opac_files
import copy
import numpy as np


def test_dims(setup_reader):
    """Test that dimensions all match each other."""
    opac, expected = setup_reader

    assert opac.p.shape == (opac.ls, opac.lp.max())
    assert opac.T.shape == (opac.ls, opac.lt.max())
    assert opac.bin_edges.shape == (opac.lf[0]+1,)
    assert opac.bin_center.shape  == (opac.lf[0],)
    assert opac.weights.shape == (opac.lg[0],)

    assert opac.ls == len(expected['spec'])
    assert all(len(l) == opac.ls for l in [opac.lp, opac.lg, opac.lf, opac.lt])

    assert opac.kcoeff.shape == (opac.ls, opac.lp.max(), opac.lt.max(), opac.lf[0], opac.lg[0])


def test_dims_after_interp(setup_interp_reader):
    """Test that dimensions all match each other, also after interpolation."""
    opac, expected = setup_interp_reader

    assert opac.pr.shape == (opac.lp[0],)
    assert opac.Tr.shape == (opac.lt[0],)
    assert opac.p.shape == (opac.ls, opac.lp[0])
    assert opac.T.shape == (opac.ls, opac.lt[0])

    assert opac.kcoeff.shape == (opac.ls, opac.lp[0], opac.lt[0], opac.lf[0], opac.lg[0])
    assert len(set(opac.lp)) == 1
    assert len(set(opac.lt)) == 1


def test_interp_edges(setup_reader):
    """Test that edges are correctly matched, when interpolating."""
    opac, expected = setup_reader

    opac_temp = copy.deepcopy(opac)
    opac_temp.setup_temp_and_pres(pres=[1e-20], temp=[1e-20])
    assert np.all(np.isclose(opac_temp.kcoeff[:,0,0,:,:], opac.kcoeff[:,0,0,:,:]))

    opac_temp = copy.deepcopy(opac)
    opac_temp.setup_temp_and_pres(pres=[1e20], temp=[1e-20])
    assert np.all(np.isclose(opac_temp.kcoeff[:,0,0,:,:], opac.kcoeff[:,-1,0,:,:]))

    opac_temp = copy.deepcopy(opac)
    opac_temp.setup_temp_and_pres(pres=[1e-20], temp=[1e20])
    assert np.all(np.isclose(opac_temp.kcoeff[:,0,0,:,:], opac.kcoeff[:,0,-1,:,:]))

    opac_temp = copy.deepcopy(opac)
    opac_temp.setup_temp_and_pres(pres=[1e20], temp=[1e20])
    assert np.all(np.isclose(opac_temp.kcoeff[:,0,0,:,:], opac.kcoeff[:,-1,-1,:,:]))


def test_interp(setup_interp_reader, setup_reader):
    """Test general interpolation."""
    opaci, _ = setup_interp_reader
    opac, expected = setup_interp_reader

    # test:
    T = expected['T_interp_test']
    pres = expected['p_interp_test']

    for speci in range(opac.ls):
        data = []
        for o in [opac, opaci]:
            pi = np.searchsorted(o.p[speci], pres) - 1
            ti = np.searchsorted(o.T[speci], T) - 1
            data.append(o.kcoeff[speci,pi,ti,:,:])
        assert np.all(np.isclose(data[0],data[1]))