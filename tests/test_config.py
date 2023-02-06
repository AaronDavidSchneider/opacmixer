import pytest
import glob
import os
import numpy as np
from opac_mixer.read import ReadOpacChubb
from opac_mixer.mix import CombineOpac

@pytest.fixture(scope='module')
def opac_files():
    """Setup files"""
    resolution ='S1'
    base = '/Users/schneider/codes/exo/prt_input_data/opacities/lines/corr_k'
    files = glob.glob(os.path.join(base, f'*_R_{resolution}/*.h5'))

    expected = {
        'spec': ('CH4', 'H2S', 'CO2', 'Na', 'K', 'SiO', 'FeH', 'HCN', 'VO', 'H2O', 'TiO', 'PH3', 'CO', 'NH3'),
        'T_interp_test': 901,
        'p_interp_test': 0.47,
        'R': resolution,
        'files': files
    }
    return expected


@pytest.fixture(scope='module')
def setup_reader(opac_files):
    """Standard reader, currently opening up exomolOP."""
    expected = opac_files
    chubb = ReadOpacChubb(expected['files'])
    chubb.read_opac()
    return chubb, expected


@pytest.fixture(scope='module')
def setup_interp_reader(opac_files):
    """Standard reader, currently opening up exomolOP + interpolation."""
    expected = opac_files
    opac = ReadOpacChubb(expected['files'])
    opac.read_opac()
    opac.setup_temp_and_pres()
    return opac, expected

@pytest.fixture(scope='module')
def setup_test_mix(setup_interp_reader):
    opac, expected = setup_interp_reader
    mmr = 1e-4 * np.ones((opac.ls, opac.lp[0], opac.lt[0]))
    mixer = CombineOpac(opac)
    mix = mixer.add_single(mmr=mmr, method='RORR')
    return opac, expected, mmr, mix, mixer
