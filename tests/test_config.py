import pytest
import glob
import os


@pytest.fixture(scope='module')
def opac_files():
    """Setup files"""
    resolution ='10'
    base = '/Users/schneider/codes/exo/prt_input_data/opacities/lines/corr_k'
    files = glob.glob(os.path.join(base, f'*_R_{resolution}/*.h5'))
    setup = {
        'R': resolution,
        'files': files
    }
    expected = {
        'spec': ('CH4', 'H2S', 'CO2', 'Na', 'K', 'SiO', 'FeH', 'HCN', 'VO', 'H2O', 'TiO', 'PH3', 'CO', 'NH3'),
        'T_interp_test': 901,
        'p_interp_test': 0.47
    }
    return setup, expected
