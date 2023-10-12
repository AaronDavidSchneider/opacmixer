import pytest
import glob
import os
import numpy as np
from opac_mixer.read import ReadOpacChubb
from opac_mixer.mix import CombineOpacIndividual, CombineOpacGrid


@pytest.fixture(scope="module")
def opac_files():
    """Setup files"""
    resolution = "S1"
    base = f'{os.environ["pRT_input_data_path"]}/opacities/lines/corr_k'
    files = glob.glob(os.path.join(base, f"*_R_{resolution}/*.h5"))

    expected = {
        "spec": (
            "CH4",
            "H2S",
            "CO2",
            "Na",
            "K",
            "SiO",
            "FeH",
            "HCN",
            "VO",
            "H2O",
            "TiO",
            "PH3",
            "CO",
            "NH3",
        ),
        "T_interp_test": 901,
        "p_interp_test": 0.47,
        "R": resolution,
        "files": files,
    }
    return expected


@pytest.fixture(scope="module")
def setup_reader(opac_files):
    """Standard reader, currently opening up exomolOP."""
    expected = opac_files
    chubb = ReadOpacChubb(expected["files"])
    chubb.read_opac()
    return chubb, expected


@pytest.fixture(scope="module")
def setup_interp_reader(opac_files):
    """Standard reader, currently opening up exomolOP + interpolation."""
    expected = opac_files
    opac = ReadOpacChubb(expected["files"])
    opac.read_opac()
    opac.setup_temp_and_pres()
    return opac, expected


@pytest.fixture(scope="module")
def setup_test_mix_grid(setup_interp_reader):
    opac, expected = setup_interp_reader
    mmr = 1e-4 * np.ones((opac.ls, opac.lp[0], opac.lt[0]))
    mixer = CombineOpacGrid(opac)
    mix = mixer.add_single(mmr, "RORR")
    return opac, expected, mmr, mix, mixer


@pytest.fixture(scope="module")
def setup_test_mix_ind(setup_interp_reader):
    opac, expected = setup_interp_reader

    mmr_batch = 1e-4 * np.ones((opac.ls, opac.lp[0], opac.lt[0]))
    pres = (
        np.ones((1, opac.lp[0], opac.lt[0]))
        * opac.pr[np.newaxis, :, np.newaxis]
    )
    temp = (
        np.ones((1, opac.lp[0], opac.lt[0]))
        * opac.Tr[np.newaxis, np.newaxis, :]
    )
    input_data = np.concatenate((mmr_batch, pres, temp), axis=0).T.reshape(
        (opac.lt[0] * opac.lp[0], opac.ls + 2)
    )

    mixer = CombineOpacIndividual(opac)
    mix = mixer.add_batch(input_data=input_data, method="RORR")
    return opac, expected, input_data, mix, mixer
