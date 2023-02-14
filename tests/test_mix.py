import numpy as np

from test_config import setup_reader, setup_interp_reader, opac_files, setup_test_mix_grid, setup_test_mix_ind
from opac_mixer.mix import CombineOpacIndividual, CombineOpacGrid
import numba


def test_mix_single(setup_interp_reader):
    """ Test that single molecule is 'mixed' identically with all mixing functions """
    opac, expected = setup_interp_reader

    mol_single_i = 0
    mmr = np.zeros((opac.ls, opac.lp[0], opac.lt[0]))
    mmr[mol_single_i,:,:] = 1.0

    mix = CombineOpacGrid(opac)
    lm = mix.add_single(mmr=mmr, method='linear')
    am = mix.add_single(mmr=mmr, method='AEE')
    rm = mix.add_single(mmr=mmr, method='RORR')

    assert np.all(np.isclose(lm, am)), 'mixes are not the same, even though we just mix one species!'
    assert np.all(np.isclose(lm, rm)), 'mixes are not the same, even though we just mix one species!'
    assert np.all(np.isclose(lm, opac.kcoeff[mol_single_i,:,:])), 'we do not get the correct mix!'


@numba.njit(nogil=True, fastmath=True, cache=True)
def kdata_conv_loop_profile(kdata1,kdata2,kdataconv,Nlay,Nw,Ng):
    """Computes the convolution of two atmospheric kdata profiles.

    Note: copied from exo-k for testing purposes

    Nothing is returned. kdataconv is changed in place.

    Parameters
    ----------
        kdata1,kdata2 : arrays
            The two ktable.kdata tables to convolve.
        kdataconv : array
            Result table where the last dimension as a length equal to Ng^2.
        Nlay : int
            Number of atmospheric layers
        Nw : int
            Number of wavenumber points
        Ng : int
            Number of g-points
    """
    for i in range(Nlay):
        for j in range(Nw):
            for l in range(Ng):
                for m in range(Ng):
                    kdataconv[i,j,l*Ng+m]=kdata1[i,j,m]+kdata2[i,j,l]


@numba.njit(nogil=True, fastmath=True, cache=True)
def RandOverlap_2_kdata_prof(Nlay, Nw, Ng, kdata1, kdata2, weights, ggrid):
    """Function to randomely mix the opacities of 2 species in an atmospheric profile.

    Note: copied from exo-k for testing purposes

    Parameters
    ----------
        Nlay, Nw, Ng: int
            Number of layers, spectral bins, and gauss points.
        kdata1, kdata2: arrays of size (Nlay, Nw, Ng)
            vmr weighted cross-sections for the two species.
        weights: array
            gauss weights.
        ggrid: array
            g-points.

    Returns
    -------
        array
            k-coefficient array of the mix over the atmospheric column.
    """
    kdataconv = np.zeros((Nlay, Nw, Ng ** 2))
    weightsconv = np.zeros(Ng ** 2)
    newkdata = np.zeros((Nlay, Nw, Ng))

    for jj in range(Ng):
        for ii in range(Ng):
            weightsconv[ii * Ng + jj] = weights[jj] * weights[ii]

    kdata_conv_loop_profile(kdata1, kdata2, kdataconv, Nlay, Nw, Ng)

    for ilay in range(Nlay):
        for iW in range(Nw):
            if (kdata1[ilay, iW, -1] - kdata2[ilay, iW, 0]) * (kdata2[ilay, iW, -1] - kdata1[ilay, iW, 0]) < 0.:
                # ii+=1
                if kdata1[ilay, iW, -1] > kdata2[ilay, iW, -1]:
                    newkdata[ilay, iW, :] = kdata1[ilay, iW, :]
                else:
                    newkdata[ilay, iW, :] = kdata2[ilay, iW, :]
                continue
            tmp = kdataconv[ilay, iW, :]
            indices = np.argsort(tmp)
            kdatasort = tmp[indices]
            weightssort = weightsconv[indices]
            newggrid = np.cumsum(weightssort)
            # ind=np.searchsorted(newggrid,ggrid,side='left')
            # newkdata[ilay,iW,:]=kdatasort[ind]
            newkdata[ilay, iW, :] = np.interp(ggrid, newggrid, kdatasort)
    return newkdata


def test_individual_rorr_vs_grid_rorr(setup_test_mix_grid, setup_test_mix_ind):
    """Test that there is no difference between the individual mixing and the grid mixing"""
    opac_ind, _, input_data, mix_ind, _ = setup_test_mix_ind
    opac_grid, _, _, mix_grid, _ = setup_test_mix_grid

    mix_ind_resh = mix_ind.reshape((opac_grid.lt[0],opac_grid.lp[0],opac_grid.lf[0],opac_grid.lg[0])).transpose(1,0,2,3)

    assert np.isclose(mix_ind_resh, mix_grid).all()


def _test_rorr_vs_aee_vs_linear(setup_test_mix_grid):
    """Plot the differences between the simplified mixing implementations"""
    import matplotlib.pyplot as plt

    opac, expected, mmr, mix, mixer = setup_test_mix_grid

    mix_l = mixer.add_single(mmr=mmr, method='linear')
    mix_a = mixer.add_single(mmr=mmr, method='AEE')

    for p in [1e-1]:
        for t in [2000]:
            pi = np.searchsorted(opac.pr, p) - 1
            ti = np.searchsorted(opac.Tr, t) - 1
            for fi in range(opac.lf[0]):
                x = opac.bin_edges[fi] + opac.weights.cumsum() * (opac.bin_edges[fi + 1] - opac.bin_edges[fi])
                l1, = plt.gca().plot(x, mix[pi, ti, fi, :], color='black', alpha=0.4, ls='-', label='RORR')
                l2, = plt.gca().plot(x, mix_l[pi, ti, fi, :], color='red', alpha=0.4, ls='-', label='linear')
                l3, = plt.gca().plot(x, mix_a[pi, ti, fi, :], color='orange', alpha=0.4, ls='-', label='aee')

            plt.xscale('log')
            plt.yscale('log')
            plt.gca().legend(handles=[l1,l2,l3])
            plt.show()


def test_mix_vs_prt(setup_test_mix_grid):
    """Compare against petitRADTRANS ck RORR implementation"""
    from petitRADTRANS import Radtrans
    import matplotlib.pyplot as plt

    opac, expected, mmr, mix, mixer = setup_test_mix_grid

    linespecies = [f.split('/')[-1].split('.')[0] for f in expected['files']]
    atmosphere = Radtrans(line_species=linespecies, pressures=opac.pr, wlen_bords_micron=[(1e4/opac.bin_edges).min(), (1e4/opac.bin_edges).max()], test_ck_shuffle_comp=True)

    sigma_lnorm = None
    fsed = None
    Kzz = None,
    radius = None,
    add_cloud_scat_as_abs = None,
    dist = "lognormal"
    a_hans = None,
    b_hans = None,
    give_absorption_opacity = None
    give_scattering_opacity = None
    mmw = 2.35  # not used
    g = 10.0  # not used

    kcoeff_prt = np.empty_like(mix)
    for i, temp in enumerate(opac.Tr):
        abunds = {spec: mmr[speci,:,i] for speci, spec in enumerate(linespecies)}

        atmosphere.interpolate_species_opa(np.ones_like(opac.pr)*temp)

        assert np.all(np.isclose(atmosphere.line_struc_kappas[:,:,:,:], opac.kcoeff[:,:,i,::-1,:].transpose(3,2,0,1), rtol=0.01)), 'interpolation gave different result'

        atmosphere.mix_opa_tot(abunds, mmw, g, sigma_lnorm, fsed, Kzz, radius,
                         add_cloud_scat_as_abs=add_cloud_scat_as_abs,
                         dist=dist, a_hans=a_hans, b_hans=b_hans,
                         give_absorption_opacity=give_absorption_opacity,
                         give_scattering_opacity=give_scattering_opacity)

        kcoeff_prt[:,i,:,:] = atmosphere.line_struc_kappas[:,::-1,0,:].transpose(2,1,0)

    print(f'prt comparison gave {100*np.isclose(kcoeff_prt, mix, rtol=0.05).sum()/len(kcoeff_prt.flatten())} % similarity at 5% tolerance.')

    for p in [1e-1]:
        for t in [1500]:
            pi = np.searchsorted(opac.pr, p) - 1
            ti = np.searchsorted(opac.Tr, t) - 1

            if not np.all(np.isclose(kcoeff_prt[pi, ti], mix[pi, ti])):
                for fi in range(opac.lf[0]):
                    x = opac.bin_edges[fi] + opac.weights.cumsum() * (opac.bin_edges[fi + 1] - opac.bin_edges[fi])
                    plt.title(f'prt: {p}, {t}')
                    plt.loglog(x, mix[pi, ti, fi, :], color='black', alpha=0.4, ls='-')
                    plt.loglog(x, kcoeff_prt[pi, ti, fi, :], color='orange', alpha=0.4, ls='-')

                plt.show()

def _test_mix_vs_exok(setup_test_mix_grid):
    """Compare against exok ck RORR implementation"""
    import matplotlib.pyplot as plt
    opac, expected, mmr, mix, mixer = setup_test_mix_grid

    kcoeff_exok = np.empty_like(mix)
    for i, temp in enumerate(opac.Tr):
        kcoeff_exok[:, i, :, :] = mmr[0,:,i, np.newaxis, np.newaxis]*opac.kcoeff[0, :, i, :, :]
        for speci in range(1,len(opac.spec)):
            # (Nlay, Nw, Ng)
            kdata2 = mmr[speci, :, i, np.newaxis, np.newaxis] * opac.kcoeff[speci, :, i, :, :]
            kdata1 = kcoeff_exok[:, i, :, :]

            kcoeff_exok[:, i, :, :] = RandOverlap_2_kdata_prof(opac.lp[0], opac.lf[0], opac.lg[0], kdata1, kdata2, opac.weights, opac.weights.cumsum())

    for p in [1e-1]:
        for t in [1500]:
            pi = np.searchsorted(opac.pr, p) - 1
            ti = np.searchsorted(opac.Tr, t) - 1

            if not np.all(np.isclose(kcoeff_exok[pi, ti], mix[pi, ti])):
                for fi in range(opac.lf[0]):
                    x = opac.bin_edges[fi] + opac.weights.cumsum() * (opac.bin_edges[fi + 1] - opac.bin_edges[fi])
                    plt.title(f'exok: {p}, {t}')
                    plt.loglog(x, mix[pi, ti, fi, :], color='black', alpha=0.4, ls='-')
                    plt.loglog(x, kcoeff_exok[pi, ti, fi, :], color='orange', alpha=0.4, ls='-')

                plt.show()






