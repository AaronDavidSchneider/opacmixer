import time

import numpy as np
from petitRADTRANS import Radtrans
from petitRADTRANS import fort_input as fi
from petitRADTRANS import fort_spec as fs
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import pyth_input as pyi

from opac_mixer import fort_bol_flux as fs_bol
from opac_mixer.utils.scalings import (
    default_input_scaling as t_x,
    default_inv_output_scaling as ti_y,
)


def simple_deep_set(kappas, weights):
    """Numpy DeepSet implementation"""
    rep = np.tensordot(kappas, weights[0], axes=(1, 0))  # first dense
    rep[rep < 0.0] = 0.0
    sum_rep = np.sum(rep, axis=1)  # sum
    dec = np.tensordot(sum_rep, weights[1], axes=(-1, 0))  # second dense
    return dec


def mix_kappa_deep_set(
    kappas, weights, input_scaling, inverse_output_scaling, *args
):
    """
    Kappa mixing for a DeepSet
    """
    shape_pred = (
        kappas.shape[1] * kappas.shape[3],
        kappas.shape[0],
        kappas.shape[2],
    )
    shape = kappas.shape[0], kappas.shape[1], kappas.shape[-1]
    kappas_resh = kappas.transpose(1, 3, 0, 2).reshape(shape_pred)
    kout = inverse_output_scaling(
        kappas_resh, simple_deep_set(input_scaling(kappas_resh), weights)
    ).T.reshape(shape)
    return kout


def mix_kappa_add(kappas, *args):
    """Simple additive mixing"""
    return np.sum(kappas, axis=2)


def mix_kappa_aee(kappas, w, g, p):
    """Adaptive equivilant extinction"""
    kappa_av = np.sum(
        kappas[:, :, :, :] * w[:, None, None, None], axis=0
    ) / np.sum(w, axis=0)
    tau_tot = (
        np.cumsum(np.sum(kappa_av[:, :, :-1], axis=1) * np.diff(p), axis=-1)
        / g
    )
    kappas_thin = np.where(tau_tot[:, None, :] < 1, kappa_av[:, :, :-1], 0.0)
    max_abs = np.argmax(np.sum(kappas_thin * np.diff(p), axis=-1), -1)
    max_abs_br = (
        np.ones_like(kappas, dtype="int") * max_abs[None, :, None, None]
    )
    kmax = np.take_along_axis(kappas, max_abs_br, 2)
    # np.testing.assert_allclose(np.sum(kmax,axis=2), kappas.shape[2]*kmax[:,:,0,:])
    kappa_av_max_abs = np.take_along_axis(kappa_av, max_abs_br[0, :, :, :], 1)
    # np.testing.assert_allclose(np.sum(kappa_av_max_abs,axis=1), kappas.shape[2]*kappa_av_max_abs[:,0,:])
    kminor_sum = np.sum(kappa_av, axis=-2) - kappa_av_max_abs[:, 0, :]
    return kmax[:, :, 0, :] + kminor_sum[None, :, :]


# @numba.njit(nogil=True, fastmath=True, cache=True)
def mix_kappa_aee_jit(kappas, w, g, p, lg, lf, ls, lp):
    kappa_av = np.zeros((lf, ls, lp))
    tau = np.zeros(ls)

    kmax = np.empty((lg, lp))
    kgray = np.zeros((lp))

    kout = np.empty((lg, lf, lp))

    w_sum = np.sum(w)
    for gi in range(lg):
        kappa_av[:, :, :] = (
            kappa_av[:, :, :] + w[gi] * kappas[gi, :, :, :] / w_sum
        )

    for fi in range(lf):
        tau_tot = 0.0
        tau[:] = 0.0
        for ki in range(lp - 1):
            for si in range(ls):
                tau[si] = (
                    tau[si] + kappa_av[fi, si, ki] * (p[ki + 1] - p[ki]) / g
                )
                tau_tot = tau_tot + tau[si]

            if tau_tot > 1:
                break

        max_idx = np.argmax(tau)
        kmax[:, :] = kappas[:, fi, max_idx, :]

        kgray[:] = 0.0
        for si in range(ls):
            if si != max_idx:
                kgray[:] = kgray[:] + kappa_av[fi, si, :]

        for ki in range(lp):
            kout[:, fi, ki] = kmax[:, ki] + kgray[ki]

    return kout


class PatchedRadtrans(Radtrans):
    def __init__(self, test_ck_shuffle_comp=True, **kwargs):
        self.test_ck_shuffle_comp = test_ck_shuffle_comp
        super().__init__(test_ck_shuffle_comp=test_ck_shuffle_comp, **kwargs)

    def setup_mixing(
        self,
        mixmethod="rorr",
        weights=None,
        input_scaling=None,
        inverse_output_scaling=None,
    ):
        """
        options: 'deepset', 'aee', 'add'
        Setup weights for deepset
        """
        self._mixmethod = mixmethod
        self._ml_weights = weights

        if mixmethod == "deepset":
            if input_scaling is None:
                input_scaling = t_x
                print(
                    "Warning: Default scaling is used for the input, change if"
                    " needed"
                )
            if inverse_output_scaling is None:
                inverse_output_scaling = ti_y
                print(
                    "Warning: Default scaling is used for the output, change"
                    " if needed"
                )
            if weights is None:
                raise ValueError("we need some weights")

        self._ml_input_scaling = input_scaling
        self._ml_inverse_output_scaling = inverse_output_scaling

    def mix_kappa(self, kappas, g):
        if self._mixmethod == "deepset":
            if self._ml_weights is None:
                raise ValueError("setup weights using _set_weights first")
            return mix_kappa_deep_set(
                kappas,
                self._ml_weights,
                self._ml_input_scaling,
                self._ml_inverse_output_scaling,
            )
        elif self._mixmethod == "add":
            return mix_kappa_add(kappas)
        elif self._mixmethod == "aee":
            return mix_kappa_aee(kappas, self.w_gauss, g, self.press)
        elif self._mixmethod == "aee_jit":
            return mix_kappa_aee_jit(
                kappas, self.w_gauss, g, self.press, *kappas.shape
            )
        elif self._mixmethod == "rorr":
            return fs.combine_opas_ck(kappas, self.g_gauss, self.w_gauss)
        else:
            raise NotImplementedError("mixmethod not implemented!")

    def mix_opa_tot(
        self,
        abundances,
        mmw,
        gravity,
        sigma_lnorm=None,
        fsed=None,
        Kzz=None,
        radius=None,
        add_cloud_scat_as_abs=None,
        dist="lognormal",
        a_hans=None,
        b_hans=None,
        give_absorption_opacity=None,
        give_scattering_opacity=None,
    ):
        t = time.time()
        # Combine total line opacities,
        # according to mass fractions (abundances),
        # also add continuum opacities, i.e. clouds, CIA...
        self.mmw = mmw
        self.scat = False

        for i_spec in range(len(self.line_species)):
            self.line_abundances[:, i_spec] = abundances[
                self.line_species[i_spec]
            ]

        self.continuum_opa = np.zeros_like(self.continuum_opa)
        self.continuum_opa_scat = np.zeros_like(self.continuum_opa_scat)
        self.continuum_opa_scat_emis = np.zeros_like(
            self.continuum_opa_scat_emis
        )

        # Calc. CIA opacity
        for key in self.CIA_species.keys():
            abund = 1

            for m in self.CIA_species[key]["molecules"]:
                abund = abund * abundances[m]

            self.continuum_opa = self.continuum_opa + self.interpolate_cia(
                key, np.sqrt(abund)
            )

        # Calc. H- opacity
        if self.Hminus:
            self.continuum_opa = self.continuum_opa + pyi.hminus_opacity(
                self.lambda_angstroem,
                self.border_lambda_angstroem,
                self.temp,
                self.press,
                mmw,
                abundances,
            )

        # Add mock gray cloud opacity here
        if self.gray_opacity is not None:
            self.continuum_opa = self.continuum_opa + self.gray_opacity

        # Add cloud opacity here, will modify self.continuum_opa
        if self._check_cloud_effect(
            abundances
        ):  # add cloud opacity only if there is actually clouds
            self.scat = True
            self.calc_cloud_opacity(
                abundances,
                mmw,
                gravity,
                sigma_lnorm,
                fsed,
                Kzz,
                radius,
                add_cloud_scat_as_abs,
                dist=dist,
                a_hans=a_hans,
                b_hans=b_hans,
            )

        # Calculate rayleigh scattering opacities
        if len(self.rayleigh_species) != 0:
            self.scat = True
            self.add_rayleigh(abundances)
        # Add gray cloud deck
        if self.Pcloud is not None:
            self.continuum_opa[:, self.press > self.Pcloud * 1e6] += 1e99
        # Add power law opacity
        if self.kappa_zero is not None:
            self.scat = True
            wlen_micron = nc.c / self.freq / 1e-4
            scattering_add = (
                self.kappa_zero * (wlen_micron / 0.35) ** self.gamma_scat
            )
            add_term = np.repeat(
                scattering_add[None], int(len(self.press)), axis=0
            ).transpose()
            self.continuum_opa_scat += add_term

            if self.do_scat_emis:
                self.continuum_opa_scat_emis += add_term

        # Check if hack_cloud_photospheric_tau is used with
        # a single cloud model. Combining cloud opacities
        # from different models is currently not supported
        # with the hack_cloud_photospheric_tau parameter
        if (
            len(self.cloud_species) > 0
            and self.hack_cloud_photospheric_tau is not None
        ):
            if (
                give_absorption_opacity is not None
                or give_scattering_opacity is not None
            ):
                raise ValueError(
                    "The hack_cloud_photospheric_tau can only be "
                    "used in combination with a single cloud model. "
                    "Either use a physical cloud model by choosing "
                    "cloud_species or use parametrized cloud "
                    "opacities with the give_absorption_opacity "
                    "and give_scattering_opacity parameters."
                )

        # Add optional absorption opacity from outside
        if give_absorption_opacity is None:
            if self.hack_cloud_photospheric_tau is not None:
                if not hasattr(self, "hack_cloud_total_abs"):
                    opa_shape = (self.freq.shape[0], self.press.shape[0])
                    self.hack_cloud_total_abs = np.zeros(opa_shape)

        else:
            cloud_abs = give_absorption_opacity(
                nc.c / self.freq / 1e-4, self.press * 1e-6
            )
            self.continuum_opa += cloud_abs

            if self.hack_cloud_photospheric_tau is not None:
                # This assumes a single cloud model that is
                # given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                self.hack_cloud_total_abs = cloud_abs

        # Add optional scatting opacity from outside
        if give_scattering_opacity is None:
            if self.hack_cloud_photospheric_tau is not None:
                if not hasattr(self, "hack_cloud_total_scat_aniso"):
                    opa_shape = (self.freq.shape[0], self.press.shape[0])
                    self.hack_cloud_total_scat_aniso = np.zeros(opa_shape)

        else:
            cloud_scat = give_scattering_opacity(
                nc.c / self.freq / 1e-4, self.press * 1e-6
            )
            self.continuum_opa_scat += cloud_scat

            if self.do_scat_emis:
                self.continuum_opa_scat_emis += cloud_scat

            if self.hack_cloud_photospheric_tau is not None:
                # This assumes a single cloud model that is
                # given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                self.hack_cloud_total_scat_aniso = cloud_scat

        # Interpolate line opacities, combine with continuum oacities
        self.line_struc_kappas = fi.mix_opas_ck(
            self.line_abundances, self.line_struc_kappas, self.continuum_opa
        )

        # Similar to the line-by-line case below, if test_ck_shuffle_comp is
        # True, we will put the total opacity into the first species slot and
        # then carry the remaining radiative transfer steps only over that 0
        # index.
        if (self.mode == "c-k") and self.test_ck_shuffle_comp:
            self.line_struc_kappas[:, :, 0, :] = (
                self.line_struc_kappas[:, :, 0, :] - self.continuum_opa
            )
            self.line_struc_kappas[:, :, 0, :] = self.mix_kappa(
                self.line_struc_kappas, gravity
            )
            self.line_struc_kappas[:, :, 0, :] = (
                self.line_struc_kappas[:, :, 0, :] + self.continuum_opa
            )

        # In the line-by-line case we can simply
        # add the opacities of different species
        # in frequency space. All opacities are
        # stored in the first species index slot
        if (self.mode == "lbl") and (int(len(self.line_species)) > 1):
            self.line_struc_kappas[:, :, 0, :] = np.sum(
                self.line_struc_kappas, axis=2
            )

        self.time_opa = time.time() - t

    def calc_bolometric_flux(
        self,
        temp,
        abunds,
        gravity,
        mmw,
        sigma_lnorm=None,
        fsed=None,
        Kzz=None,
        radius=None,
        gray_opacity=None,
        Pcloud=None,
        kappa_zero=None,
        gamma_scat=None,
        add_cloud_scat_as_abs=None,
        Tstar=None,
        Rstar=None,
        semimajoraxis=None,
        geometry="dayside_ave",
        theta_star=0,
    ):
        """Method to calculate the bolometric flux for each atmospheric layer

        Args:
            temp:
                the atmospheric temperature in K, at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            abunds:
                if not use_grid: dictionary of mass fractions for all atmospheric absorbers.
                Dictionary keys are the species names.
                Every mass fraction array
                has same length as pressure array.
                If use_grid: List of dictionaries: Each dict must be a
                dictionary of mass fractions for all atmospheric absorbers.
                Dictionary keys are the species names.
                Every mass fraction array
                has same length as pressure array.
            gravity: float
                Surface gravity in cgs. Vertically constant for emission
                spectra.
            mmw:
                the atmospheric mean molecular weight in amu,
                at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            sigma_lnorm: Optional[float]
                width of the log-normal cloud particle size distribution
            fsed: Optional[float]
                cloud settling parameter
            Kzz: Optional
                the atmospheric eddy diffusion coeffiecient in cgs untis
                (i.e. :math:`\\rm cm^2/s`),
                at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            radius: Optional
                dictionary of mean particle radii for all cloud species.
                Dictionary keys are the cloud species names.
                Every radius array has same length as pressure array.
            gray_opacity: Optional[float]
                Gray opacity value, to be added to the opacity at all
                pressures and wavelengths (units :math:`\\rm cm^2/g`)
            Pcloud: Optional[float]
                Pressure, in bar, where opaque cloud deck is added to the
                absorption opacity.
            kappa_zero: Optional[float]
                Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
            gamma_scat: Optional[float]
                Has to be given if kappa_zero is definded, this is the
                wavelength powerlaw index of the parametrized scattering
                opacity.
            add_cloud_scat_as_abs: Optional[bool]
                If ``True``, 20 % of the cloud scattering opacity will be
                added to the absorption opacity, introduced to test for the
                effect of neglecting scattering.
            Tstar: Optional[float]
                The temperature of the host star in K, used only if the
                scattering is considered. If not specified, the direct
                light contribution is not calculated.
            Rstar: Optional[float]
                The radius of the star in Solar radii. If specified,
                used to scale the to scale the stellar flux,
                otherwise it uses PHOENIX radius.
            semimajoraxis: Optional[float]
                The distance of the planet from the star. Used to scale
                the stellar flux when the scattering of the direct light
                is considered.
            geometry: Optional[string]
                if equal to ``'dayside_ave'``: use the dayside average
                geometry. if equal to ``'planetary_ave'``: use the
                planetary average geometry. if equal to
                ``'non-isotropic'``: use the non-isotropic
                geometry.
            theta_star: Optional[float]
                Inclination angle of the direct light with respect to
                the normal to the atmosphere. Used only in the
                non-isotropic geometry scenario.
            stellar_intensity:
                Alternatively set the stellar intensity directly
            flux_scaling: "BB", value or just anything
                Scale the stellar flux. If BB, it will scale to the Blackbody of Tstar, if value, it will use the value for scaling.
                Else it will not do any scaling
        """
        self.hack_cloud_photospheric_tau = None
        self.Pcloud = Pcloud
        self.kappa_zero = kappa_zero
        self.gamma_scat = gamma_scat
        self.gray_opacity = gray_opacity
        self.geometry = geometry

        self.interpolate_species_opa(temp)

        t = time.time()
        self.mix_opa_tot(
            abundances=abunds,
            mmw=mmw,
            gravity=gravity,
            sigma_lnorm=sigma_lnorm,
            fsed=fsed,
            Kzz=Kzz,
            radius=radius,
            add_cloud_scat_as_abs=add_cloud_scat_as_abs,
        )
        self.time_opa = time.time() - t

        self.mu_star = np.cos(theta_star * np.pi / 180.0)
        if self.mu_star <= 0.0:
            self.mu_star = 1e-8
            self.stellar_intensity = np.zeros_like(self.freq)
        else:
            self.get_star_spectrum(Tstar, semimajoraxis, Rstar)

        self.calc_opt_depth(gravity)

        # self.continuum_opa_scat_emis = np.zeros_like(self.continuum_opa_scat_emis)
        self.calc_RT_bolometric()

    def calc_RT_bolometric(self):
        # Calculate the bolometric flux
        if not self.do_scat_emis:
            self.photon_destruction_prob = np.ones_like(
                self.total_tau[:, :, 0, :]
            )

        # Only use 0 index for species because for lbl or test_ck_shuffle_comp = True
        # everything has been moved into the 0th index
        self.flux_bol, self.flux_star = fs_bol.feautrier_rad_trans_bolometric(
            self.border_freqs,
            self.total_tau[:, :, 0, :],
            self.temp,
            self.mu,
            self.w_gauss_mu,
            self.w_gauss,
            self.photon_destruction_prob,
            self.reflectance,
            self.emissivity,
            self.stellar_intensity,
            self.geometry,
            self.mu_star,
        )
