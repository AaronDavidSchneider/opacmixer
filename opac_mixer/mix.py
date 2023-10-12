"""Housing the mixing methods.

There are two mixers: CombineOpacIndividual and CombineOpacGrid

CombineOpacIndividual:
-- takes arbitrary abundances and temperatures, pressures for each species
-- slow

CombineOpacGrid:
-- takes arbitrary abundances but keeps temperatures, pressures for each species from underlying grid
-- fast

The current implementation of the Emulator builds on the CombineOpacGrid, since its faster
"""
from functools import partial
from multiprocessing.pool import Pool

import numba
import numpy as np
import tqdm

from .utils.interp import interp_2d

DEFAULT_METHOD = "RORR"


@numba.njit(nogil=True, fastmath=True, cache=True)
def resort_rebin_njit(
    kout_conv, k1, k2, weights_in, weights_conv, Np, Nt, Nf, Ng
):
    """
    Resort and rebin the convoluted kappas. Note that this function works with g values calculated half integer.
    Fast, because it uses Numba

    Parameters
    ----------
    kout_conv: array(Np, Nt, Nf, Ng*Ng)
        the convoluted k-tables of species one and two
    k1: array(Np, Nt, Nf, Ng)
        ktable of species one
    k2: array(Np, Nt, Nf, Ng)
        ktable of species two
    weights_in: Ng
        The original weights (Delta g)
    weights_conv: Ng*Ng
        The weights of the convoluted k-tables (Delta g_1 * Delta g_2)
    Np: int
        number of pressure points in k-table grid
    Nt: int
        number of temperature points in k-table grid
    Nf: int
        number of frequency points in k-table grid
    Ng: int
        number of g value grid points in k-table grid

    Returns
    -------
    kout (array(Np, Nt, Nf, Ng)):
        RORR mixed kappa
    """

    # Initialize arrays
    kout = np.zeros((Np, Nt, Nf, Ng), dtype=np.float64)
    len_resort = Ng * Ng
    kout_conv_resorted = np.zeros(
        len_resort + 1, dtype=np.float64
    )  # note: We add +1 for the right edge
    g_resorted = np.zeros(
        len_resort + 1, dtype=np.float64
    )  # note: We add +1 for the right edge
    ggrid = compute_ggrid(weights_in, Ng)

    # Start looping over p, t and freq, because we need to do the resorting and rebinning individually

    for pi in range(Np):
        for ti in range(Nt):
            for freqi in range(Nf):
                # Sort and resort:
                index_sort = np.argsort(kout_conv[pi, ti, freqi])
                kout_conv_resorted[:len_resort] = kout_conv[pi, ti, freqi][
                    index_sort
                ]
                weights_resorted = weights_conv[index_sort]
                # compute new g-grid:
                g_resorted[:len_resort] = compute_ggrid(
                    weights_resorted, Ng * Ng
                )
                # edges:
                g_resorted[len_resort] = 1.0
                kout_conv_resorted[len_resort] = (
                    k1[pi, ti, freqi, -1] + k2[pi, ti, freqi, -1]
                )
                kout_conv_resorted[0] = (
                    k1[pi, ti, freqi, 0] + k2[pi, ti, freqi, 0]
                )
                # interpolate:
                kout[pi, ti, freqi, :] = np.interp(
                    ggrid, g_resorted, kout_conv_resorted
                )
    return kout


@numba.njit(nogil=True, fastmath=True, cache=True)
def compute_ggrid(w, Ng):
    """
    Helper function that calculates the ggrid for given weights. Works on a halfinteger grid.

    Parameters
    ----------
    w: array(Ng)
        weights ($Delta g$)
    Ng: int
        number of weights/ $g$-values


    Returns
    -------
    gcomp: array(Ng)
        The $g$ values
    """
    cum_sum = 0.0
    gcomp = np.empty(Ng)

    for i in range(Ng):
        gcomp[i] = cum_sum + 0.5 * w[i]
        cum_sum = cum_sum + w[i]

    return gcomp


@numba.njit(nogil=True, fastmath=True, cache=True)
def compute_weights(g, Ng):
    """
    Calculate $g$ values from weights ($Delta g$)

    Parameters
    ----------
    g: array(Ng)
        $g$ values
    Ng: int
        number of weights/ $g$-values

    Returns
    -------
    weights (array(Ng)):
        The weights ($Delta g$)
    """
    weights = np.empty(Ng)

    cum_sum = 0.0
    for i in range(Ng):
        weights[i] = 2.0 * (g[i] - cum_sum)
        cum_sum = cum_sum + weights[i]

    return weights


@numba.njit(nogil=True, fastmath=True, cache=True, parallel=False)
def _rorr_single(
    ktable,
    weights,
    weights_conv,
    ls,
    lf,
    lg,
    temp_old,
    press_old,
    lt_old,
    lp_old,
    input_data,
):
    """
    A numba accelerated function that performs RORR on one pressure and temperature point
    on the grid over all opacity species.

    Parameters
    ----------
    ktable: array(ls,lp,lt,lf,lg)
        The complete ktable grid
    weights: array(lg)
        The weights ($Delta g$)
    weights_conv: array(lg*lg)
        The convoluted weights ($Delta g_1*Delta g_2$)
    ls: int
        number of species
    lf: int
        number of frequency points
    lg: int
        number of $g$ values
    temp_old: array(lt)
        the temperature grid of the ktable grid
    press_old: array(lp)
        the pressure grid of the ktable grid
    lt_old: int
        the number of temperature grid points in the ktable grid
    lp_old: int
        the number of pressure grid points in the ktable grid
    input_data: array(ls+2)
        an array holding the individual abundances (mass mixing ratios) a_1,...,a_Ns ,
        Should come in the form of (a_1,...,a_N, p, T),
    Returns
    -------
    kout: array(lf,lg)
        the RORR mixed ktable at p and T from input_data

    """
    kout = np.empty((1, 1, lf, lg), dtype=np.float64)
    kout_conv = np.empty((1, 1, lf, lg * lg), dtype=np.float64)
    mixed_ktables = np.empty((ls, 1, 1, lf, lg), dtype=np.float64)

    temp = np.asarray([input_data[-1]])
    press = np.asarray([input_data[-2]])
    mmr = np.asarray(input_data[:-2])

    ki = interp_2d(
        temp_old,
        press_old,
        temp,
        press,
        ktable,
        ls,
        lf,
        lg,
        lt_old,
        lp_old,
        1,
        1,
    )

    for speci in range(ls):
        mixed_ktables[speci, 0, 0, :, :] = mmr[speci] * ki[speci, 0, 0, :, :]

    kout[:, :, :, :] = mixed_ktables[0, :, :, :, :]

    for speci in range(1, ls):
        k1 = kout
        k2 = mixed_ktables[speci, :, :, :, :]
        for gi in range(lg):
            for gj in range(lg):
                kout_conv[0, 0, :, gi + lg * gj] = (
                    k1[0, 0, :, gj] + k2[0, 0, :, gi]
                )

        kout = resort_rebin_njit(
            kout_conv, k1, k2, weights, weights_conv, 1, 1, lf, lg
        )

    return kout[0, 0, :, :]


class CombineOpac:
    """Parent class for CombineOpcacIndividual and CombineOpacGrid"""

    def __init__(self, opac):
        """
        The constructor of CombineOpac

        Parameters
        ----------
        opac: ReadOpac instance
            An instance of an ReadOpac child class, which has already been interpolated
        """
        self.opac = opac
        assert (
            self.opac.interp_done
        ), "yo, dude, you need to run setup_temp_and_pres on opac first"

    def add_batch(self, input_data, method=DEFAULT_METHOD):
        """mix multiple kgrids. Needs to be implemented in child class."""
        raise NotImplementedError

    def add_batch_parallel(self, input_data, method=DEFAULT_METHOD):
        """mix multiple kgrids. Needs to be implemented in child class."""
        raise NotImplementedError


class CombineOpacIndividual(CombineOpac):
    """A class for mixing arbitrary abundances and temperatures, pressures for each species
    """

    def add_batch(self, input_data, method=DEFAULT_METHOD):
        """mix the kgrid with a batch of different pressure temperature and abundances values.

        Parameters
        ----------
        input_data: array(batchsize, ls+2)
            input data to be mixed. Should be a two-dimensional array
            each input data sample should come in the form of (a_1,...,a_N, p, T),
            where a_i are the abundances and p and T are pressure and Temperature
        method: str
            The mixing method to be used (only RORR supported).

        Returns
        -------
        kappa: batchsize, lf, lg
            The mixed k-tables.
        """
        input_data = self._check_input_shape(input_data)
        mix_func = self._get_mix_func(method, use_mult=False)
        return mix_func(input_data)

    def add_batch_parallel(self, input_data, method=DEFAULT_METHOD):
        """mix one kgrid

        Parameters
        ----------
        input_data: array(batchsize, ls+2)
            input data to be mixed. Should be a two-dimensional array
            each input data sample should come in the form of (a_1,...,a_N, p, T),
            where a_i are the abundances and p and T are pressure and Temperature
        method: str
            The mixing method to be used (only RORR supported)

        Returns
        -------
        kappa: array(batchsize, lf, lg)
            The mixed k-tables.
        """
        input_data = self._check_input_shape(input_data)
        mix_func = self._get_mix_func(method, use_mult=True)
        return mix_func(input_data)

    def _get_mix_func(self, method, use_mult):
        """wraps the mixing function together with the underlying opacity data in a partial.

        Parameters
        ----------
        method: str
            The mixing method to be used (only RORR supported)
        use_mult: bool
            use or don't use multiprocessing

        Returns
        -------
        f:
            Function that takes the input data and returns the mix

        """
        if method == "RORR":
            return partial(
                self._add_rorr,
                self.opac.kcoeff,
                self.opac.weights,
                self.opac.Tr,
                self.opac.pr,
                use_mult,
            )

        raise NotImplementedError("Mixing method not implemented.")

    def _check_input_shape(self, input_data):
        """
        Checks that they are in the correct shape.

        Parameters
        ----------
        input_data (array(batchsize, ls+2)):
            input data to be mixed. Should be a two-dimensional array
            each input data sample should come in the form of (a_1,...,a_N, p, T),
            where a_i are the abundances and p and T are pressure and Temperature

        Returns
        -------
        input_data (array(batchsize, ls+2)):
            same input data, if no error is raised

        """
        assert len(input_data.shape) == 2
        assert input_data.shape[1] == self.opac.ls + 2
        return input_data

    @staticmethod
    def _add_rorr(ktable, weights, temp_old, press_old, use_mult, input_data):
        """Add up ktables by random overlap with resorting and rebinning.

        Parameters
        ----------
        ktable: array(ls, lp, lt, lf, lg)
            The kgrid with the individual ktables to be mixed,
            should have the same shape as kcoeff from a ReadOpac class
        weights: array(lg)
            The g-weights to be used
        temp_old: array(lt)
            The temperature in the grid of ktables
        press_old: array(lp)
            The pressure in the grid of ktables
        use_mult: bool
            do multiprocessing or not
        input_data: array(batchsize, ls+2)
            input data to be mixed. Should be a two-dimensional array
            each input data sample should come in the form of (a_1,...,a_N, p, T),
            where a_i are the abundances and p and T are pressure and Temperature

        Returns
        -------
        kappa: array(batchsize, lf, lg)
            The mixed k-tables.
        """
        Nsamples = input_data.shape[0]
        ls = input_data.shape[1] - 2
        lp_old = np.ones(ls, np.int8) * len(press_old)
        lt_old = np.ones(ls, np.int8) * len(temp_old)
        temp_old = np.ones((ls, lt_old[0])) * temp_old[np.newaxis, :]
        press_old = np.ones((ls, lp_old[0])) * press_old[np.newaxis, :]
        lf = ktable.shape[-2]
        lg = ktable.shape[-1]

        assert ls == ktable.shape[0]
        assert lp_old[0] == ktable.shape[1]
        assert lt_old[0] == ktable.shape[2]

        weights_conv = np.outer(weights, weights).flatten()

        func = partial(
            _rorr_single,
            ktable,
            weights,
            weights_conv,
            ls,
            lf,
            lg,
            temp_old,
            press_old,
            lt_old,
            lp_old,
        )

        if use_mult:
            with Pool() as pool:
                return np.asarray(
                    list(
                        tqdm.tqdm(
                            pool.imap(func, input_data, chunksize=100),
                            total=Nsamples,
                        )
                    ),
                    dtype=np.float64,
                )
        else:
            return np.asarray(
                list(tqdm.tqdm(map(func, input_data), total=Nsamples)),
                dtype=np.float64,
            )


class CombineOpacGrid(CombineOpac):
    """A class for mixing arbitrary abundances
    but keeps temperatures, pressures for each species from underlying grid"""

    def _get_mix_func(self, method):
        """
        Wraps the mixing function.

        Parameters
        ----------
        method: str
            Can be RORR, or linear.
            The mixing method to be used

        Returns
        -------
        f: function
            The mixing function

        """
        if method == "linear":
            # A simple sum
            return partial(self._add_linear, self.opac.kcoeff)
        if method == "RORR":
            # The RORR method
            return partial(self._add_rorr, self.opac.kcoeff, self.opac.weights)

        raise NotImplementedError("Method not implemented.")

    def _check_mmr_shape(self, mmr):
        """
        Reshapes the mass mixing ratios and check that they are in the correct shape.

        Parameters
        ----------
        mmr: array(ls, lp, lt) or dict
            The mass mxing ratios for every pressure-temperature grid point for all species.
            The mmr could be a dictionary of species names {spec_i: mmr_i for spec_i in self.opac.spec}

        Returns
        -------
        mmr: array(ls, lp, lt)
            The mass mixing ratios in proper shape
        """

        if isinstance(mmr, dict):
            mmr = np.array([mmr[speci] for speci in self.opac.spec])
        assert mmr.shape == (
            self.opac.ls,
            self.opac.lp[0],
            self.opac.lt[0],
        ), "shape of mmr needs to be species, pressure, temperature"
        return mmr

    def add_single(self, input_data, method=DEFAULT_METHOD):
        """
        mix one kgrid

        Parameters
        ----------
        input_data: array(ls, lp, lt) or dict:
            The mass mxing ratios for every pressure-temperature grid point for all species.
            The mmr could be a dictionary of species names {spec_i: mmr_i for spec_i in self.opac.spec}
        method: str
            Can be RORR, or linear.
            The mixing method to be used

        Returns
        -------
        kout: array(lp,lt,lf,lg)
            The mixed k tables
        """
        mmr = self._check_mmr_shape(input_data)
        mix_func = self._get_mix_func(method)
        return mix_func(mmr)

    def add_batch(self, input_data, method=DEFAULT_METHOD):
        """
        mix the kgrid multiple times.

        Parameters
        ----------
        input_data: array(batchsize, ls, lp, lt) or dict
            The mass mxing ratios for every pressure-temperature grid point for all species.
            The mmr could be a dictionary of species names {spec_i: mmr_i for spec_i in self.opac.spec}
        method: str
            Can be RORR, or linear.
            The mixing method to be used

        Returns
        -------
        kout: array(batchsize, lp,lt,lf,lg)
            The mixed k tables
        """
        mmr = [self._check_mmr_shape(mmr_i) for mmr_i in input_data]
        mix_func = self._get_mix_func(method)
        return np.asarray([mix_func(mmr_i) for mmr_i in tqdm.tqdm(mmr)])

    def add_batch_parallel(
        self, input_data, method=DEFAULT_METHOD, **pool_kwargs
    ):
        """Parallel version of add_batch

        Parameters
        ----------
        input_data: array(batchsize, ls, lp, lt) or dict
            The mass mxing ratios for every pressure-temperature grid point for all species.
            The mmr could be a dictionary of species names {spec_i: mmr_i for spec_i in self.opac.spec}
        method: str
            Can be RORR, or linear.
            The mixing method to be used
        pool_kwargs: dict
            anything else that may be of interest for the multiprocessing.Pool instance
            (e.g., pool size, etc.)

        Returns
        -------
        kout: array(batchsize, lp,lt,lf,lg)
            The mixed k tables
        """
        mmr = [self._check_mmr_shape(mmr_i) for mmr_i in input_data]
        mix_func = self._get_mix_func(method)
        with Pool(**pool_kwargs) as pool:
            return np.asarray(
                list(tqdm.tqdm(pool.imap(mix_func, mmr), total=len(mmr))),
                dtype=np.float64,
            )

    @staticmethod
    def _add_linear(ktable, mmr):
        """
        linear additive mixing of a kgrid.

        Parameters
        ----------
        ktable: array(ls,lp,lt,lf,lg)
            the ktable to be mixed
        mmr: array(ls, lp, lt)
            The mass mxing ratios for every pressure-temperature grid point for all species.

        Returns
        -------
        kout: array(lp,lt,lf,lg)
            The mixed k tables
        """
        return np.sum(ktable * mmr[:, :, :, np.newaxis, np.newaxis], axis=0)

    @staticmethod
    def _add_rorr(ktable, weights, mmr):
        """
        add up ktables by random overlap with resorting and rebinning.

        Parameters
        ----------
        ktable: array(ls,lp,lt,lf,lg)
            the ktable to be mixed
        weights: array(lg)
            The weights ($Delta g$) of the k-tables
        mmr: array(ls, lp, lt)
            The mass mxing ratios for every pressure-temperature grid point for all species.

        Returns
        -------
        kout: array(lp,lt,lf,lg)
            The mixed k tables
        """

        mixed_ktables = (
            mmr[:, :, :, np.newaxis, np.newaxis] * ktable[:, :, :, :, :]
        )
        kout = mixed_ktables[0, :, :, :, :]
        weights_conv = np.outer(weights, weights).flatten()

        for speci in range(1, ktable.shape[0]):
            k1 = kout
            k2 = mixed_ktables[speci]
            kout_conv = (
                k1[..., :, np.newaxis] + k2[..., np.newaxis, :]
            ).reshape(*kout.shape[:-1], weights_conv.shape[0])
            kout = resort_rebin_njit(
                kout_conv, k1, k2, weights, weights_conv, *kout.shape
            )

        return kout
