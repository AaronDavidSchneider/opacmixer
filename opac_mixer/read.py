"""The Module that houses the ktable grid model and read in capabilities"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

from .utils.interp import interp_2d


class ReadOpac:
    """
    The opacity reader base class

    The reader class only needs to define a read in function and pass important metadata to the constructor of the parent class. That's it.

    The constructor (`__init__`) needs to call the parent constructor with the following arguments:

    1. `ls (int)`: number of species that are read in
    2. `lp (array(ls))`: array that holds the number of pressure grid points for each species
    3. `lt (array(ls))`: array that holds the number of temperature grid points for each species
    4. `lf (array(ls))`: array that holds the number of frequency grid points for each species
    5. `lg (array(ls))`: array that holds the number of $g$ grid points for each species

    *Note, that we require that `lf[0]==lf[i]` and `lg[0]==lg[i]` for all i in number of species*

    The read in function (`read_opac`) has to fill the following arrays:

    1. `self.spec (array(ls)`: array holding the names of the opacity species
    2. `self.T (array(ls, max(lt)))`: array holding the temperature in K at which the k-table grid is defined
    3. `self.p (array(ls, max(lp)))`: array holding the pressure values in bar at which the k-table grid is defined
    4. `self.bin_edges (array(ls, lf[0]+1))`: array holding the wave number ($1/lambda$) values in 1/cm of the edges of the wavenumber grid at which the k-table grid is defined
    5. `self.bin_center (array(ls, lf[0]))`: array holding the wave number ($1/lambda$) values in 1/cm of the center of the wavenumber grid at which the k-table grid is defined.
    6. `self.weights (array(ls, lg[0]))`: array holding the weights of the k-tables (see below for conversion from $g$ values)
    7. `self.kcoeff (array(ls, max(lp), max(lt), lf[0], lg[0])`: array holding the actual values of the k-table grid in cm2/g.

    Note, the data arrays are initialized with space up unto the maximum number of temperature and pressure grid points, hence the `max(lt)` and `max(lp)`.

    Note, that we need weights instead of g-values. The conversion between the two can be done using these two functions:
    compute_ggrid(w, Ng), compute_weights(g, Ng) from mix.py
    """

    def __init__(self, ls, lp, lt, lf, lg):
        """
        Construct the reader. Initialize all arrays.

        Parameters
        ----------
        ls: int
            number of species that are read in
        lp: array(ls)
            array that holds the number of pressure grid points for each species
        lt: array(ls)
            array that holds the number of temperature grid points for each species
        lf: array(ls)
            array that holds the number of frequency grid points for each species
        lg: array(ls)
            array that holds the number of $g$ grid points for each species
        """

        self.ls, self.lp, self.lt, self.lf, self.lg = ls, lp, lt, lf, lg

        assert self.ls > 1, "no files found"
        assert len(set(self.lf)) <= 1, "frequency needs to match"
        assert len(set(self.lg)) <= 1, "g grid needs to match"

        # initialize arrays:
        self.kcoeff = np.zeros(
            (self.ls, self.lp.max(), self.lt.max(), self.lf[0], self.lg[0]),
            dtype=np.float64,
        )
        self.bin_edges = np.zeros(
            self.lf[0] + 1, dtype=np.float64
        )  # wavenumbers at edges (1/lambda) in 1/cm
        self.bin_center = np.zeros(
            self.lf[0], dtype=np.float64
        )  # wavenumbers at center(1/lambda) in 1/cm
        self.weights = np.zeros(
            self.lg[0], dtype=np.float64
        )  # ktable weights of distribution function
        self.T = np.zeros(
            (self.ls, self.lt.max()), dtype=np.float64
        )  # temperature of k-table grid for each species in K
        self.p = np.zeros(
            (self.ls, self.lp.max()), dtype=np.float64
        )  # pressure of k-table grid for each species in bar
        self.spec = self.ls * [""]  # names of opacity species

        # Initialize reduced arrays (will only be set during interpolation)
        self.pr = np.empty(
            self.lp.max(), dtype=np.float64
        )  # pressure in interpolated k table grid
        self.Tr = np.empty(
            self.lt.max(), dtype=np.float64
        )  # temperature in interpolated k table grid
        self.interp_done = False  # flag to indicate sucessful interpolation
        self.read_done = False  # flag to indicate read in

    def read_opac(self):
        """read in the opacity, dependent on the opac IO model."""
        self.read_done = True
        return NotImplementedError("to be implemented in childclass")

    def setup_temp_and_pres(self, temp=None, pres=None):
        """
        Interpolate k coeffs to different pressure and temperature values.

        Parameters
        ----------
        temp: optional, array-like
            A 1D temperature array (K) to which the k-table grid should be interpolated to.
            If not set, it wil use a linspace grid between the maximum and minimum found in the temperature grids.
        pres: optional, array-like
            A 1D pressure array (bar) to which the k-table grid should be interpolated to.
            If not set, it wil use a logspace grid between the maximum and minimum found in the pressure grids.

        Note that, right now, it takes the values outside of the defined range to be the last defined values.
        """

        assert self.read_done, "run read_opac first"
        if pres is None:
            pmin = min(min(self.p[i, : self.lp[i]]) for i in range(self.ls))
            pres = np.logspace(
                np.log10(pmin), np.log10(self.p.max()), len(self.p[0])
            )
        else:
            pres = np.array(pres)

        if temp is None:
            tmin = min(min(self.T[i, : self.lt[i]]) for i in range(self.ls))
            temp = np.logspace(
                np.log10(tmin), np.log10(self.T.max()), len(self.T[0])
            )
        else:
            temp = np.array(temp)

        lp_new = self.ls * [len(pres)]
        lt_new = self.ls * [len(temp)]

        self.kcoeff = interp_2d(
            self.T,
            self.p,
            temp,
            pres,
            self.kcoeff,
            self.ls,
            self.lf[0],
            self.lg[0],
            self.lt,
            self.lp,
            lt_new[0],
            lp_new[0],
        )

        self.pr = pres  # the new pressure values (same for all species)
        self.Tr = temp  # the new temperature values (same for all species)
        self.T = np.ones((self.ls, lt_new[0]), dtype=np.float64) * temp
        self.p = np.ones((self.ls, lp_new[0]), dtype=np.float64) * pres
        self.lp = lp_new
        self.lt = lt_new
        self.interp_done = True

    def remove_sparse_frequencies(self):
        """Check for zeros in the opacity and remove them"""

        # Search for the zeros in every species
        nonzero_index = np.empty((self.ls, self.lf[0]))
        for i in range(self.ls):
            nonzero_index[i] = np.all(
                self.kcoeff[i, : self.lp[i], : self.lt[i], :, :],
                axis=(0, 1, 3),
            )

        # Search for common zeros in every species
        nonzero_index = np.all(nonzero_index, axis=0)

        # Construct the array for the edges
        edges_nonzero = np.ones(self.lf[0] + 1)  # default case, no zeros
        if not nonzero_index[0] or not nonzero_index[-1]:
            # We need to add the outer borders to the edges
            edges_nonzero = np.append(nonzero_index, 1.0)
        else:
            if np.count_nonzero(nonzero_index) != self.lf[0]:
                # We want that the zeros start at the frequency edges
                # nonzero_index[-1] or nonzero_index[0] would then need to be zero
                raise ValueError(
                    "zeros in the middle. Cant handle that. It makes no sense."
                )

        # adapt the members accordingly
        self.lf = np.repeat(np.count_nonzero(nonzero_index), self.ls)
        self.bin_edges = self.bin_edges[np.asarray(edges_nonzero, dtype=bool)]
        self.bin_center = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])
        self.kcoeff = self.kcoeff[
            :, :, :, np.asarray(nonzero_index, dtype=bool), :
        ]

    def plot_opac(self, pres, temp, spec, ax=None, **plot_kwargs):
        """
        Simple plotting routine of the opacity.

        Parameters
        ----------
        pres: float
            pressure at which the opacity is to be plotted, will pick closest lower point
        temp: float
            temperature at which the opacity is to be plotted, will pick closest lower point
        spec: str
            name of species to plot
        ax: matplotlib ax
            optional, matplotlib ax object on which the plot should be placed
        plot_kwargs:
            everything else will be just passed to the plotting routine

        Returns
        -------
        lines: list
            list of line plots
        """
        if ax is None:
            ax = plt.gca()

        speci = self.spec.index(spec)
        pi = np.searchsorted(self.p[speci], pres) - 1
        ti = np.searchsorted(self.T[speci], temp) - 1
        print("p:", self.p[speci, pi])
        print("T:", self.T[speci, ti])

        lines = []
        for fi in range(self.lf[0]):
            x = self.bin_edges[fi] + self.weights.cumsum() * (
                self.bin_edges[fi + 1] - self.bin_edges[fi]
            )
            lines.append(
                ax.loglog(x, self.kcoeff[speci, pi, ti, fi, :], **plot_kwargs)
            )

        return lines


class ReadOpacChubb(ReadOpac):
    """A ktable grid reader for the ExomolOP-pRT k-table format"""

    def __init__(self, files) -> None:
        """
        Construct the chubb reader for the ExomolOP-pRT k-table format.

        Parameters
        ----------
        files: list
            A list of filenames of the h5 files in which the k-tables are stored.
        """
        ls = len(
            files
        )  # Number of opacity species is the number of k-table grid files
        self._files = files  # This is custom to this reader, since we do the readin later

        # initialize the arrays that hold the dimensions
        # of pressure, temperature, frequency and g values for each species
        lp, lt, lf, lg = (
            np.empty(ls, dtype=int),
            np.empty(ls, dtype=int),
            np.empty(ls, dtype=int),
            np.empty(ls, dtype=int),
        )

        # read in this metadata for all species
        for i, file in enumerate(files):
            with h5py.File(file) as f:
                lp[i], lt[i], lf[i], lg[i] = np.array(f["kcoeff"]).shape

        # call the parent constructor with the metadata
        super().__init__(ls, lp, lt, lf, lg)

    def read_opac(self):
        """Read in the kcoeff from h5 file."""
        # initialize some arrays
        bin_edges = np.empty((self.ls, self.lf[0] + 1), dtype=np.float64)
        weights = np.empty((self.ls, self.lg[0]), dtype=np.float64)

        # Iterate over all species and fill in the data
        for i, file in enumerate(self._files):
            with h5py.File(file) as f:
                bin_edges[i, :] = np.array(f["bin_edges"], dtype=np.float64)
                weights[i, :] = np.array(f["weights"], dtype=np.float64)

                # store species name
                self.spec[i] = f["mol_name"][0].decode("ascii")

                # store pressure and temperature of the opacity species
                self.T[i, : self.lt[i]] = np.array(f["t"], dtype=np.float64)
                self.p[i, : self.lp[i]] = np.array(f["p"], dtype=np.float64)

                # convert k-table grid from cm2/mol to cm2/g:
                conversion_factor = 1 / (
                    np.float64(f["mol_mass"][0]) * const.atomic_mass * 1000
                )
                kcoeff = (
                    np.array(f["kcoeff"], dtype=np.float64) * conversion_factor
                )

                # store ktable grid
                self.kcoeff[i, : self.lp[i], : self.lt[i], :, :] = kcoeff

        # Do the check if the frequencies and g values are the same for all species
        assert np.all(
            bin_edges[1:, :] == bin_edges[:-1, :]
        ), "frequency needs to match"
        assert np.all(
            weights[1:, :] == weights[:-1, :]
        ), "g grid needs to match"

        # store the weights and frequency edges
        self.weights = weights[0, :]
        self.bin_edges = bin_edges[0, :]

        # This removes those frequencies from the grid that have no k-table data (kappa=0)
        self.remove_sparse_frequencies()  # this function also sets self.bin_center

        # Set the read_done switch to true, since we are done with reading in the ktables
        self.read_done = True
