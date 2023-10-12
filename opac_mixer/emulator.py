"""This module houses the Emulator of the opacity mixing"""
import os.path

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from MITgcmutils import wrmds
from sklearn.metrics import (
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras

from opac_mixer.utils.callbacks import CustomCallback
from opac_mixer.utils.models import get_deepset
from opac_mixer.utils.scalings import (
    default_input_scaling,
    default_output_scaling,
    default_inv_output_scaling,
)
from .mix import CombineOpacGrid
from .read import ReadOpac

DEFAULT_PRANGE = (1e-6, 1000, 30)
DEFAULT_TRANGE = (100, 10000, 30)

DEFAULT_MMR_RANGES = {
    "CO": (1e-30, 0.005522337070205542),
    "H2O": (3.509581940975492e-22, 0.0057565911404275204),
    "HCN": (1e-30, 9.103077483740115e-05),
    "C2H2,acetylene": (1e-30, 1.581540423097846e-05),
    "CH4": (1e-30, 0.0031631031028604537),
    "PH3": (1e-30, 6.401082202603451e-06),
    "CO2": (1e-30, 0.00015319944152172055),
    "NH3": (3.8119208513224578e-25, 0.00084362326521647),
    "H2S": (2.0093762682408387e-18, 0.0003290905470710346),
    "VO": (1e-30, 1.6153195092178982e-07),
    "TiO": (1e-30, 3.925184850731112e-06),
    "Na": (1e-30, 2.524986071526357e-05),
    "K": (1e-30, 1.932224843084919e-06),
    "SiO": (1e-30, 0.0010448970102509476),
    "FeH": (1e-30, 0.000203477300968298),
}


class DataIO:
    """IO Class for storing the emulator"""

    def __init__(self, filename):
        """
        Setup the IO class

        Parameters
        ----------
        filename: str
            The filename of the IO
        """
        self.filename = filename

    def load(self):
        """load data"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                "we could not load the data, because it doesnt exist."
            )

        with h5py.File(self.filename, "r") as f:
            input_data = np.asarray(f["input"])
            mix = np.asarray(f["mix"])
            split_seed = int(f["mix"].attrs["split_seed"])
            test_size = float(f["mix"].attrs["test_size"])

        return mix, input_data, split_seed, test_size

    def write_out(self, mix, input_data, split_seed, test_size):
        """
        write data

        Parameters
        ----------
        mix: array(batchsize, lg)
            The mixed k-tables
        input_data: array(batchsize, lg, ls)
            The input data
        split_seed: float
            The random seed used in splitting training and test data
        test_size: float
            The size of the training set
        """
        with h5py.File(self.filename, "w") as f:
            inp_ds = f.create_dataset(
                "input", input_data.shape, dtype=input_data.dtype
            )
            mix_ds = f.create_dataset("mix", mix.shape, dtype=input_data.dtype)
            mix_ds.attrs["split_seed"] = split_seed
            mix_ds.attrs["test_size"] = test_size
            mix_ds[...] = mix
            inp_ds[...] = input_data


class Emulator:
    """
    The supervisor that handels the training and evaluation of the opacity emulator.
    """

    def __init__(
        self,
        opac,
        prange_opacset=DEFAULT_PRANGE,
        trange_opacset=DEFAULT_TRANGE,
        filename_data=None,
    ):
        """
        Construct the emulator class.

        Parameters
        ----------
        opac: opac_mixer.read.ReadOpac
            a list of input opacity readers. Can be setup, but does not need to. Will do the setup itself otherwise.
        prange_opacset: array(3)
            optional, the range to which the reader should interpolate the pressure grid to (lower, upper, num_points).
        trange_opacset: array(3)
            optional, the range to which the reader should interpolate the temperature grid to (lower, upper, num_points).
        filename_data: str
            A filename, used to save the training and testing data to
        """
        if isinstance(opac, list):
            assert all(isinstance(opac_i, ReadOpac) for opac_i in opac)
            self.opac = opac
        elif isinstance(opac, ReadOpac):
            self.opac = [opac]
        else:
            raise ValueError(
                "opac needs to be either a list of ReadOpac or a ReadOpac"
                " instance"
            )

        for opac_i in self.opac:
            if not opac_i.read_done:
                opac_i.read_opac()
            if not opac_i.interp_done:
                opac_i.setup_temp_and_pres(
                    pres=np.logspace(
                        np.log10(prange_opacset[0]),
                        np.log10(prange_opacset[1]),
                        prange_opacset[2],
                    ),
                    temp=np.linspace(*trange_opacset),
                )

        self.input_scaling = default_input_scaling
        self.output_scaling = default_output_scaling
        self.inv_output_scaling = default_inv_output_scaling

        self.mixer = [CombineOpacGrid(opac) for opac in self.opac]

        ls = [int(opac.ls) for opac in self.opac]
        lg = [int(opac.lg[0]) for opac in self.opac]

        if len(ls) > 1:
            assert (
                np.diff(ls) == 0.0
            ), "we need the same number of species for all ReadOpac instances"
            assert (
                np.diff(lg) == 0.0
            ), "we need the same number of g points for all ReadOpac instances"

        self._lg = lg[0]
        self._ls = ls[0]
        self._input_dim = (self._lg, self._ls)

        if filename_data is not None:
            self._io = DataIO(filename=filename_data)

        # initialisation of variables
        self._has_input = False
        self._has_mix = False
        self._has_model = False
        self._is_trained = False

        self._batchsize_resh = []
        self._batchsize = []
        self.abus = []
        self.verbose = False

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self._model_filename = None

        self.input_data = np.empty((0, *self._input_dim))

    def setup_scaling(
        self, input_scaling=None, output_scaling=None, inv_output_scaling=None
    ):
        """
        (optional) Change the callback functions for the scaling of in and output.
        Defaults are given as opac_mixer.scalings.default_<name>.
        See opac_mixer/utils/scalings.py for inspiration

        Parameters
        ----------
        input_scaling: function or None
            The function to use for input scaling.
            If None, use opac_mixer.scalings.default_input_scaling
        output_scaling: function or None
            The function to use for output scaling.
            If None, use opac_mixer.scalings.default_output_scaling
        inv_output_scaling: function or None
            The function to use for output scaling.
            If None, use opac_mixer.scalings.default_inv_output_scaling
        """
        if input_scaling is not None:
            self.input_scaling = input_scaling
        if output_scaling is not None:
            self.output_scaling = output_scaling
        if inv_output_scaling is not None:
            self.inv_output_scaling = inv_output_scaling

        if hasattr(self, "X_train"):
            np.testing.assert_allclose(
                self.inv_output_scaling(
                    self.X_train,
                    self.output_scaling(self.X_train, self.y_train),
                ),
                self.y_train,
            )
        if hasattr(self, "X_test"):
            np.testing.assert_allclose(
                self.inv_output_scaling(
                    self.X_test, self.output_scaling(self.X_test, self.y_test)
                ),
                self.y_test,
            )

    def setup_sampling_grid(
        self, approx_batchsize=8e5, extra_abus=None, bounds=None
    ):
        """
        Setup the sampling grid. Sampling along MMR and pressure is in logspace.
        Sampling along temperature is in linspace.

        Parameters
        ----------
        approx_batchsize: int
            Number of total sampling points. Needs to be a power of 2 for sobol sampling
        bounds: dict or None
            the lower and upper bounds for sampling. Shape: {'species':(lower, upper)}
            The key can be either a species name in opac.spec or p and T for pressure and Temperature.
            It will use opac_mixer.emulator.DEFAULT_MMR_RANGES for mmrs, opac_mixer.emulator.DEFAULT_PRANGE for pressure,
            and opac_mixer.emulator.DEFAULT_TRANGE for temperautre for all missing values
        extra_abus: array(num_sample, ls, lp, lt)
            Extra abundancies (mmrs) used for the training data. Could be e.g., a grid of eq. chem abundancies

        Returns
        -------
        input_data (array(batchsize, opac.lg, opac.ls)):
            The sampled inputdata to train/test the emulator.
            The input_data consists of kappas(g) for each species
        """
        if bounds is None:
            bounds = {}
        self._batchsize_resh = []
        self._batchsize = []
        self.abus = []

        self.input_data = np.empty((0, *self._input_dim))

        for opac in self.opac:
            if extra_abus is not None:
                assert extra_abus.shape[1] == self._ls, (
                    "wrong shape in extra_abus second dimension (number of"
                    " species)"
                )
                assert extra_abus.shape[2] == opac.lp[0], (
                    "wrong shape in extra_abus second dimension (number of"
                    " pressure points)"
                )
                assert extra_abus.shape[3] == opac.lt[0], (
                    "wrong shape in extra_abus second dimension (number of"
                    " temperature points)"
                )
                extra_batchsize = int(extra_abus.shape[0])
            else:
                extra_batchsize = 0

            batchsize = (
                int(approx_batchsize) // opac.lp[0] // opac.lt[0] // opac.lf[0]
            )
            batchsize_resh = batchsize * opac.lp[0] * opac.lt[0] * opac.lf[0]
            self._batchsize_resh.append(batchsize_resh)
            self._batchsize.append(batchsize)

            l_bounds = []
            u_bounds = []
            for sp in opac.spec:
                if sp not in DEFAULT_MMR_RANGES and sp not in bounds:
                    raise ValueError(f"We miss the bounds for {sp}.")

                default_l, default_u = DEFAULT_MMR_RANGES.get(sp)
                l, u = bounds.get(sp, (default_l, default_u))
                l_bounds.append(np.maximum(l, 1.0e-20))
                u_bounds.append(u)

            # Use a standard uniform distribution
            sample = np.random.uniform(
                low=0.0,
                high=1.0,
                size=(
                    batchsize - extra_batchsize,
                    self._ls,
                    opac.lp[0],
                    opac.lt[0],
                ),
            )

            # Scale the sampling to the actual bounds
            # Note: We use loguniform like scaling for mmrs

            abus = np.exp(
                sample[:, :, :, :]
                * (
                    np.log(u_bounds)[np.newaxis, :, np.newaxis, np.newaxis]
                    - np.log(l_bounds)[np.newaxis, :, np.newaxis, np.newaxis]
                )
                + np.log(l_bounds)[np.newaxis, :, np.newaxis, np.newaxis]
            )

            if extra_abus is not None:
                abus = np.concatenate((abus, extra_abus), axis=0)

            weighted_kappas = (
                abus[:, :, :, :, np.newaxis, np.newaxis]
                * opac.kcoeff[np.newaxis, ...]
            )
            weighted_kappas = weighted_kappas.transpose((0, 2, 3, 4, 1, 5))
            self.abus.append(abus)

            self.input_data = np.concatenate(
                (
                    self.input_data,
                    weighted_kappas.reshape(
                        batchsize_resh, opac.ls, opac.lg[0]
                    ).transpose(0, 2, 1),
                ),
                axis=0,
            )

        self._check_input_data(self.input_data)

        self._has_input = True

        return self.input_data

    def _check_input_data(self, input_data):
        """
        Checks the shape of the input data and raises an error if wrong

        Parameters
        ----------
        input_data: array(batchsize, opac.lg, opac.ls
            The input data
        """
        shape = input_data.shape
        if len(shape) != 3 or shape[1:] != self._input_dim:
            raise ValueError("input data does not match")
        assert (input_data >= 0).all(), "We need positive input data!"

    def setup_mix(self, test_size=0.2, split_seed=None, do_parallel=True):
        """
        Setup the mixer and generate the training and testdata.

        Parameters
        ----------
        test_size: float
            fraction of data used for testing
        split_seed: int
            A seed to be used for shuffling training and test data before splitting
        do_parallel bool
            If you want to create the data in parallel or not
        """
        if not self._has_input:
            raise AttributeError(
                "we do not have input yet. Run setup_sampling_grid first."
            )

        if split_seed is None:
            split_seed = np.random.randint(2**32 - 1)

        # make sure the filename comes without the npy suffix
        mixes = np.empty((0, self._lg))
        for mixer, abus, batchsize_resh in zip(
            self.mixer, self.abus, self._batchsize_resh
        ):
            if do_parallel:
                mix = mixer.add_batch_parallel(abus)
            else:
                mix = mixer.add_batch(abus)

            mixes = np.concatenate(
                (mixes, mix.reshape(batchsize_resh, self._lg)), axis=0
            )

        self.X_train, self.X_test, self.y_train, self.y_test = self._do_split(
            self.input_data, mixes, split_seed, test_size, use_split_seed=True
        )

        if hasattr(self, "_io"):
            self._io.write_out(mixes, self.input_data, split_seed, test_size)

        self._has_mix = True

        return self.X_train, self.X_test, self.y_train, self.y_test

    def load_data(
        self,
        filename=None,
        test_size=None,
        split_seed=None,
        use_split_seed=True,
    ):
        """
        Load the training and test data from a h5 file.

        Parameters
        ----------
        filename: str
            optional, can be set either here or in the constructor. Make sure the filename comes without the npy suffix
        test_size: float
            optional, use a different test size than the one loaded
        split_seed: int
            optional, use a different seed to shuffle data before spliting training and testing data
        use_split_seed: bool
            optional, if true, it will just use the provided or loaded split seed, else it will create a new random one
        """

        if not hasattr(self, "_io") and filename is None:
            raise ValueError(
                "we have no clue where we could get the data from. Set a"
                " filename either in this method or the constructor"
            )

        if filename is not None:
            self._io = DataIO(filename=filename)

        mix, input_data, split_seed_l, test_size_l = self._io.load()

        self.input_data = input_data
        self._check_input_data(self.input_data)

        if test_size is None:
            test_size = test_size_l
        if split_seed is None:
            split_seed = split_seed_l

        self.X_train, self.X_test, self.y_train, self.y_test = self._do_split(
            self.input_data, mix, split_seed, test_size, use_split_seed
        )

        self._has_input = True
        self._has_mix = True

        return self.X_train, self.X_test, self.y_train, self.y_test

    @staticmethod
    def _do_split(input_data, mix, split_seed, test_size, use_split_seed=True):
        """Do the split of training and testing data.

        Parameters
        ----------
        input_data: array(batchsize, opac.lg, opac.ls)
            The input (X)
        mix: array(batchsize, opac.lg)
            The output (y)
        split_seed: float
            a specific random seed to use
        test_size: float
            The size of the test-set
        use_split_seed: bool
            If the split seed is to be used or not
        """
        if (mix <= 0).any():
            raise ValueError(
                "We found negative crosssections. Something is wrong here."
            )

        return train_test_split(
            input_data,
            mix,
            test_size=test_size,
            random_state=split_seed if use_split_seed else None,
        )

    def setup_model(
        self,
        model=None,
        filename=None,
        load=False,
        learning_rate=1e-3,
        hidden_units=None,
        verbose=True,
        **model_kwargs,
    ):
        """
        Setup the emulator model and train it.
        Note: This will reset all previously trained weights in keras models.

        Parameters
        ----------
        model: sklearn compatible model
            (optional): a model to learn. Needs to be contructed already. Use DeepSet by default
        filename: str or None
            optional, A filename to save the model
        load: bool
            optional, load a -pretrained- model instead of constructing one


        Parameters for DeepSet
        ----------------------
        Check keras.compile docs for more arguments. Any extra argument is directly passed to keras.compile
        learning_rate: float
            optional, learning rate of optimizer (adam per default, change by setting optimizer=<name>)
        hidden_units: int
            optional, number of hidden units in the encoder (per default equals number of g-points)

        (model_kwargs)
            arguments to pass to keras.compile for construction (only when model=None is used)
        """
        keras.backend.clear_session()
        self.verbose = verbose

        if load:
            # Load model
            self.model = keras.models.load_model(filename)
            self._is_trained = True
        else:
            if not self._has_mix:
                raise AttributeError(
                    "we do not have a mix to work with yet. Run"
                    " setup_sampling_grid and setup_mix first."
                )

            if model is None:
                self.model = get_deepset(
                    ng=self._lg, hidden_units=hidden_units
                )
            elif model is not None:
                print(
                    "WARNING: make sure your model is permutation invariant!"
                )
                # Use provided model (needs to be sklearn compatible)
                self.model = model

            if isinstance(self.model, keras.Model):
                extra_model_kwargs = {
                    "optimizer": keras.optimizers.Adam(
                        learning_rate=learning_rate
                    ),
                    "loss": "mean_squared_error",
                }
                extra_model_kwargs.update(model_kwargs)

                self.model.compile(**extra_model_kwargs)
                if self.verbose:
                    self.model.summary()

            if filename is not None:
                # Save filename for later use
                self._model_filename = filename

        self._has_model = True

    def fit(self, *args, **kwargs):
        """
        Train the model.

        Parameters
        ----------
        args:
            Whatever you want to pass to the model to fit
        kwargs:
            Whatever you want to pass to the model to fit

        """
        fit_args = {}
        if isinstance(self.model, keras.Model):
            fit_args["epochs"] = 100
            fit_args["batch_size"] = 32
            fit_args["verbose"] = 0

        fit_args.update(kwargs)

        if not self._has_model:
            raise AttributeError(
                "we do not have a model yet. Run setup_sampling_grid,"
                " setup_mix and setup_model first."
            )

        # fit the model on the training dataset
        X_train = self.input_scaling(self.X_train)
        y_train = self.output_scaling(self.X_train, self.y_train)

        if isinstance(self.model, keras.Model):
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="loss", patience=3)
            ]
            if self.verbose:
                callbacks.append(CustomCallback(self))

            self.model.fit(X_train, y_train, callbacks=callbacks, **fit_args)
        else:
            self.model.fit(
                X_train.reshape(len(X_train), -1), y_train, *args, **kwargs
            )

        if hasattr(self, "_model_filename") and callable(
            getattr(self.model, "save", None)
        ):
            print(f"Saving model to {self._model_filename}")
            self.model.save(self._model_filename)

        self._is_trained = True

    def predict(self, X, *args, **kwargs):
        """
        Predict using the trained model.

        Parameters
        ----------
        X: array(num_samples, opac.lg, opac.ls
            The values you want predictions for
        args:
            Whatever you want to pass to the model for prediction
        kwargs:
            Whatever you want to pass to the model for prediction
        """
        self._check_trained()
        self._check_input_data(X)

        # fit the model on the training dataset
        if isinstance(self.model, keras.Model):
            verbose = 0 if not self.verbose else "auto"
            return self.inv_output_scaling(
                X,
                self.model.predict(
                    self.input_scaling(X), verbose=verbose, *args, **kwargs
                ),
            )

        return self.inv_output_scaling(
            X,
            self.model.predict(
                self.input_scaling(X).reshape(len(X), -1), *args, **kwargs
            ),
        )

    def score(self, validation_set=None):
        """
        Print some metrics for the training and test data.

        Parameters
        ----------
        validation_set: list(X_test, y_test)
            validation set to be used instead of (self.X_test, self.y_test)
            Note the dimensions of X_test: array(num_samples, opac.lg, opac.ls)
            and y_test: array(num_samples, opac.lg)
        """
        if validation_set is None:
            X_test = self.X_test
            y_test = self.y_test
        else:
            X_test = validation_set[0]
            y_test = validation_set[1]
            self._check_input_data(X_test)

        self._check_trained()

        y_p_test = self.predict(X_test)
        y_p_train = self.predict(self.X_train)

        train_mask = self.y_train > 1e-45
        test_mask = y_test > 1e-45

        log_err_out = (y_p_train[train_mask] > 0).all()

        y_add_test = np.sum(X_test, axis=-1)
        y_add_train = np.sum(self.X_train, axis=-1)

        y_test_masked = y_test[test_mask]
        y_train_masked = self.y_train[train_mask]
        y_p_test_masked = y_p_test[test_mask]
        y_p_train_masked = y_p_train[train_mask]
        y_add_test_masked = y_add_test[test_mask]
        y_add_train_masked = y_add_train[train_mask]

        e_add_test = np.sqrt(
            mean_squared_error(y_test_masked, y_add_test_masked)
        )
        e_add_train = np.sqrt(
            mean_squared_error(y_train_masked, y_add_train_masked)
        )
        e_p_test = np.sqrt(mean_squared_error(y_test_masked, y_p_test_masked))
        e_p_train = np.sqrt(
            mean_squared_error(y_train_masked, y_p_train_masked)
        )

        if log_err_out:
            e_log_add_test = np.sqrt(
                mean_squared_log_error(y_test_masked, y_add_test_masked)
            )
            e_log_add_train = np.sqrt(
                mean_squared_log_error(y_train_masked, y_add_train_masked)
            )
            e_log_p_test = np.sqrt(
                mean_squared_log_error(y_test_masked, y_p_test_masked)
            )
            e_log_p_train = np.sqrt(
                mean_squared_log_error(y_train_masked, y_p_train_masked)
            )

        r2_add_test = r2_score(y_test_masked, y_add_test_masked)
        r2_add_train = r2_score(y_train_masked, y_add_train_masked)
        r2_p_test = r2_score(y_test_masked, y_p_test_masked)
        r2_p_train = r2_score(y_train_masked, y_p_train_masked)

        if log_err_out:
            r2_log_add_test = r2_score(
                np.log(y_test_masked), np.log(y_add_test_masked)
            )
            r2_log_add_train = r2_score(
                np.log(y_train_masked), np.log(y_add_train_masked)
            )
            r2_log_p_test = r2_score(
                np.log(y_test_masked), np.log(y_p_test_masked)
            )
            r2_log_p_train = r2_score(
                np.log(y_train_masked), np.log(y_p_train_masked)
            )

        print(
            "test           (add, model): {:.2e}, {:.2e}".format(
                e_add_test, e_p_test
            )
        )
        print(
            "train          (add, model): {:.2e}, {:.2e}".format(
                e_add_train, e_p_train
            )
        )
        if log_err_out:
            print(
                "log test       (add, model): {:.2e}, {:.2e}".format(
                    e_log_add_test, e_log_p_test
                )
            )
            print(
                "log train      (add, model): {:.2e}, {:.2e}".format(
                    e_log_add_train, e_log_p_train
                )
            )
        print(
            "r2 test        (add, model): {:.2e}, {:.2e}".format(
                r2_add_test, r2_p_test
            )
        )
        print(
            "r2 train       (add, model): {:.2e}, {:.2e}".format(
                r2_add_train, r2_p_train
            )
        )
        if log_err_out:
            print(
                "log r2 test    (add, model): {:.2e}, {:.2e}".format(
                    r2_log_add_test, r2_log_p_test
                )
            )
            print(
                "log r2 train   (add, model): {:.2e}, {:.2e}".format(
                    r2_log_add_train, r2_log_p_train
                )
            )

    def _check_trained(self):
        """Just a random check if the model has been trained or not."""
        if not self._is_trained:
            raise AttributeError(
                "we do not have a trained model yet. Run setup_sampling_grid,"
                " setup_mix and setup_model and fit first."
            )

    def export(self, path, file_format="exorad"):
        """
        Export the weights

        Parameters
        ----------
        path: str
            path where the weights should be stored
        file_format: str
            the format in which the weights should be stored. Can be either exorad or numpy.
        """
        self._check_trained()
        if isinstance(self.model, keras.Model):
            for i, weights in enumerate(self.model.weights):
                if file_format in ("np", "numpy"):
                    np.save(f"{path}/ml_coeff_{i}.npy", weights.numpy())
                elif file_format == "exorad":
                    wrmds(
                        f"{path}/ml_coeff_{i}",
                        weights.numpy().flatten(order="F"),
                        dataprec="float64",
                    )
                else:
                    raise NotImplementedError("format is not supported yet.")
        else:
            raise NotImplementedError(
                "not implemented for the type of model used"
            )

    def plot_predictions(self, validation_set=None):
        """
        Plot the predictions vs the true values

        Parameters
        ----------
        validation_set: list(X_test, y_test)
            validation set to be used instead of (self.X_test, self.y_test)
            Note the dimensions of X_test: array(num_samples, opac.lg, opac.ls)
            and y_test: array(num_samples, opac.lg)
        """

        if validation_set is None:
            X_test = self.X_test
            y_test = self.y_test
        else:
            X_test = validation_set[0]
            self._check_input_data(X_test)
            y_test = validation_set[1]

        y_p_test = self.predict(X_test)
        y_add_test = np.sum(X_test, axis=-1)

        for index in range(y_p_test.shape[-1]):
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

            axes[0].plot(
                y_add_test[:, index],
                y_test[:, index],
                "bo",
                ms=0.01,
                linestyle="None",
            )
            axes[1].plot(
                y_p_test[:, index],
                y_test[:, index],
                "ro",
                ms=0.01,
                linestyle="None",
            )
            fig.suptitle(f"g index = {index}")
            axes[0].set_title("simple sum")
            axes[1].set_title("model predictions")

            for ax in axes:
                ax.plot(
                    [y_p_test[:, index].min(), y_p_test[:, index].max()],
                    [y_p_test[:, index].min(), y_p_test[:, index].max()],
                    color="gray",
                    ls="--",
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylabel("true")
                ax.set_xlabel("predicted")
                ax.set_aspect("equal")

            plt.show()

    def plot_weights(self, do_log=True):
        """
        Plot the weights of the model

        Parameters
        ----------
        do_log: bool
            do log scaling in the colorbar
        """
        self._check_trained()
        if isinstance(self.model, keras.Model):
            for i, weights in enumerate(self.model.weights):
                if do_log:
                    if (weights.numpy() < 0).any():
                        vmax = abs(weights.numpy()).max()
                        vmin = -vmax
                        linthr = abs(weights.numpy()).min()
                        # linthr = 1e-1
                        norm = mcolors.SymLogNorm(
                            linthr=linthr, vmin=vmin, vmax=vmax
                        )
                        cmap = "BrBG"
                    else:
                        norm = mcolors.LogNorm()
                        cmap = "viridis"
                else:
                    norm = mcolors.Normalize()
                    cmap = "viridis"

                img = plt.imshow(weights.numpy(), norm=norm, cmap=cmap)
                plt.title(f"weight {i}")

                plt.colorbar(img)
                plt.show()
