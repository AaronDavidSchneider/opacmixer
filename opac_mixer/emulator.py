import numpy as np
from .mix import CombineOpacIndividual
from scipy.stats import qmc
import xgboost as xg
from sklearn.model_selection import train_test_split

DEFAULT_PRANGE = (1e-6, 1000)
DEFAULT_TRANGE = (100, 10000)

DEFAULT_MMR_RANGES = {
    'CO': (1e-50, 0.005522337070205542),
    'H2O': (3.509581940975492e-22, 0.0057565911404275204),
    'HCN': (1e-50, 9.103077483740115e-05),
    'C2H2,acetylene': (1e-50, 1.581540423097846e-05),
    'CH4': (2.289698067595399e-31, 0.0031631031028604537),
    'PH3': (1e-50, 6.401082202603451e-06),
    'CO2': (1e-50, 0.00015319944152172055),
    'NH3': (3.8119208513224578e-25, 0.00084362326521647),
    'H2S': (2.0093762682408387e-18, 0.0003290905470710346),
    'VO': (1e-50, 1.6153195092178982e-07),
    'TiO': (1e-50, 3.925184850731112e-06),
    'Na': (1e-50, 2.524986071526357e-05),
    'K': (1e-50, 1.932224843084919e-06),
    'SiO': (1e-50, 0.0010448970102509476),
    'FeH': (1e-50, 0.000203477300968298)
}

const_c = 2.99792458e10  # speed of light in cgs


def default_input_scaling(x):
    """Default function used for input scaling"""
    return x


def default_output_scaling(y):
    """Default function used for output scaling"""
    return np.log10(y)


def default_inv_output_scaling(y):
    """Default function used to recover output scaling"""
    return 10**y



class Emulator:
    def __init__(self, opac, prange_opacset=DEFAULT_PRANGE, trange_opacset=DEFAULT_TRANGE):
        """
        Construct the emulator class.

        Parameters
        ----------
        opac: opac_mixer.read.ReadOpac
            the input opacity reader. Can be setup, but does not need to. Will do the setup itself otherwise.
        prange_opacset: (lower, upper)
            (optional): the range to which the reader should interpolate the pressure grid to
        trange_opacset: (lower, upper)
            (optional): the range to which the reader should interpolate the temperature grid to
        """
        self.opac = opac

        if not self.opac.read_done:
            self.opac.read_opac()
        if not self.opac.interp_done:
            self.opac.setup_temp_and_pres(pres=np.logspace(np.log(prange_opacset[0]),np.log(prange_opacset[1]), 100),
                                          temp=np.linspace(*trange_opacset, 100))

        self.mixer = CombineOpacIndividual(self.opac)

        self.input_scaling = default_input_scaling
        self.output_scaling = default_output_scaling
        self.inv_output_scaling = default_inv_output_scaling

        self._has_input = False
        self._has_mix = False
        self._has_model = False
        self._is_trained = False

    def setup_scaling(self, input_scaling = None, output_scaling = None, inv_output_scaling = None):
        """
        (optional) Change the callback functions for the scaling of in and output.
        Defaults are given as opac_mixer.emulator.default_<name>.
        """
        if input_scaling is not None:
            self.input_scaling = input_scaling
        if output_scaling is not None:
            self.output_scaling = output_scaling
        if inv_output_scaling is not None:
            self.inv_output_scaling = inv_output_scaling

    def setup_sampling_grid(self, batchsize=524288, bounds={}, use_sobol=True, filename=None, load=False):
        """
        Setup the sampling grid. Sampling along MMR and pressure is in logspace.
        Sampling along temperature is in linspace.

        Dimension of a sample: (mmr_0, .., mmr_n, p, T)

        Parameters
        ----------
        batchsize: int
            Number of total sampling points. Needs to be a power of 2 for sobol sampling
        bounds: dict
            the lower and upper bounds for sampling. Shape: {'species':(lower, upper)}
            The key can be either a species name in opac.spec or p and T for pressure and Temperature.
            It will use opac_mixer.emulator.DEFAULT_MMR_RANGES for mmrs, opac_mixer.emulator.DEFAULT_PRANGE for pressure,
            and opac_mixer.emulator.DEFAULT_TRANGE for temperautre for all missing values
        use_sobol: bool
            Use sobol sampling. If false, a uniform sampling is used instead.
        filename: str
            A filename, used to save the input data to
        load: bool
            Load the input_data instead of computing it

        Returns
        -------
        input_data: np.array((batchsize, opac.ls+2))
            The sampled inputdata to train/test the emulator.
            Shape: [..., [mmr_{0,i}, .., mmr_{ls,i}, p_i, T_i], ...]
        """
        # make sure the filename comes without the npy suffix
        filename = filename.replace('.npy', '')

        if filename is not None and load:
            input_data = np.load(f"{filename}.npy")
            self._check_input_data(input_data)
            self.batchsize = input_data.shape[0]
            self.input_data = input_data
            self._has_input = True
            return self.input_data

        self.batchsize = batchsize

        l_bounds = []
        u_bounds = []
        for sp in self.opac.spec:
            if sp not in DEFAULT_MMR_RANGES and sp not in bounds:
                raise ValueError(f"We miss the bounds for {sp}.")

            default_l, default_u = DEFAULT_MMR_RANGES.get(sp)
            l, u = bounds.get(sp, (default_l, default_u))
            l_bounds.append(np.maximum(l, 1.0e-50))
            u_bounds.append(u)

        low_p, high_p = bounds.get("p", DEFAULT_PRANGE)
        low_T, high_T = bounds.get("T", DEFAULT_TRANGE)

        l_bounds.extend([low_p, low_T])
        u_bounds.extend([high_p, high_T])

        dim = int(self.opac.ls + 2)

        if use_sobol:
            # Sample along the sobol sequence
            if np.log2(self.batchsize) % 1 != 0:
                raise ValueError("Sobol sampling requires a batchsize given by a power of 2")

            if self.batchsize < 2 ** dim:
                print(
                    f'WARNING: sobol sampling might not have enough data. You should use a batchsize of minimum: {2 ** dim}')

            sampler = qmc.Sobol(d=self.opac.ls + 2, scramble=False)
            sample = sampler.random(self.batchsize)
        else:
            # Use a standard uniform distribution instead
            sample = np.random.uniform(low=0.0, high=1.0, size=(batchsize, dim))

        # Scale the sampling to the actual bounds
        self.input_data = np.empty((batchsize, dim))
        self.input_data[:, :-1] = np.exp(sample[:, :-1] * (np.log(u_bounds)[np.newaxis, :-1]-np.log(l_bounds)[np.newaxis, :-1]) + np.log(l_bounds)[np.newaxis, :-1])
        self.input_data[:, -1] = sample[:, -1] * (u_bounds[-1]-l_bounds[-1]) + l_bounds[-1]

        self._check_input_data(self.input_data)

        if filename is not None:
            np.save(f"{filename}.npy", self.input_data)

        self._has_input = True

        return self.input_data

    def _check_input_data(self, input_data):
        shape = input_data.shape
        if len(shape) != 2 or shape[1] != self.opac.ls + 2:
            raise ValueError('input data does not match')
        assert (input_data >= 0).all(), "We need positive input data!"

    def setup_mix(self, test_size=0.2, filename=None, do_parallel=True, load=False, **split_kwargs):
        """
        Setup the mixer and generate the training and testdata.

        Parameters
        ----------
        test_size: float
            fraction of data used for testing
        filename: str
            A filename, used to save the mixed crosssections to.
        load: bool
            Load the mixed croessections instead of computing them
        """
        if not self._has_input:
            raise AttributeError('we do not have input yet. Run setup_sampling_grid first.')

        # make sure the filename comes without the npy suffix
        filename = filename.replace('.npy', '')

        if not load:
            if do_parallel:
                mix = self.mixer.add_batch_parallel(self.input_data).reshape(
                    (self.batchsize, self.opac.lf[0] * self.opac.lg[0]))
            else:
                mix = self.mixer.add_batch(self.input_data).reshape(
                    (self.batchsize, self.opac.lf[0] * self.opac.lg[0]))
        else:
            mix = np.load(f"{filename}.npy")

        if filename is not None and not load:
            np.save(f"{filename}.npy", mix)

        if (mix <= 0).any():
            raise ValueError('We found negative crosssections. Something is wrong here.')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.input_data,
            mix,
            test_size=test_size,
            **split_kwargs
        )

        self._has_mix = True

        return self.X_train, self.X_test, self.y_train, self.y_test

    def setup_model(self, model=None, filename=None, load=False, **model_kwargs):
        """
        Setup the emulator model and train it.

        Parameters
        ----------
        model: sklearn compatible model
            (optional): a model to learn. Needs to be contructed already. Use XGBRegressor by default
        filename: str or None
            (opyional): A filename to save the model
        load: bool
            (optional): load a -pretrained- model instead of constructing one
        
        
        Parameters for XGBoost
        ----------------------
        Check XGBoost docs for more arguments. Any extra argument is directly passed to XGBoosts
        n_estimators: int
            (optional): number of trees in the ensemble (only when model=None is used)
        max_depth: int
            (optional): maximum depth of each tree in the ensemble (only when model=None is used)
        tree_method: int
            (optional): method to use for training of the trees in the ensemble (only when model=None is used)


        (model_kwargs)
            arguments to pass to XGBRegressor for construction (only when model=None is used)
        """
        if not self._has_mix:
            raise AttributeError('we do not have a mix to work with yet. Run setup_sampling_grid and setup_mix first.')

        if model is None:
            # Use an XGB Regression ensemble
            # Hyperopt
            xgb_params = {
                'max_depth': 8,
                'n_estimators': 50,
                'tree_method': 'hist',
                'eval_metric': 'rmse',
                'early_stopping_rounds': 10,
            }
            xgb_params.update(model_kwargs)
            self.model = xg.XGBRegressor(**xgb_params)

        elif model is not None:
            # Use provided model (needs to be sklearn compatible)
            self.model = model

        if load:
            # Load model
            self.model.load_model(filename)
            self._is_trained = True

        if filename is not None and not load:
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
        if not self._has_model:
            raise AttributeError('we do not have a model yet. Run setup_sampling_grid, setup_mix and setup_model first.')

        # fit the model on the training dataset
        X_train = self.input_scaling(self.X_train)
        X_test = self.input_scaling(self.X_test)
        y_train = self.output_scaling(self.y_train)
        y_test = self.output_scaling(self.y_test)

        if isinstance(self.model, xg.XGBRegressor):
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], *args, **kwargs)
        else:
            self.model.fit(X_train, y_train, *args, **kwargs)

        if hasattr(self, "_model_filename") and callable(getattr(self.model, "save_model", None)):
            print(f"Saving model to {self._model_filename}")
            self.model.save_model(self._model_filename)

        self._is_trained = True

    def predict(self, X, shape='opac', prt_freq=None, *args, **kwargs):
        """
        Train the model.

        Parameters
        ----------
        X: array like (num_samples, input_dim)
            The values you want predictions for
        shape: str
            The output shape for the predictions.
            'opac': same shape as opac reader: (num_samples, lf, lg)
            'same': native flat shape: (num_samples, lf*lg)
            'prt': same as 'opac' but with reverse freq
        prt_freq: array
            Radtrans.freq array, only used for matching frequencies if shape=='prt'.
            Note: This mode also requires self.opac.bin_center to be wavenumbers in cgs
        args:
            Whatever you want to pass to the model for prediction
        kwargs:
            Whatever you want to pass to the model for prediction

        """
        if not self._is_trained:
            raise AttributeError('we do not have a trained model yet. Run setup_sampling_grid, setup_mix and setup_model and fit first.')

        if len(X.shape) != 2 or X.shape[1] != self.opac.ls + 2:
            raise ValueError('input data does not match')

        # fit the model on the training dataset
        y_predict = self.inv_output_scaling(self.model.predict(self.input_scaling(X), *args, **kwargs))
        return self.reshape(y_predict, shape=shape, prt_freq=prt_freq)


    def reshape(self, y, shape = 'opac', prt_freq=None):
        """
        Train the model.

        Parameters
        ----------
        y: array like (num_samples, input_dim)
            The y values you want to be reshaped
        shape: str
            The output shape for the predictions.
            'opac': same shape as opac reader: (num_samples, lf, lg)
            'same': native flat shape: (num_samples, lf*lg)
            'prt': same as 'opac' but with reverse freq
        prt_freq: array
            Radtrans.freq array, only used for matching frequencies if shape=='prt'.
            Note: This mode also requires self.opac.bin_center to be wavenumbers in cgs


        Returns
        -------
        y: reshaped array
        """
        if shape == 'same':
            return y

        y_resh = y.reshape(y.shape[0], self.opac.lf[0], self.opac.lg[0])

        if shape == 'opac':
            return y_resh
        elif shape == 'prt':
            y_prt = np.empty((self.opac.lg[0], self.opac.lf[0], y.shape[0]))
            if prt_freq is not None:
                for freqi, freq in enumerate(self.opac.bin_center):
                    idx = np.abs(prt_freq / const_c - freq).argmin()
                    y_prt[:, idx, :] = y_resh[:, freqi, :].T
            else:
                raise ValueError('We need the frequencies from prt to match the prediction')
            return y_prt
        else:
            raise NotImplementedError('shape not available.')

    def plot_ktable(self):
        # TODO: Some stuff here
        pass

    def calc_score(self):
        # TODO: Some stuff here
        pass
