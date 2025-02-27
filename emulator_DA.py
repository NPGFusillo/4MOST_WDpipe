import itertools
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from sklearn.decomposition import PCA

# Replacing interpolation of models at fixed grid points with a principle component analysis approach is an idea of Stuart Littlefair (Univeristy of Sheffield). Much of this code is taken from him, please see his page for a complete understanding:https://github.com/StuartLittlefair/wd_emulator


class Emulator_DA:

    def __init__(self, wavelengths, teffs, gravities, spectra):
        """
        Spectral Emulator using PCA.


        Parameters
        ----------
        wavelengths: `np.ndarray`
            wavelengths on which the model grid is calculated
        teffs: `np.ndarray`
            the effective temperatures at which the grid is sampled
        gravities: `np.ndarray`
            the log g values at which the grid is sampled
        spectra: `np.ndarray`
            the model spectra. Should be of shape (nteffs, ngrav, nspec)
        """
        if (teffs.size, gravities.size,wavelengths.size) != spectra.shape:
            raise ValueError('shapes of inputs do not match')

        nteffs, nlogg,nwavs = spectra.shape
        self.model_fluxes = spectra.reshape(nteffs*nlogg, nwavs)
        self.wavelengths = wavelengths

        # a nspectra by 2 array of logg and teff
        log_teff = np.log10(teffs)
        self.x = np.array(list(itertools.product(log_teff, gravities)))
        self.nteffs = nteffs
        self.nlogg = nlogg
        # Normalize to an average of 1 to remove overall brightness changes
        self._normalised_fluxes = (self.model_fluxes)
        self._pca = None
        self._gps = None
        self._weight_interpolator = None

    def run_pca(self, target_variance=0.995, **pca_kwargs):
        """
        Perform PCA on model grid.

        Parameters
        ----------
        target_variance: float
            variance to aim for
        pca_kwargs: dict
            any additional arguments to pass directly to sklearn's PCA
            class
        """
        default_pca_kwargs = dict(n_components=target_variance,
                                  svd_solver="full", whiten=True)
        default_pca_kwargs.update(pca_kwargs)
        self._pca = PCA(**default_pca_kwargs)
        self._pca_weights = self._pca.fit_transform(self._normalised_fluxes)
        self.eigenspectra = self._pca.components_
        self.ncomps = self._pca.n_components_
        exp_var = self._pca.explained_variance_ratio_.sum()

        # save this quantity for later use in reconstructing spectra
        self._X = (np.sqrt(self._pca.explained_variance_[:, np.newaxis]) *
                   self.eigenspectra)
        return exp_var

    
    def get_index(self, params):
        """
        Given a list of stellar parameters (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, grid_points, and weights.

        Parameters
        ----------
        params : array_like
            The stellar parameters

        Returns
        -------
        index : int
        """
        params = np.atleast_2d(params)
        marks = np.abs(self.x - np.expand_dims(params, 1)).sum(axis=-1)
        return marks.argmin(axis=1).squeeze()

    def _predict_weights(self, pars):
        """
        Linear interpolation of weight maps.

        """

        if self._weight_interpolator is None:
            self._weight_interpolator = LinearNDInterpolator(
                self.x, self._pca_weights, rescale=True
            )
        return self._weight_interpolator(pars)

    def __call__(self, pars):
        """
        Emulate the spectrum by interpolating weight maps to find eigenvector weights
        at this temperature and gravity.

        Parameters
        ----------
        params: ~np.ndarray
            A pair of log_10(Teff) and log_g values.

        Returns
        -------
        spectrum: ~np.ndarray
            The interpolated spectrum or spectra.
        """

      
        weights = self._predict_weights(pars)
        spectra = weights @ self._X + self._pca.mean_
        return spectra.squeeze()
