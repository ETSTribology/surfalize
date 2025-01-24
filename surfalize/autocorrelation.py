import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cache import CachedInstance, cache
from .mathutils import interpolate_line_on_2d_array, argmin_all, argmax_all, argclosest


class AutocorrelationFunction(CachedInstance):
    """
    Computes the 2D autocorrelation function of a Surface object and calculates 
    autocorrelation length (Sal) and texture aspect ratio (Str).

    Parameters
    ----------
    surface : Surface
        The Surface object to analyze.
    """

    def __init__(self, surface):
        super().__init__()
        self.surface = surface.center()
        self.data = self.calculate_autocorrelation()
        self.center = np.array(self.data.shape) // 2

    def calculate_autocorrelation(self):
        """Calculates the 2D autocorrelation function using FFT."""
        data_fft = np.fft.fft2(self.surface.data)
        acf = np.fft.ifft2(data_fft * np.conj(data_fft)).real
        acf /= self.surface.data.size
        return np.fft.fftshift(acf)

    @cache
    def _calculate_decay_lengths(self, threshold_fraction):
        """
        Determines the shortest and longest decay lengths based on a threshold.

        Parameters
        ----------
        threshold_fraction : float
            Fraction of the maximum ACF value to set as the threshold.

        Returns
        -------
        tuple of float
            (shortest_decay_length, longest_decay_length)
        """
        threshold = threshold_fraction * self.data.max()
        mask = self.data > threshold
        labeled, num_features = ndimage.label(mask)
        region = labeled == labeled[self.center[0], self.center[1]]
        edge = region ^ ndimage.binary_dilation(region)

        if not np.any(edge):
            return 0.0, 0.0

        idx_edge = np.argwhere(edge)
        distances = np.linalg.norm((idx_edge - self.center) * np.array([self.surface.step_y, self.surface.step_x]), axis=1)

        min_idx = idx_edge[argmin_all(distances)]
        max_idx = idx_edge[argmax_all(distances)]

        decay_lengths = self._interpolate_decay_length(min_idx, threshold), self._interpolate_decay_length(max_idx, threshold)
        return decay_lengths

    def _interpolate_decay_length(self, idx, threshold):
        """Interpolates the decay length to the threshold value."""
        length = np.hypot(*(idx - self.center) * np.array([self.surface.step_y, self.surface.step_x]))
        n_points = 1000
        interpolated_values = interpolate_line_on_2d_array(self.data, self.center, idx, num_points=n_points)
        interpolated_lengths = np.linspace(0, length, n_points)
        closest_idx = argclosest(threshold, interpolated_values)
        return interpolated_lengths[closest_idx] if closest_idx < len(interpolated_lengths) else length

    @cache
    def Sal(self, s=0.2):
        """
        Calculates the autocorrelation length Sal.

        Parameters
        ----------
        s : float, optional
            Threshold fraction (default is 0.2).

        Returns
        -------
        float
            Autocorrelation length Sal.
        """
        Sal, _ = self._calculate_decay_lengths(s)
        return Sal

    @cache
    def Str(self, s=0.2):
        """
        Calculates the texture aspect ratio Str.

        Parameters
        ----------
        s : float, optional
            Threshold fraction (default is 0.2).

        Returns
        -------
        float
            Texture aspect ratio Str.
        """
        Sal, Sll = self._calculate_decay_lengths(s)
        return Sal / Sll if Sll != 0 else 0.0

    def plot_autocorrelation(self, ax=None, cmap='jet', show_cbar=True):
        """
        Plots the autocorrelation function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. Creates a new one if None.
        cmap : str, optional
            Colormap for the plot (default is 'jet').
        show_cbar : bool, optional
            Whether to display the colorbar (default is True).

        Returns
        -------
        tuple
            (Figure, Axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        im = ax.imshow(
            self.data,
            cmap=cmap,
            extent=(0, self.surface.width_um, 0, self.surface.height_um)
        )

        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, label='ACF [µm²]')
        else:
            ax.figure.colorbar(im, ax=ax, label='ACF [µm²]')

        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
        ax.set_title('Autocorrelation Function')
        plt.tight_layout()

        return fig, ax
