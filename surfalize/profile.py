import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import euclidean, cityblock, minkowski, cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

class Profile:
    """
    Represents a 1D surface profile and provides methods to analyze surface roughness parameters.

    Parameters
    ----------
    height_data : array-like
        1D array of height measurements.
    step : float
        Spatial step between measurements (e.g., micrometers per step).
    length_um : float
        Total length of the profile in micrometers.
    preprocess : bool, default False
        If True, applies Savitzky-Golay smoothing to the data before analysis.
    sg_window : int, default 9
        Window length for Savitzky-Golay filter (must be odd).
    sg_poly : int, default 3
        Polynomial order for Savitzky-Golay filter.

    Examples
    --------
    >>> import numpy as np
    >>> from surfalize import Profile
    >>> height = np.random.normal(0, 1, 1000)
    >>> profile = Profile(height, step=0.1, length_um=100.0, preprocess=True)
    >>> print(profile.Ra())
    0.998
    >>> profile.show()
    """

    def __init__(self, height_data, step, length_um, preprocess=False, sg_window=9, sg_poly=3):
        self._data = np.array(height_data, dtype=float)
        self._step = step
        self._length_um = length_um

        if preprocess:
            if sg_window % 2 == 0:
                sg_window += 1  # Ensure window length is odd
            if sg_window > len(self._data):
                sg_window = len(self._data) if len(self._data) % 2 != 0 else len(self._data) - 1
            self._data = savgol_filter(self._data, window_length=sg_window, polyorder=sg_poly)

    def __repr__(self):
        return f'{self.__class__.__name__}(Length: {self._length_um:.2f} µm)'

    def _repr_png_(self):
        """
        Repr method for Jupyter notebooks. When Jupyter makes a call to repr, it checks first if a _repr_png_ is
        defined. If not, it falls back on __repr__.
        """
        self.show()

    def period(self):
        """
        Estimates the dominant spatial period of the profile using FFT.

        Returns
        -------
        period : float
            Dominant period in micrometers.

        Raises
        ------
        ValueError
            If the profile data is flat or insufficient for period estimation.
        """
        fft_vals = np.abs(fft(self._data))
        freq = fftfreq(len(self._data), d=self._step)

        # Consider only positive frequencies
        pos_mask = freq > 0
        fft_vals = fft_vals[pos_mask]
        freq = freq[pos_mask]

        if len(fft_vals) == 0:
            raise ValueError("Insufficient data for period estimation.")

        peaks, properties = find_peaks(fft_vals, distance=10, prominence=np.max(fft_vals)*0.1)
        if len(peaks) == 0:
            raise ValueError("No significant peaks found in FFT for period estimation.")

        dominant_freq = freq[peaks[np.argmax(properties['prominences'])]]
        period = 1 / dominant_freq
        return period

    def Ra(self):
        """
        Calculates the arithmetic mean height (Ra).

        Returns
        -------
        Ra : float
            Arithmetic mean height.
        """
        return np.mean(np.abs(self._data - np.mean(self._data)))

    def Rq(self):
        """
        Calculates the root mean square height (Rq).

        Returns
        -------
        Rq : float
            Root mean square height.
        """
        return np.sqrt(np.mean((self._data - np.mean(self._data)) ** 2))

    def Rp(self):
        """
        Calculates the maximum peak height (Rp).

        Returns
        -------
        Rp : float
            Maximum peak height.
        """
        return np.max(self._data - np.mean(self._data))

    def Rv(self):
        """
        Calculates the maximum valley depth (Rv).

        Returns
        -------
        Rv : float
            Maximum valley depth.
        """
        return np.abs(np.min(self._data - np.mean(self._data)))

    def Rz(self):
        """
        Calculates the ten-point height range (Rz).

        Returns
        -------
        Rz : float
            Ten-point height range.
        """
        return self.Rp() + self.Rv()

    def Rsk(self):
        """
        Calculates the skewness (Rsk) of the profile.

        Returns
        -------
        Rsk : float
            Skewness.
        """
        variance = np.var(self._data)
        if variance < 1e-20:
            return 0.0
        return skew(self._data)

    def Rku(self):
        """
        Calculates the kurtosis (Rku) of the profile.

        Returns
        -------
        Rku : float
            Kurtosis.
        """
        variance = np.var(self._data)
        if variance < 1e-20:
            return 0.0
        return kurtosis(self._data, fisher=False)

    def RSm(self):
        """
        Estimates the mean spacing between peaks (RSm) using continuous wavelet transform.

        Returns
        -------
        RSm : float
            Mean spacing between peaks in micrometers.
        """
        widths = np.arange(1, 20)
        peaks = find_peaks_cwt(np.abs(self._data), widths=widths)
        if len(peaks) > 1:
            spacings = np.diff(peaks) * self._step
            return np.mean(spacings)
        else:
            return 0.0

    def Rdq(self):
        """
        Calculates the root mean square slope (Rdq) of the profile.

        Returns
        -------
        Rdq : float
            Root mean square slope.
        """
        slopes = np.diff(self._data) / self._step
        return np.sqrt(np.mean(slopes ** 2))

    def show(self, annotate_peaks=True):
        """
        Plots the profile with optional peak and valley annotations.

        Parameters
        ----------
        annotate_peaks : bool, default True
            If True, marks peaks and valleys on the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.linspace(0, self._length_um, self._data.size)
        ax.plot(x, self._data, color='k', lw=1, label='Profile')
        ax.set_xlabel('Position [µm]')
        ax.set_ylabel('Height [µm]')
        ax.set_title('1D Surface Profile')

        if annotate_peaks:
            peaks, _ = find_peaks(self._data, prominence=0.05 * np.max(self._data))
            valleys, _ = find_peaks(-self._data, prominence=0.05 * np.abs(np.min(self._data)))
            ax.plot(x[peaks], self._data[peaks], 'ro', label='Peaks')
            ax.plot(x[valleys], self._data[valleys], 'bo', label='Valleys')
            for peak in peaks:
                ax.annotate(f'{self._data[peak]:.2f}', (x[peak], self._data[peak]),
                            textcoords="offset points", xytext=(0,10), ha='center', color='red')
            for valley in valleys:
                ax.annotate(f'{self._data[valley]:.2f}', (x[valley], self._data[valley]),
                            textcoords="offset points", xytext=(0,-15), ha='center', color='blue')

        ax.legend()
        plt.tight_layout()
        plt.show()

    def apply_filter(self, filter_obj, inplace=False):
        """
        Applies a filter to the profile data.

        Parameters
        ----------
        filter_obj : callable
            A filter object with an `apply` method or callable that takes a Profile instance.
        inplace : bool, default False
            If True, modifies the current Profile instance. Otherwise, returns a new filtered Profile.

        Returns
        -------
        filtered_profile : Profile or None
            Returns a new Profile instance if `inplace=False`. Returns `None` if `inplace=True`.
        """
        if hasattr(filter_obj, 'apply'):
            filtered = filter_obj.apply(self, inplace=inplace)
            return filtered
        elif callable(filter_obj):
            filtered = filter_obj(self)
            return filtered
        else:
            raise TypeError("filter_obj must have an 'apply' method or be callable.")

    def calculate_material_ratio(self, percentile_low=10, percentile_high=80):
        """
        Calculates the material ratio at specified percentiles.

        Parameters
        ----------
        percentile_low : float, default 10
            Lower percentile for material ratio.
        percentile_high : float, default 80
            Upper percentile for material ratio.

        Returns
        -------
        material_ratio_low : float
            Height at the lower percentile.
        material_ratio_high : float
            Height at the upper percentile.
        """
        sorted_data = np.sort(self._data)
        mr_low = np.percentile(sorted_data, percentile_low)
        mr_high = np.percentile(sorted_data, percentile_high)
        return mr_low, mr_high

     def plot_radar(self, param_list=None, title="1D Surface Parameters Radar"):
        """
        Creates a radar plot to visualize surface roughness parameters.

        Parameters
        ----------
        param_list : list of str, optional
            List of parameters to include in the radar plot. If None, defaults to all available parameters.
        title : str, default "1D Surface Parameters Radar"
            Title of the radar plot.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive radar plot.
        """
        # Default parameters to include in the radar plot
        if param_list is None:
            param_list = ['Ra', 'Rq', 'Rp', 'Rv', 'Rz', 'Rsk', 'Rku', 'RSm', 'Rdq']

        # Get parameter values
        params = self.to_dict()
        values = [params.get(p, 0) for p in param_list]

        # Ensure the radar plot is closed by repeating the first value at the end
        values += [values[0]]
        param_list += [param_list[0]]

        # Create radar plot
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=param_list,
            fill='toself',
            name='Profile Parameters',
            line=dict(color='blue')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1] if max(values) > 0 else [0, 1]
                )
            ),
            title=title,
            showlegend=True,
            height=600
        )

        return fig

    def compare_profiles(self, other_profile, title="Profile Comparison"):
        """
        Compare two profiles using similarity metrics and generate visualizations.

        Parameters
        ----------
        other_profile : Profile
            Another Profile instance to compare with.
        title : str, default "Profile Comparison"
            Title for the comparison plots.

        Returns
        -------
        scores : dict
            Dictionary of similarity scores.
        """
        # Extract height data from both profiles
        profile1 = self._data
        profile2 = other_profile._data

        # Ensure both profiles have the same length
        if len(profile1) != len(profile2):
            raise ValueError("Profiles must have the same length for comparison.")

        # Calculate similarity metrics
        scores = self._calculate_similarity_scores(profile1, profile2)

        # Generate comparison plots
        self._plot_comparison(profile1, profile2, scores, title)

        return scores

    def _calculate_similarity_scores(self, profile1, profile2):
            """
            Computes various similarity metrics between two profiles.
    
            Parameters
            ----------
            profile1 : array-like
                First profile's height data.
            profile2 : array-like
                Second profile's height data.
    
            Returns
            -------
            scores : dict
                Dictionary containing similarity metrics.
            """
            scores = {}
            scores['MSE'] = mean_squared_error(profile1, profile2)
            scores['RMSE'] = np.sqrt(scores['MSE'])
            scores['MAE'] = mean_absolute_error(profile1, profile2)
            scores['R2'] = r2_score(profile1, profile2)
            scores['Euclidean'] = euclidean(profile1, profile2)
            scores['Manhattan'] = cityblock(profile1, profile2)
            scores['Minkowski_p3'] = minkowski(profile1, profile2, p=3)
            cos_sim = cosine_similarity([profile1], [profile2])[0][0]
            scores['Cosine_Similarity'] = cos_sim
            return scores

    def _plot_comparison(self, profile1, profile2, scores, title):
        """
        Generates comparison plots for two profiles.

        Parameters
        ----------
        profile1 : array-like
            First profile's height data.
        profile2 : array-like
            Second profile's height data.
        scores : dict
            Dictionary of similarity scores.
        title : str
            Title for the plots.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        x_self = np.linspace(0, self._length_um, len(profile1))
        # Plot profiles
        axes[0].plot(x_self, profile1, label='Current Profile')
        axes[0].plot(x_self, profile2, label='Other Profile', alpha=0.7)
        axes[0].set_xlabel('Position [µm]')
        axes[0].set_ylabel('Height [µm]')
        axes[0].legend()
        axes[0].set_title('Profile Height Comparison')
        # Plot scores
        metric_names = list(scores.keys())
        values = list(scores.values())
        axes[1].bar(metric_names, values)
        axes[1].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1].set_ylabel('Score Value')
        axes[1].set_title('Similarity Metrics')
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

    def to_dict(self):
        """
        Compiles all calculated parameters into a dictionary.

        Returns
        -------
        params : dict
            Dictionary of all calculated roughness parameters.
        """
        return {
            'Ra': self.Ra(),
            'Rq': self.Rq(),
            'Rp': self.Rp(),
            'Rv': self.Rv(),
            'Rz': self.Rz(),
            'Rsk': self.Rsk(),
            'Rku': self.Rku(),
            'RSm': self.RSm(),
            'Rdq': self.Rdq(),
            'Period': self.period(),
            'MaterialRatio_10': self.calculate_material_ratio()[0],
            'MaterialRatio_80': self.calculate_material_ratio()[1],
        }
