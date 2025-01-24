import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .mathutils import argclosest, interp1d, trapezoid
from .cache import CachedInstance, cache


class AbbottFirestoneCurve(CachedInstance):
    """
    Computes the Abbott-Firestone curve of a Surface object and derives functional roughness parameters.
    
    Parameters
    ----------
    surface : Surface
        Surface object from which to calculate the Abbott-Firestone curve.
    nbins : int, default 10000
        Number of bins for the material density histogram. Higher values increase accuracy at the cost of computation time.
    """
    
    EQUIVALENCE_LINE_WIDTH = 40  # Width in % as defined by ISO 25178-2

    def __init__(self, surface, nbins=10000):
        super().__init__()
        self.surface = surface
        self.nbins = nbins
        self.height, self.material_ratio = self._get_material_ratio_curve()
        self.slope, self.intercept, self.y_upper, self.y_lower = self._calculate_equivalence_line()
    
    @cache
    def _get_material_ratio_curve(self):
        """Computes height bins and cumulative material ratio."""
        hist, height = np.histogram(self.surface.data, bins=self.nbins)
        hist = hist[::-1]
        height = height[::-1]
        material_ratio = np.concatenate(([100], np.cumsum(hist) / hist.sum() * 100))
        return height, material_ratio

    @cache
    def _calculate_equivalence_line(self):
        """Finds the equivalence line slope and intercept based on the specified width."""
        smc_fit = interp1d(self.material_ratio, self.height)
        # Find slope_min by iterating until the slope is maximized
        max_slope = -np.inf
        istart_final = 0
        for istart in range(len(self.material_ratio)):
            if self.material_ratio[istart] > 100 - self.EQUIVALENCE_LINE_WIDTH:
                break
            mr_shifted = self.material_ratio[istart] + self.EQUIVALENCE_LINE_WIDTH
            if mr_shifted > 100:
                continue
            slope = (smc_fit(mr_shifted) - self.height[istart]) / self.EQUIVALENCE_LINE_WIDTH
            if slope > max_slope:
                max_slope = slope
                istart_final = istart
        
        intercept = self.height[istart_final] - max_slope * self.material_ratio[istart_final]
        y_upper = intercept
        y_lower = max_slope * 100 + intercept
        return max_slope, intercept, y_upper, y_lower

    @cache
    def Sk(self):
        """Calculates Sk."""
        return self.y_upper - self.y_lower

    def Smr(self, c):
        """Calculates Smr(c)."""
        return float(interp1d(self.height, self.material_ratio)(c))

    def Smc(self, mr):
        """Calculates Smc(mr)."""
        return float(interp1d(self.material_ratio, self.height)(mr))

    @cache
    def Smr1(self):
        """Calculates Smr1."""
        return self.Smr(self.y_upper)

    @cache
    def Smr2(self):
        """Calculates Smr2."""
        return self.Smr(self.y_lower)

    @cache
    def Spk(self):
        """Calculates Spk."""
        idx = argclosest(self.y_upper, self.height)
        area = trapezoid(self.material_ratio[:idx], self.height[:idx])
        return 2 * np.abs(area) / self.Smr1()

    @cache
    def Svk(self):
        """Calculates Svk."""
        idx = argclosest(self.y_lower, self.height)
        area = trapezoid(100 - self.material_ratio[idx:], self.height[idx:])
        return 2 * np.abs(area) / (100 - self.Smr2())

    @cache
    def Vmp(self, p=10):
        """Calculates Vmp(p)."""
        c = self.Smc(p)
        idx = argclosest(c, self.height)
        return np.abs(trapezoid(self.material_ratio[:idx], self.height[:idx]) / 100)

    @cache
    def Vmc(self, p=10, q=80):
        """Calculates Vmc(p, q)."""
        return np.abs(trapezoid(self.material_ratio[:argclosest(self.Smc(q), self.height)], self.height[:argclosest(self.Smc(q), self.height)])) / 100 - self.Vmp(p)

    @cache
    def Vvv(self, q=80):
        """Calculates Vvv(q)."""
        idx = argclosest(self.Smc(q), self.height)
        return np.abs(trapezoid(100 - self.material_ratio[idx:], self.height[idx:])) / 100

    @cache
    def Vvc(self, p=10, q=80):
        """Calculates Vvc(p, q)."""
        idx = argclosest(self.Smc(p), self.height)
        return np.abs(trapezoid(100 - self.material_ratio[idx:], self.height[idx:])) / 100 - self.Vvv(q)

    def plot(self, nbars=20, ax=None):
        """Plots the Abbott-Firestone curve and histogram."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        # Histogram
        hist, bins = np.histogram(self.surface.data, bins=nbars)
        hist = hist[::-1]
        bins = bins[::-1]
        ax.barh(bins[:-1] + np.diff(bins)/2, hist / hist.sum() * 100, height=(self.surface.data.max() - self.surface.data.min()) / nbars, edgecolor='k', color='lightblue', label='Histogram')
        
        # Abbott-Firestone curve
        ax2 = ax.twiny()
        ax2.plot(self.material_ratio, self.height, color='r', label='Abbott-Firestone Curve')
        ax2.set_xlabel('Material Ratio (%)')
        
        # Labels and Legend
        ax.set_xlabel('Height (µm)')
        ax.set_ylabel('Material Distribution (%)')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        
        return fig, (ax, ax2)

    def plot_parameter_study(self, ax=None):
        """Visualizes the parameter study with equivalence lines and functional parameters."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(self.height.min(), self.height.max())
        
        # Plot equivalence line
        x = np.linspace(0, 100, 100)
        ax.plot(x, self.slope * x + self.intercept, color='k', label='Equivalence Line')
        
        # Plot polygons for Spk and Svk
        ax.fill_betweenx([self.y_upper, self.y_upper + self.Spk()], 0, self.Smr1(), color='orange', alpha=0.5, label='Spk Area')
        ax.fill_betweenx([self.y_lower, self.y_lower - self.Svk()], 100, self.Smr2(), color='orange', alpha=0.5, label='Svk Area')
        
        # Plot Abbott-Firestone curve
        ax.plot(self.material_ratio, self.height, color='r', label='Abbott-Firestone Curve')
        
        # Horizontal lines for y_upper and y_lower
        ax.axhline(self.y_upper, color='k', linestyle='--')
        ax.axhline(self.y_lower, color='k', linestyle='--')
        
        # Vertical lines for Smr1 and Smr2
        ax.axvline(self.Smr1(), color='k', linestyle=':')
        ax.axvline(self.Smr2(), color='k', linestyle=':')
        
        # Labels and Legend
        ax.set_xlabel('Material Ratio (%)')
        ax.set_ylabel('Height (µm)')
        ax.legend()
        plt.tight_layout()
        
        return fig, ax
