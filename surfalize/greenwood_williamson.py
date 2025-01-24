import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

from .cache import CachedInstance, cache
from .surface import Surface


class GreenwoodWilliamsonModel(CachedInstance):
    """
    Implements the Greenwood and Williamson (GW) model for contact mechanics between rough surfaces.
    
    The GW model represents a rough surface as an ensemble of asperities (peaks) with a 
    Gaussian height distribution and uniform radius. It calculates contact area and load 
    based on the applied pressure.
    
    Parameters
    ----------
    surface : Surface
        Surface object representing the rough surface.
    asperity_radius : float
        Radius of the spherical asperities (µm).
    asperity_density : float, optional
        Number of asperities per unit area (asperities/µm²). If not provided, it is estimated 
        from the surface roughness parameters.
    
    Examples
    --------
    >>> from surfalize import Surface, GreenwoodWilliamsonModel
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, (100, 100))
    >>> surface = Surface(data=data, step_x=0.1, step_y=0.1, width_um=10, height_um=10)
    >>> gw_model = GreenwoodWilliamsonModel(surface, asperity_radius=0.5)
    >>> contact_area = gw_model.contact_area(pressure=100)
    >>> total_load = gw_model.total_load(pressure=100)
    """

    def __init__(self, surface: Surface, asperity_radius: float, asperity_density: float = None):
        super().__init__()
        self.surface = surface
        self.asperity_radius = asperity_radius
        self.asperity_density = asperity_density or self._estimate_asperity_density()
        self.height_std = np.std(self.surface.data)
        self.mean_height = np.mean(self.surface.data)
    
    def _estimate_asperity_density(self):
        """
        Estimates asperity density based on surface roughness parameters.
        Assumes asperity density is proportional to the inverse of asperity radius.
        
        Returns
        -------
        float
            Estimated asperity density (asperities/µm²).
        """
        # Simple estimation: higher roughness leads to higher asperity density
        return (self.height_std / self.asperity_radius) * 0.1  # Scaling factor can be adjusted
    
    @cache
    def asperity_distribution(self):
        """
        Returns the probability density function (PDF) of asperity heights.
        
        Returns
        -------
        function
            Gaussian PDF of asperity heights.
        """
        return norm(loc=self.mean_height, scale=self.height_std)
    
    @cache
    def contact_probability(self, pressure: float):
        """
        Calculates the probability that an asperity is in contact under the given pressure.
        
        Parameters
        ----------
        pressure : float
            Applied pressure (µN/µm²).
        
        Returns
        -------
        float
            Probability of contact (0 <= P <= 1).
        """
        # Assuming linear elastic contact: contact occurs if asperity height <= pressure * radius
        contact_height = pressure * self.asperity_radius
        pdf = self.asperity_distribution()
        return pdf.cdf(contact_height)
    
    @cache
    def contact_area_per_asperity(self, pressure: float):
        """
        Calculates the contact area for a single asperity under the given pressure.
        
        Parameters
        ----------
        pressure : float
            Applied pressure (µN/µm²).
        
        Returns
        -------
        float
            Contact area per asperity (µm²).
        """
        if pressure <= 0:
            return 0.0
        contact_height = pressure * self.asperity_radius
        if contact_height <= 0:
            return 0.0
        # Contact area of a spherical cap: A = 2 * π * r * h
        return 2 * np.pi * self.asperity_radius * contact_height
    
    def contact_area(self, pressure: float):
        """
        Calculates the total contact area under the given pressure.
        
        Parameters
        ----------
        pressure : float
            Applied pressure (µN/µm²).
        
        Returns
        -------
        float
            Total contact area (µm²).
        """
        P = self.contact_probability(pressure)
        A = self.contact_area_per_asperity(pressure)
        return self.asperity_density * P * A
    
    def total_load(self, pressure: float):
        """
        Calculates the total load carried by the contact area under the given pressure.
        
        Parameters
        ----------
        pressure : float
            Applied pressure (µN/µm²).
        
        Returns
        -------
        float
            Total load (µN).
        """
        return self.contact_area(pressure) * pressure
    
    def plot_contact_probability(self, pressures, ax=None):
        """
        Plots the contact probability as a function of applied pressure.
        
        Parameters
        ----------
        pressures : array-like
            Array of pressure values to evaluate.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. Creates a new one if None.
        
        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            Figure and Axes objects of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        probabilities = [self.contact_probability(p) for p in pressures]
        ax.plot(pressures, probabilities, label='Contact Probability')
        ax.set_xlabel('Pressure (µN/µm²)')
        ax.set_ylabel('Probability of Contact')
        ax.set_title('Contact Probability vs Pressure')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_contact_area(self, pressures, ax=None):
        """
        Plots the total contact area as a function of applied pressure.
        
        Parameters
        ----------
        pressures : array-like
            Array of pressure values to evaluate.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. Creates a new one if None.
        
        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            Figure and Axes objects of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        contact_areas = [self.contact_area(p) for p in pressures]
        ax.plot(pressures, contact_areas, label='Total Contact Area', color='green')
        ax.set_xlabel('Pressure (µN/µm²)')
        ax.set_ylabel('Contact Area (µm²)')
        ax.set_title('Total Contact Area vs Pressure')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig, ax
