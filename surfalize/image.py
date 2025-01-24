import PIL.Image
import numpy as np
from typing import Union, Optional

class Image:
    """
    Enhanced wrapper for image data with advanced processing and validation.
    
    Features:
    - Automatic dtype conversion and normalization
    - Color space management
    - Basic image processing operations
    - Enhanced display and metadata handling
    
    Parameters
    ----------
    data : np.ndarray
        Image data in (H, W) grayscale or (H, W, C) multi-channel format
    mode : Optional[str]
        Force color mode ('L', 'RGB', 'RGBA'). Auto-detected if None
    """
    
    VALID_MODES = {'L', 'RGB', 'RGBA'}
    SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32, np.float64)
    
    def __init__(self, data: np.ndarray, mode: Optional[str] = None):
        self._validate_input(data)
        self._data = self._normalize_array(data, mode)
        
    def _validate_input(self, data):
        """Ensure input meets image requirements"""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
            
        if data.ndim not in (2, 3):
            raise ValueError(f"Invalid shape {data.shape}. Must be 2D or 3D array")
            
        if data.dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {data.dtype}. Supported: {self.SUPPORTED_DTYPES}")

    def _normalize_array(self, data: np.ndarray, mode: Optional[str]) -> np.ndarray:
        """Convert array to standard form (H, W, C) with uint8/uint16 dtype"""
        # Squeeze singleton dimensions
        if data.ndim == 3 and data.shape[2] == 1:
            data = data.squeeze(axis=2)
            
        # Handle dtype conversion
        if np.issubdtype(data.dtype, np.floating):
            data = self._convert_float_to_int(data)
            
        # Add channel dimension for 2D arrays
        if data.ndim == 2:
            data = data[..., np.newaxis]
            
        # Validate/Set color mode
        return self._set_color_mode(data, mode)

    def _convert_float_to_int(self, data: np.ndarray) -> np.ndarray:
        """Convert float array (0-1 range) to uint8/uint16"""
        if data.max() <= 1.0:
            return (data * 255).astype(np.uint8)
        return data.astype(np.uint16)

    def _set_color_mode(self, data: np.ndarray, mode: Optional[str]) -> np.ndarray:
        """Validate or convert color mode"""
        channels = data.shape[2]
        mode_map = {1: 'L', 3: 'RGB', 4: 'RGBA'}
        
        if mode:
            if mode not in self.VALID_MODES:
                raise ValueError(f"Invalid mode {mode}. Valid options: {self.VALID_MODES}")
            expected_channels = 1 if mode == 'L' else len(mode)
            if data.shape[2] != expected_channels:
                raise ValueError(f"Mode {mode} requires {expected_channels} channels, got {data.shape[2]}")
            return data
            
        if channels not in mode_map:
            raise ValueError(f"Unsupported number of channels: {channels}")
        return data

    @property
    def mode(self) -> str:
        """Return color mode string"""
        channels = self._data.shape[2]
        return {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(channels, 'UNKNOWN')
    
    @property
    def shape(self) -> tuple:
        """Return (height, width, channels) tuple"""
        return self._data.shape
    
    @property
    def dtype(self) -> type:
        """Return numpy dtype of image data"""
        return self._data.dtype
    
    def __repr__(self) -> str:
        return (f"Image({self.shape[1]}x{self.shape[0]} {self.mode}, "
                f"dtype: {self.dtype.__name__}, "
                f"range: [{self._data.min()}, {self._data.max()}])")

    def _repr_png_(self):
        """IPython display integration"""
        return self.show()._repr_png_()

    def save(self, path: Union[str, Path], **kwargs):
        """
        Save image to disk with automatic format detection.
        
        Parameters
        ----------
        path : str | Path
            Output file path (extension determines format)
        kwargs : dict
            Additional parameters for PIL.Image.save
        """
        pil_image = self._to_pil_image()
        pil_image.save(path, **kwargs)
        
    def show(self) -> PIL.Image.Image:
        """Return display-ready PIL Image"""
        return self._to_pil_image()

    def _to_pil_image(self) -> PIL.Image.Image:
        """Convert internal array to PIL Image"""
        data = self._data.squeeze() if self.mode == 'L' else self._data
        return PIL.Image.fromarray(data, mode=self.mode)

    def to_grayscale(self, method: str = 'luminosity') -> 'Image':
        """
        Convert to grayscale using specified method.
        
        Parameters
        ----------
        method : str
            Conversion method ('luminosity', 'average', 'value')
            
        Returns
        -------
        Image
            New grayscale image instance
        """
        if self.mode == 'L':
            return self.copy()
            
        weights = {
            'luminosity': [0.2126, 0.7152, 0.0722],
            'average': [1/3, 1/3, 1/3],
            'value': [0, 0, 1]
        }.get(method.lower(), [0.2126, 0.7152, 0.0722])
        
        gray = np.dot(self._data[..., :3], weights)
        return Image(gray.astype(self.dtype), mode='L')

    def resize(self, size: tuple, resample: int = PIL.Image.BILINEAR) -> 'Image':
        """
        Resize image to specified dimensions.
        
        Parameters
        ----------
        size : tuple (width, height)
            Target dimensions
        resample : int
            PIL resampling filter (default: BILINEAR)
            
        Returns
        -------
        Image
            Resized image instance
        """
        pil_image = self._to_pil_image()
        resized = pil_image.resize(size, resample=resample)
        return Image(np.array(resized), mode=self.mode)

    def adjust_contrast(self, factor: float) -> 'Image':
        """
        Adjust image contrast.
        
        Parameters
        ----------
        factor : float
            Contrast adjustment factor (0.0 = solid gray, 1.0 = original)
            
        Returns
        -------
        Image
            Contrast-adjusted image
        """
        if factor < 0:
            raise ValueError("Contrast factor must be >= 0")
            
        mean = np.mean(self._data) if self.mode == 'L' else np.mean(self._data, axis=(0,1))
        adjusted = mean + factor * (self._data - mean)
        return Image(np.clip(adjusted, 0, 255).astype(self.dtype), mode=self.mode)

    def histogram(self, bins: int = 256) -> dict:
        """
        Calculate image histogram.
        
        Parameters
        ----------
        bins : int
            Number of histogram bins
            
        Returns
        -------
        dict
            Channel-wise histogram data
        """
        hist = {}
        for i in range(self._data.shape[2]):
            channel = self._data[..., i]
            counts, edges = np.histogram(channel, bins=bins, range=(0, 255))
            hist[f'channel_{i}'] = {'counts': counts, 'bins': edges}
        return hist

    def copy(self) -> 'Image':
        """Return a deep copy of the image"""
        return Image(self._data.copy(), mode=self.mode)

    @classmethod
    def from_pil(cls, image: PIL.Image.Image) -> 'Image':
        """Create Image instance from PIL Image"""
        return cls(np.array(image), mode=image.mode)

    @staticmethod
    def is_grayscale(array: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if array represents grayscale image with optional tolerance.
        
        Parameters
        ----------
        array : np.ndarray
            Input image array
        tolerance : float
            Allowed difference between channels
            
        Returns
        -------
        bool
            True if all channels are identical within tolerance
        """
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            return False
            
        return np.allclose(array[..., 0], array[..., 1], atol=tolerance) and \
               np.allclose(array[..., 0], array[..., 2], atol=tolerance)
