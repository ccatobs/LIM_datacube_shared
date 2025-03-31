"""
A collection of functions for modifying data cube

Author: Ankur Dev (adev@astro.uni-bonn.de)
Last Updated: March 2025
"""

def rebin_frequency_axis(tomo_data, bin_size=3):
    """
    Rebins the frequency axis of a 3D data cube by averaging over bin_size slices.

    Parameters:
    data (numpy.ndarray): Input 3D array of shape (Nx, Ny, Nf).
    bin_size (int): Number of frequency slices to average together.

    Returns:
    numpy.ndarray: Re-binned 3D array of shape (Nx, Ny, Nf//bin_size).
    """
    Nx, Ny, Nf = tomo_data.shape

    # Ensure Nf is a multiple of bin_size
    if Nf % bin_size != 0:
        raise ValueError("Number of frequency channels must be a multiple of bin_size")

    # Reshape and average along the last axis
    rebinned_data = tomo_data.reshape(Nx, Ny, Nf // bin_size, bin_size).mean(axis=-1)
    return rebinned_data

def lorentzian_kernel_1d(gamma, x_size=None):
    """
    Create a 1D Lorentzian kernel normalized so that the sum of its elements is 1.
    
    Parameters
    ----------
    gamma : float
        The half-width at half-maximum (HWHM) of the Lorentzian.
    x_size : int, optional
        Length of the kernel array. 
    
    Returns
    -------
    numpy.ndarray
        Normalized 1D Lorentzian kernel.
    """
    # Default kernel size
    if x_size is None:
        x_size = int(np.ceil(8 * gamma))
        if x_size % 2 == 0:
            x_size += 1

    # Create symmetric x positions centered at zero.
    half_size = x_size // 2
    x = np.arange(-half_size, half_size + 1)

    # Evaluate the Lorentzian function
    kernel = (1 / np.pi) * gamma / (x**2 + gamma**2)

    # Normalize the kernel so its sum equals 1.
    kernel /= kernel.sum()
    return kernel

def conv_lorentz1d_spectral(data_cube, fwhm):
    """
    Applies 1D Lorentzian smoothing along the spectral axis (axis=2) in a 3D data cube.

    Parameters:
    data_cube (numpy.ndarray): 3D data array (e.g., shape [RA, DEC, Frequency]).
    fwhm (float): FWHM in spectral channels.

    Returns:
    numpy.ndarray: Spectrally smoothed 3D data cube.
    """
    gamma = fwhm / 2  # Convert FWHM to gamma
    num_freqs = data_cube.shape[2]  # Number of spectral channels
    
    # Create 1D Lorentzian kernel
    kernel = lorentzian_kernel_1d(gamma, x_size=num_freqs)
    print(f' Kernel sum : {kernel.sum():.3f}') 
    # Apply 1D convolution along the spectral axis for each (RA, DEC) pixel
    smoothed_cube = np.apply_along_axis(lambda spectrum: convolve(spectrum, kernel, boundary='extend'),
                                        axis=2, arr=data_cube)
    return rebin_frequency_axis(smoothed_cube)
