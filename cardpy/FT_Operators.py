def fft2c(image_space):
    """
    ########## Definition Inputs ##################################################################################################################
    # image_space       : Matrix in image space.
    ########## Definition Outputs #################################################################################################################
    # k_space           : Matrix in k-space.
    """
    ########## Definition Information #############################################################################################################
    ### Originally written by John Pauly in MATLAB
    ### Converted to Python by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import numpy as np                                                      # Import numpy module
    ########## Convert to k-space ##################################################################################################################
    k_space = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image_space)))    # Convert image space data to k-space data
    return k_space

def ifft2c(k_space):
    """
    ########## Definition Inputs ##################################################################################################################
    # k_space           : Matrix in k-space.
    ########## Definition Outputs #################################################################################################################
    # image_space       : Matrix in image space.
    """
    ########## Definition Information #############################################################################################################
    ### Originally written by John Pauly in MATLAB
    ### Converted to Python by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import numpy as np                                                      # Import numpy module
    ########## Convert to image space ##############################################################################################################
    image_space = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_space)))  # Convert k-space data to image space data
    return image_space
