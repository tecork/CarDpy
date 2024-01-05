def DWI_recon(original_matrix, original_bvals, original_bvecs):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values.
    original_bvecs          : Sorted b-vectors.
    ########## Definition Outputs #################################################################################################################
    Standard_DWI_Metrics    : Python dictionary containing all of the standard DWI quantitative metrics, which can be seen directly below.
                              - Apparent diffusion coefficient (ADC) [scaled to mm^2 / μs]
                              - Trace (TR) [scaled to mm^2 / μs]
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    import numpy                         as np                                                                                  # Import numpy module
    from   cardpy.Data_Sorting           import sorted2stacked, stacked2sorted                                                  # Import sorted to stacked and stacked to sorted from CarDpy
    import warnings                                                                                                             # Import warning module
    ########## Address data type of original data #################################################################################################
    if original_matrix.dtype == 'complex128':                                                                                   # If data type is complex ...
            original_matrix = np.abs(original_matrix)                                                                               # Convert data to magnitude
    ########## Convert sorted data into stacked data ##############################################################################################
    [stacked_matrix, stacked_bvals, stacked_bvecs] = sorted2stacked(original_matrix, original_bvals, original_bvecs)            # Convert sorted data into stacked data
    ########## Detection of different b-values ####################################################################################################
    (unique, index, counts) = np.unique(stacked_bvals, return_counts = True, return_index = True, axis = 0)                     # Extract number of different diffusion directions
    numBvals                = len(unique)                                                                                       # Identify number of b-values in stacked b-values
    bval_low                = min(unique)                                                                                       # Identify lower b-value (typically b = 0)
    bval_low_indicies       = np.where(stacked_bvals == bval_low)[0]                                                            # Identify index of lower b-value in stacked b-values
    bval_high               = max(unique)                                                                                       # Identify higher b-value
    bval_high_indicies      = np.where(stacked_bvals == bval_high)[0]                                                           # Identify indices of higher b-values in stacked b-values
    ########## Check orthogonality of b-vectors ###################################################################################################
    Orthogonality_Check_1 = np.dot(stacked_bvecs[bval_high_indicies[0]], stacked_bvecs[bval_high_indicies[1]])                  # Dot product of b-vector 1 and b-vector 2
    Orthogonality_Check_2 = np.dot(stacked_bvecs[bval_high_indicies[0]], stacked_bvecs[bval_high_indicies[2]])                  # Dot product of b-vector 1 and b-vector 3
    Orthogonality_Check_3 = np.dot(stacked_bvecs[bval_high_indicies[1]], stacked_bvecs[bval_high_indicies[2]])                  # Dot product of b-vector 2 and b-vector 3
    if (np.round(Orthogonality_Check_1, 2) == np.round(Orthogonality_Check_2, 2) == np.round(Orthogonality_Check_3, 2)):        # If all diffusion directions are orthogonal ...
        pass                                                                                                                        # Continue on with calculations
    else:                                                                                                                       # Otherwise ...
        warnings.warn('Directionally-specific diffusions are not orthogonal. ADC and trace values are corrupted.')                  # Issue warning of corrupted data
    ########## Calculate trace and apparent diffusion coefficient #################################################################################
    b                                = bval_high - bval_low                                                                     # Calculate effective b-value
    S_0                              = stacked_matrix[:, :, :, bval_low_indicies[0]]                                            # Extract lower b-value matrix from original matrix
    S_x                              = stacked_matrix[:, :, :, bval_high_indicies[0]]                                           # Extract higher b-value matrix from original matrix in the x direction
    S_y                              = stacked_matrix[:, :, :, bval_high_indicies[1]]                                           # Extract higher b-value matrix from original matrix in the y direction
    S_z                              = stacked_matrix[:, :, :, bval_high_indicies[2]]                                           # Extract higher b-value matrix from original matrix in the z direction
    D_xx                             = -(1 / b) * np.log(S_x / S_0)                                                             # Calculate x diffusion coefficient
    D_yy                             = -(1 / b) * np.log(S_y / S_0)                                                             # Calculate y diffusion coefficient
    D_zz                             = -(1 / b) * np.log(S_z / S_0)                                                             # Calculate z diffusion coefficient
    trace                            = (D_xx + D_yy + D_zz) * 1000                                                              # Calculate trace and scale to mm^2 / μs
    trace                            = np.clip(trace, 0, 12)                                                                    # Clip trace values to be between 0 and 12
    apparent_diffusion_coefficient   = trace / 3                                                                                # Calculate apparent diffusion coefficient that is already scaled to mm^2 / μs
    ########## Store standard DWI metrics in a dictionary ##########################################################################################
    Standard_DWI_Metrics = dict()                                                                                               # Initialize DWI metrics dictionary
    Standard_DWI_Metrics['ADC'] = apparent_diffusion_coefficient                                                                # Store apparent diffusion coefficient in DWI metrics dictionary
    Standard_DWI_Metrics['TR']  = trace                                                                                         # Store trace in DWI metrics dictionary
    return [Standard_DWI_Metrics]
