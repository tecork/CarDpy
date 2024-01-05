def average(original_matrix, original_bvals = [], original_bvecs = [], operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values (Optional).
    original_bvecs          : Sorted b-vectors (Optional).
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    averaged_matrix         : Sorted averaged diffusion data (5D - [rows, columns, slices, directions, singleton dimension]).
    averaged_bvals          : Sorted b-values (Optional).
    averaged_bvecs          : Sorted b-vectors (Optional).
    ########## References #########################################################################################################################
    Please cite the following articles if using complex averaging:
    [1] Fan Q, Nummenmaa A, Witzel T, Ohringer N, Tian Q, Setsompop K, et al.
        Axon diameter index estimation independent of fiber orientation distribution using high-gradient diffusion MRI.
        NeuroImage 2020;222:117197.
        doi: 10.1016/j.neuroimage.2020.117197
    [2] Scott AD, Nielles-Vallespin S, Ferreira PF, McGill L-A, Pennell DJ, Firmin DN.
        The effects of noise in cardiac diffusion tensor imaging and the benefits of averaging complex data.
        NMR in Biomedicine 2016;29:588–99.
        doi: 10.1002/nbm.3500
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    import numpy               as     np                                                                                            # Import numpy module
    from   cardpy.FT_Operators import fft2c, ifft2c                                                                                 # Import Fourier transform operators from CarDpy
    ########## Address data type of original data and average along 5th dimension #################################################################
    if original_matrix.dtype == 'complex128':                                                                                       # If data type is complex ...
        if operation_type == 'Complex':                                                                                                 # If operation type is complex ...
            rows                            = original_matrix.shape[0]                                                                      # Extract number of rows
            columns                         = original_matrix.shape[1]                                                                      # Extract number of columns
            slices                          = original_matrix.shape[2]                                                                      # Extract number of slices
            directions                      = original_matrix.shape[3]                                                                      # Extract number of directions
            averages                        = original_matrix.shape[4]                                                                      # Extract number of averages
            background_phase_removed_matrix = np.zeros(original_matrix.shape)                                                               # Initalize background phased removed matrix
            background_phase_removed_matrix = background_phase_removed_matrix.astype(np.complex128)                                         # Cast background phased removed matrix as complex
            rows_subdivide = int(np.rint(np.nextafter(rows / 8,  rows / 8 + 1)))                                                            # Subdivide rows in to 8ths
            rows_keep      = rows - (rows_subdivide * 2)                                                                                    # Keep ~75% of rows
            pad_rows_num   = rows_subdivide                                                                                                 # Determine number of rows to pad on each side
            cols_subdivide = int(np.rint(np.nextafter(columns / 8,  columns / 8 + 1)))                                                      # Subdivide columns in to 8ths
            cols_keep      = columns - (cols_subdivide * 2)                                                                                 # Keep ~75% of columns
            pad_cols_num   = cols_subdivide                                                                                                 # Determine number of columns to pad on each side
            hamming_filter                  = np.sqrt(np.outer(np.hamming(rows_keep), np.hamming(cols_keep)))                               # Creating hamming filter with
            background_phase_removal_filter = np.pad(hamming_filter, [(pad_rows_num, pad_rows_num), (pad_cols_num, pad_cols_num)])          # Pad with zeros back to original size
            for slc in range(slices):                                                                                                       # Iterate through slices
                    for dif in range(directions):                                                                                               # Iterate through diffusion directions
                        for avg in range(averages):                                                                                                    # Iterate through averages
                            original_image                 = original_matrix[:, :, slc, dif, avg]                                                       # Store original image from original matrix
                            filtered_kspace                = fft2c(original_image) * background_phase_removal_filter                                    # Apply background phase removal filter in k space
                            background_phase_removed_image = ifft2c(filtered_kspace)                                                                    # Convert background phase removed k space to image space
                            magnitude                      = np.abs(original_image)                                                                     # Store original magnitude data
                            phase                          = np.exp(1j * np.angle(original_image))                                                      # Store original phase data
                            phase_background_removed       = np.exp(1j * np.angle(background_phase_removed_image))                                      # Store background removed phase data
                            background_phase_removed_matrix[:, :, slc, dif, avg] = magnitude * (phase / phase_background_removed)                       # Apply phase background removal to complex data and store
            averaged_matrix = np.nanmean(background_phase_removed_matrix, axis = 4)                                                            # Complex average along the 5th dimesnion
        if operation_type == 'Magnitude':                                                                                               # If operation type is magnitude ...
            averaged_matrix = np.nanmean(np.abs(original_matrix), axis = 4)                                                                    # Magnitude average along 5th dimension
            print('Input data type is complex, but magnitude is being executed.')                                                           # Print warning
    else:                                                                                                                               # Otherwise ...
        averaged_matrix = np.nanmean(original_matrix, axis = 4)                                                                                # Magnitude average along 5th dimension
    ########## Tie-up-loose-ends ... ##############################################################################################################
    averaged_matrix = averaged_matrix[:, :, :, :, np.newaxis]                                                                           # Add singleton dimension in 5th dimension
    averaged_bvals  = original_bvals                                                                                                    # Store averaged b-values
    averaged_bvecs  = original_bvecs                                                                                                    # Store averaged b-vectors
    return [averaged_matrix, averaged_bvals, averaged_bvecs]
