def zero_filled(original_matrix, original_bvals = [], original_bvecs = [], operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values (Optional).
    original_bvecs          : Sorted b-vectors (Optional).
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    interpolated_matrix     : Sorted interpolated diffusion data (5D - [rows, columns, slices, directions, averages]).
                              Zero-interpolation filling (ZIP) method by a factor of 2 using a 2D Kaiser (24.6) spectral filter.
    interpolated_bvals      : Sorted b-values (Optional).
    interpolated_bvecs      : Sorted b-vectors (Optional).
    ########## References #########################################################################################################################

    """
    ########## Definition Information ##############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules ######################################################################################################################
    import numpy               as     np                                                                                        # Import numpy module
    from   cardpy.FT_Operators import fft2c, ifft2c                                                                             # Import Fourier transform operators from MySTiC MRpy
    ########## Initialize interpolated matrix ######################################################################################################
    rows                            = original_matrix.shape[0]                                                                  # Extract number of rows
    columns                         = original_matrix.shape[1]                                                                  # Extract number of columns
    slices                          = original_matrix.shape[2]                                                                  # Extract number of slices
    directions                      = original_matrix.shape[3]                                                                  # Extract number of directions
    averages                        = original_matrix.shape[4]                                                                  # Extract number of averages
    interpolated_matrix             = np.zeros([rows * 2, columns * 2, slices, directions, averages])                           # Initialize interpolated matrix
    ########## Address data type of original data ##################################################################################################
    if original_matrix.dtype == 'complex128':                                                                                   # If data type is complex ...
        if operation_type == 'Complex':                                                                                             # If operation type is complex ...
            interpolated_matrix  = interpolated_matrix.astype(np.complex128)                                                            # Cast interpolated matrix for complex data
        if operation_type == 'Magnitude':                                                                                           # If operation type is magnitude ...
            interpolated_matrix = interpolated_matrix.astype(np.float64)                                                                # Cast interpolated matrix for magnitude data
            original_matrix     = np.abs(original_matrix)                                                                               # Convert complex original matrix to magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                                       # Print warning
    else:                                                                                                                       # Otherwise ...
        interpolated_matrix = interpolated_matrix.astype(np.float64)                                                                # Cast interpolated matrix for magnitude data
    ########## Create spectral filter for interpolated matrix ######################################################################################
    spectral_filter = np.sqrt(np.outer(np.kaiser(interpolated_matrix.shape[0], 24.6), np.kaiser(interpolated_matrix.shape[1], 24.6)))
    ########## Zero-interpolation filling ##########################################################################################################
    pad_rows_num        = int(rows / 2)                                                                                         # Determine number of rows to pad on each side
    pad_cols_num        = int(columns / 2)                                                                                      # Determine number of columns to pad on each side
    for slc in range(slices):                                                                                                   # Iterate through slices
        for dif in range(directions):                                                                                               # Iterate through diffusion directions
            for avg in range(averages):                                                                                                 # Iterate through averages
                reject_adjustment = 'OFF'
                if np.sum(original_matrix[:, :, slc, dif, avg]) < 0:
                    reject_adjustment = 'ON'
                    temp_image        = original_matrix[:, :, slc, dif, avg]
                    temp_image_min    = temp_image.min()
                    temp_image_offset = temp_image - temp_image_min
                    original_matrix[:, :, slc, dif, avg] = temp_image_offset
                original_image      = original_matrix[:, :, slc, dif, avg]                                                                  # Extract single image from matrix
                original_kspace     = fft2c(original_image)                                                                                 # Convert single image to k-space
                if operation_type == 'Complex':                                                                                             # If operation type is complex ...
                    real_kspace              = np.real(original_kspace)                                                                         # Separate real part from complex data
                    imag_kspace              = np.imag(original_kspace)                                                                         # Separate imaginary part from complex data
                    interpolated_real_kspace = np.pad(real_kspace, [(pad_rows_num, pad_rows_num), (pad_cols_num, pad_cols_num)])                # Add row and column zero pads to interpolate real k-space
                    interpolated_imag_kspace = np.pad(imag_kspace, [(pad_rows_num, pad_rows_num), (pad_cols_num, pad_cols_num)])                # Add row and column zero pads to interpolate imaginary k-space
                    interpolated_real_kspace = interpolated_real_kspace * spectral_filter                                                       # Multiply interpolated real k-space by spectral filter (2D Kaiser)
                    interpolated_imag_kspace = interpolated_imag_kspace * spectral_filter                                                       # Multiply interpolated imaginary k-space by spectral filter (2D Kaiser)
                    interpolated_kspace      = interpolated_real_kspace + (1j * interpolated_imag_kspace)                                       # Combine interpolated real and imaginary k-space into one array
                    interpolated_image       = ifft2c(interpolated_kspace)                                                                      # Convert interpolated k-space to interpolated image
                if operation_type == 'Magnitude':                                                                                           # If operation type is magnitude ...
                    interpolated_kspace = np.pad(original_kspace, [(pad_rows_num, pad_rows_num), (pad_cols_num, pad_cols_num)])                 # Add row and column zero pads to interpolate k-space
                    interpolated_kspace = interpolated_kspace * spectral_filter                                                                 # Multiply interpolated k-space by spectral filter (2D Kaiser)
                    interpolated_image  = np.abs(ifft2c(interpolated_kspace))                                                                   # Convert interpolated k-space to interpolated image
                interpolated_matrix[:, :, slc, dif, avg] = interpolated_image * 4                                                           # Scale interpolated image intensity values up by a factor of 4
                if reject_adjustment == 'ON':
                    temp_image        = interpolated_matrix[:, :, slc, dif, avg]
                    temp_image_max    = temp_image.max()
                    temp_image_offset = temp_image - temp_image_max
                    interpolated_matrix[:, :, slc, dif, avg] = temp_image_offset
    ########## Tie-up-loose-ends ... ##############################################################################################################
    interpolated_bvals  = original_bvals                                                                                        # Store b-value information
    interpolated_bvecs  = original_bvecs                                                                                        # Store b-vector information
    return [interpolated_matrix, interpolated_bvals, interpolated_bvecs]
