def ADC_Filter(original_matrix, original_bvals, original_bvecs, operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values (Optional).
    original_bvecs          : Sorted b-vectors (Optional).
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    averaged_matrix         : Sorted ADC filtered diffusion data (5D - [rows, columns, slices, directions, singleton dimension]).
    averaged_bvals          : Sorted b-values (Optional).
    averaged_bvecs          : Sorted b-vectors (Optional).
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2023
    ########## Import modules #####################################################################################################################
    import numpy as     np                                                                                                                          # Import numpy module
    ########## Detection of different b-values ####################################################################################################
    (unique, index, counts) = np.unique(original_bvals, return_counts = True, return_index = True, axis = 0)                                        # Extract number of different diffusion directions
    numBvals                = len(unique)                                                                                                           # Identify number of b-values in stacked b-values
    bval_low                = min(unique)                                                                                                           # Identify lower b-value (typically b = 0)
    bval_low_indicies       = np.where(original_bvals == bval_low)[0]                                                                               # Identify index of lower b-value in stacked b-values
    bval_high               = max(unique)                                                                                                           # Identify higher b-value
    bval_high_indicies      = np.where(original_bvals == bval_high)[0]                                                                              #
    ###
    filtered_matrix = np.copy(original_matrix)                                                                                                      #
    temp_matrix     = np.abs(original_matrix)                                                                                                       #
    for dif in range(original_matrix.shape[3]):                                                                                                     #
        if dif == bval_low_indicies:                                                                                                                    #
            print('Low b value, nothing computed')                                                                                                          #
        else:                                                                                                                                           #
            for slc in range(original_matrix.shape[2]):                                                                                                     #
                for col in range(original_matrix.shape[1]):                                                                                                     #
                    for row in range(original_matrix.shape[0]):                                                                                                     #
                        b    = bval_high - bval_low                                                                                                                     # Calculate effective b-value
                        S_0  = temp_matrix[row, col, slc, bval_low_indicies[0], :]                                                                                      # Extract lower b-value matrix from original matrix
                        S_x  = temp_matrix[row, col, slc, dif, :]                                                                                                       # Extract higher b-value matrix from original matrix in the x direction
                        D_xx = -(1 / b) * np.log(S_x / S_0)                                                                                                             # Calculate x diffusion coefficient
                        D_xx = D_xx * 1000                                                                                                                              #
                        if all(x >= 3 for x in D_xx):                                                                                                                   #
                            min_val = min(D_xx)                                                                                                                             #
                            for avg in range(len(D_xx)):                                                                                                                    #
                                if D_xx[avg] != min_val:                                                                                                                        #
                                    filtered_matrix[row, col, slc, dif, avg] = np.nan                                                                                               #
                        else:                                                                                                                                           #
                            for avg in range(len(D_xx)):                                                                                                                    #
                                if D_xx[avg] >= 3:                                                                                                                              #
                                    filtered_matrix[row, col, slc, dif, avg] = np.nan                                                                                               #
    filtered_bvals  = original_bvals                                                                                                                # Store b-values
    filtered_bvecs  = original_bvecs                                                                                                                # Store b-vectors
    return [filtered_matrix, filtered_bvals, filtered_bvecs]
