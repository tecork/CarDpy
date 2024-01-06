def sorted2stacked(sorted_matrix, sorted_bvals, sorted_bvecs):
    """
    ########## Definition Inputs ##################################################################################################################
    # sorted_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    # sorted_bvals          : Sorted b-values.
    # sorted_bvecs          : Sorted b-vectors.
    ########## Definition Outputs #################################################################################################################
    # stacked_matrix        : Stacked diffusion data (4D - [rows, columns, slices, directions]).
    # stacked_bvals         : Stacked b-values.
    # stacked_bvecs         : Stacked b-vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import numpy as np                                                                                                  # Import numpy module
    ########## Identify new matrix dimensions ######################################################################################################
    numRow                  = sorted_matrix.shape[0]                                                                    # Define number of rows (x)
    numCol                  = sorted_matrix.shape[1]                                                                    # Define number of columns (y)
    numSlc                  = sorted_matrix.shape[2]                                                                    # Define number of slices (z)
    numDif                  = sorted_matrix.shape[3] * sorted_matrix.shape[4]                                           # Define number of stacked diffusion directions
    stacked_matrix          = np.zeros([numRow, numCol, numSlc, numDif])                                                # Initialize sorted matrix
    ########## Address data type of sorted matrix ##################################################################################################
    if sorted_matrix.dtype == 'complex128':                                                                             # If data type is complex ...
            stacked_matrix = stacked_matrix.astype(np.complex128)                                                           # Cast sorted matrix for complex data
    else:                                                                                                               # Otherwise ...
            stacked_matrix = stacked_matrix.astype(np.float64)                                                              # Cast sorted matrix for magnitude data
    ########## Stack 5D diffusion data into 4D #####################################################################################################
    counter                 = -1                                                                                        # Initialize counter for 4th dimension
    for dif in range(sorted_matrix.shape[4]):                                                                           # Iterate through diffusion directions
        for slc in range(sorted_matrix.shape[3]):                                                                           # Iterate through stacked diffusion dimension
            counter = counter + 1                                                                                               # Increase counter by one
            stacked_matrix[:, :, :, counter] = sorted_matrix[:, :, :, slc, dif]                                                 # Store stacked matrix from current index of sorted matrix
    ########## Tie-up-loose-ends ... ##############################################################################################################
    stacked_bvals  = np.tile(sorted_bvals, reps = sorted_matrix.shape[4])                                               # Store b-value information
    stacked_bvecs  = np.tile(sorted_bvecs, reps = (sorted_matrix.shape[4], 1))                                          # Store b-vector information
    return [stacked_matrix, stacked_bvals, stacked_bvecs]
    
def stacked2sorted(stacked_matrix, stacked_bvals, stacked_bvecs):
    """
    ########## Definition Inputs ##################################################################################################################
    # stacked_matrix        : Stacked diffusion data (4D - [rows, columns, slices, directions]).
    # stacked_bvals         : Stacked b-values.
    # stacked_bvecs         : Stacked b-vectors.
    ########## Definition Outputs #################################################################################################################
    # sorted_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    # sorted_bvals          : Sorted b-values.
    # sorted_bvecs          : Sorted b-vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import numpy as np                                                                                                  # Import numpy module
    ########## Identify new matrix dimensions ######################################################################################################
    numRow                  = stacked_matrix.shape[0]                                                                   # Define number of rows (x)
    numCol                  = stacked_matrix.shape[1]                                                                   # Define number of columns (y)
    numSlc                  = stacked_matrix.shape[2]                                                                   # Define number of slices (z)
    (unique, index, counts) = np.unique(stacked_bvecs, return_counts = True, return_index = True, axis = 0)             # Extract number of different diffusion directions
    unique                  = unique[np.argsort(index)]                                                                 # Retain stacked diffusion direction order
    numDif                  = len(unique)                                                                               # Define number of diffusion directions
    numAvg                  = max(counts)                                                                               # Define number of averages
    sorted_matrix           = np.zeros([numRow, numCol, numSlc, numDif, numAvg])                                        # Initialize sorted matrix
    ########## Address data type of stacked matrix #################################################################################################
    if stacked_matrix.dtype == 'complex128':                                                                            # If data type is complex ...
        sorted_matrix = sorted_matrix.astype(np.complex128)                                                                # Cast sorted matrix for complex data
    else:                                                                                                               # Otherwise ...
        sorted_matrix = sorted_matrix.astype(np.float64)                                                                   # Cast sorted matrix for magnitude data
    ########## Sort 4D diffusion data into 5D ######################################################################################################
    stacked_index_list      = []                                                                                        # Initialize stacked index list
    for dif in range(numDif):                                                                                           # Iterate through diffusion directions
        idxAvg = -1                                                                                                         # Initialize average index
        for idx in range(stacked_matrix.shape[3]):                                                                          # Iterate through stacked diffusion dimension
            current_search = unique[dif]                                                                                        # Define current diffusion direction
            if np.array_equal(current_search, stacked_bvecs[idx, :]) == True:                                                   # Check if current diff dir matches desired diff dir ...
                idxAvg                                 = idxAvg + 1                                                                 # Index current average
                idxDif                                 = np.where(np.all(unique == stacked_bvecs[idx, :],axis = 1))[0][0]           # Index current diffuion direction
                sorted_matrix[:, :, :, idxDif, idxAvg] = stacked_matrix[:, :, :, idx]                                               # Store sorted matrix from current index of stacked matrix
    ########## Tie-up-loose-ends ... ##############################################################################################################
    sorted_bvecs  = unique                                                                                              # Store sorted b-vectors
    sorted_bvals  = stacked_bvals[0:numDif]                                                                             # Store sorted b-values
    return [sorted_matrix, sorted_bvals, sorted_bvecs]
