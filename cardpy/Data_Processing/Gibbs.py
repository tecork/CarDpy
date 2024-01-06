def unrung(original_matrix, original_bvals, original_bvecs, operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values.
    original_bvecs          : Sorted b-vectors.
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    unrung_matrix           : Sorted averaged diffusion data (5D - [rows, columns, slices, directions, singleton dimension]).
    unrung_bvals            : Sorted b-values (Optional).
    unrung_bvecs            : Sorted b-vectors (Optional).
    ########## References #########################################################################################################################
    Please cite the following articles:
    [1] Henriques NR, Correia MM.
        Advanced methods for diffusion MRI data analysis and their application to the Healthy Ageing Brain. [Doctoral thesis]
        doi: 10.17863/CAM.29356
    [2] Kellner E, Dhital B, Kiselev VG, Reisert M.
        Gibbs-ringing artifact removal based on local subvoxel-shifts.
        Magnetic Resonance in Medicine 2015;76:1574–81.
        doi: 10.1002/mrm.26054
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    from   cardpy.Data_Sorting import sorted2stacked, stacked2sorted                                          # Import sorted to stacked and stacked to sorted from CarDpy
    import numpy               as     np                                                                      # Import numpy module
    from   dipy.denoise.gibbs  import gibbs_removal                                                           # Import Gibb's removal from DiPy
    ########## Address data type of stacked matrix ################################################################################################
    [stacked_matrix, stacked_bvals, stacked_bvecs] = sorted2stacked(original_matrix, original_bvals, original_bvecs)    # Convert sorted data into stacked data
    if stacked_matrix.dtype == 'complex128':                                                                            # If data type is complex ...
        if operation_type == 'Complex':                                                                                     # If operation type is complex ...
            magnitude_matrix = np.abs(stacked_matrix)                                                                           # Separate magnitude part from complex data
            phase_matrix     = np.angle(stacked_matrix)                                                                         # Separate phase part from complex data
            unrung_matrix    = np.zeros(stacked_matrix.shape)                                                                   # Initialize unrung matrix
            unrung_matrix    = unrung_matrix.astype(np.complex128)                                                              # Cast unrung matrix for complex data
        if operation_type == 'Magnitude':                                                                                   # If operation type is magnitude ...
            stacked_matrix = np.abs(stacked_matrix)                                                                             # Convert complex original matrix to magnitude data
            unrung_matrix  = np.zeros(stacked_matrix.shape)                                                                     # Initialize unrung matrix
            unrung_matrix  = unrung_matrix.astype(np.float64)                                                                   # Cast unrung matrix for magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                               # Print warning
    else:                                                                                                               # Otherwise ...
        unrung_matrix = np.zeros(stacked_matrix.shape)                                                                      # Initialize unrung matrix
        unrung_matrix = unrung_matrix.astype(np.float64)                                                                    # Cast unrung matrix for magnitude data
    ########## Tie-up-loose-ends ... ##############################################################################################################
    if operation_type == 'Complex':                                                                                     # If operation type is complex ...
        unrung_matrix_magnitude = gibbs_removal(magnitude_matrix, inplace = False)                                          # Compute unrung magnitude data
        unrung_matrix           = unrung_matrix_magnitude * np.exp(1j * phase_matrix)                                       # Store unrung complex data
    if operation_type == 'Magnitude':                                                                                   # If operation type is magnitude ...
        unrung_matrix = gibbs_removal(stacked_matrix, inplace = False)                                                      # Compute unrung magnitude data
    ########## Tie-up-loose-ends ... ##############################################################################################################
    [unrung_matrix, _, _] = stacked2sorted(unrung_matrix, stacked_bvals, stacked_bvecs)                                 # Convert stacked data into sorted data
    unrung_bvals          = original_bvals                                                                              # Store unrung b-values
    unrung_bvecs          = original_bvecs                                                                              # Store unrung b-vectors
    return [unrung_matrix, unrung_bvals, unrung_bvecs]
