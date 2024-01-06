def respiratory_sorting(original_matrix, original_bvals, original_bvecs, zoom = 'ON', IntERACT_zoom = 'ON', organ = 'Liver', operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values.
    original_bvecs          : Sorted b-vectors.
    zoom                    :
    IntERACT_zoom           :
    organ                   :
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    respiratory_matrix      : Sorted respiratory ordered diffusion data (5D - [rows, columns, slices, directions, averages]).
                              Data is sorted in the average dimension from expiration to inspiration.
    respiratory_bvals       : Sorted b-values.
    respiratory_bvecs       : Sorted b-vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    import numpy                            as np                                                                                                   # Import numpy module
    from   cardpy.Data_Sorting              import sorted2stacked, stacked2sorted                                                                   # Import sorted to stacked and stacked to sorted from CarDpy
    from   cardpy.GUI_Tools.IntERACT        import INTERACT_GUI, execute_crop, next_slice, finish_program, update_plots
    from   cardpy.Data_Processing.Denoising import denoise                                                                                          #
    from   skimage.filters.rank             import entropy                                                                                          #
    from   skimage.morphology               import disk                                                                                             #
    from   skimage.metrics                  import structural_similarity as ssim                                                                    # Import structural similarity index from skimage module
    from   skimage.metrics                  import mean_squared_error as mse                                                                        # Import mean squared error from skimage module
    ########## Extract information about original data matrix #####################################################################################
    directions = original_matrix.shape[3]                                                                                                           # Extract number of directions
    slices     = original_matrix.shape[2]                                                                                                           # Extract number of slices
    averages   = original_matrix.shape[4]                                                                                                           # Extract number of averages
    ########## Initialize respiratory matrix and address data type of original data ###############################################################
    respiratory_matrix = np.zeros(original_matrix.shape)                                                                                            # Initialize respiratory matrix
    if original_matrix.dtype == 'complex128':                                                                                                       # If data type is complex ...
        if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
            respiratory_matrix = respiratory_matrix.astype(np.complex128)                                                                                  # Cast respiratory matrix for complex data
        if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
            respiratory_matrix = respiratory_matrix.astype(np.float64)                                                                                      # Cast respiratory matrix for magnitude data
            original_matrix    = np.abs(original_matrix)                                                                                                    # Convert complex original matrix to magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                                                           # Print warning
        temporary_matrix = np.abs(original_matrix)                                                                                                      # Create temporary matrix variable using magnitude data
    else:                                                                                                                                           # Otherwise ...
        respiratory_matrix = respiratory_matrix.astype(np.float64)                                                                                      # Cast interpolated matrix for magnitude data
        temporary_matrix   = original_matrix                                                                                                            # Create temporary matrix variable
    [temporary_matrix_stacked, original_bvals_stacked, original_bvecs_stacked] = sorted2stacked(temporary_matrix, original_bvals, original_bvecs)   #
    ########## ROI Cropping for Image #############################################################################################################
    Slice_Crop_Coortinates = []                                                                                                                     #
    if zoom == 'ON':                                                                                                                                #
        if IntERACT_zoom == 'ON':                                                                                                                       #
            [x_start, x_end, y_start, y_end] = INTERACT_GUI(temporary_matrix_stacked, organ)                                                            #
            Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]                                                                                   #
        if IntERACT_zoom == 'OFF':                                                                                                                  #
            print('Interact is off')                                                                                                                    #
            x_start = []                                                                                                                                #
            x_end   = []                                                                                                                                #
            y_start = []                                                                                                                                #
            y_end   = []                                                                                                                                #
            temporary_matrix_list = []                                                                                                                  #
            for slc in range(slices):                                                                                                                   # Iterate through slices
                image               = np.max(temporary_matrix_stacked[:, :, slc, :], axis = 2)                                                              #
                image_normalization = image / image.max()                                                                                                   #
                title_pre = 'Select area for slice ' + str(slc + 1)                                                                                         #
                r = cv2.selectROI(title_pre, image_normalization, False, False)
                cv2.startWindowThread()
                x_start.append(int(r[0]))
                x_end.append(int(r[0]+r[2]))
                y_start.append(int(r[1]))
                y_end.append(int(r[1]+r[3]))
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
            Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]
    if zoom == 'OFF':
        x_start = []
        x_end   = []
        y_start = []
        y_end   = []
        temporary_matrix_list = []
        for slc in range(original_matrix_stacked.shape[2]):
            x_start.append(int(0))
            x_end.append(int(original_matrix_stacked.shape[1]))
            y_start.append(int(0))
            y_end.append(int(original_matrix_stacked.shape[0]))
        Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]

    ### Extract Crop Coordinates
    x_start = Slice_Crop_Coordinates[0]
    x_end   = Slice_Crop_Coordinates[1]
    y_start = Slice_Crop_Coordinates[2]
    y_end   = Slice_Crop_Coordinates[3]

    ########## Denoise temporary matrix ###########################################################################################################
    [denoised_matrix, _, _] = denoise(temporary_matrix,
                                      original_bvals,
                                      original_bvecs,
                                      denoising_algorithm = 'LocalPCA')                                                                             # Denoise temporary matrix
    ########## Apply local entropy filter for feature detection ###################################################################################
    entropy_matrix = np.zeros(denoised_matrix.shape)                                                                                                # Initialize entropy matrix
    
    for avg in range(averages):                                                                                                                     # Iterate through averages
        for dif in range(directions):                                                                                                                   # Iterate through diffusion directions
            for slc in range(slices):                                                                                                                       # Iterate through slices
                image = denoised_matrix[:, :, slc, dif, avg]
                image_normalization = image / image.max()
                normalized_entropy_image = entropy(image_normalization, disk(7))
#                entropy_image = entropy(denoised_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, dif, avg] / denoised_matrix[:, :, slc, dif, avg].max(), disk(7))                             # Compute entropy image
                entropy_matrix[:, :, slc, dif, avg] = normalized_entropy_image                                                                                            # Store entropy image in entropy matrix
    ##### Structural similarity index measure (SSIM) and root mean square error (RMSE) of entropy matrix across averages ##########################
    SSIM = np.zeros([averages, averages, directions, slices])                                                                                       # Initialize SSIM matrix
    RMSE = np.zeros([averages, averages, directions, slices])                                                                                       # Initialize RMSE matrix
    for dif in range(directions):                                                                                                                   # Iterate through diffusion directions
        for slc in range(slices):                                                                                                                       # Iterate through slices
            for avg_i in range(averages):                                                                                                                   # Iterate through averages (ith average)
                for avg_j in range(averages):                                                                                                                   # Iterate through averages (jth average)
                    SSIM[avg_i, avg_j, dif, slc] = np.round(ssim(entropy_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, dif, avg_i],
                                                                 entropy_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, dif, avg_j]), 2)                                           # Calculate SSIM between ith and jth average
                    RMSE[avg_i, avg_j, dif, slc] = np.round(np.sqrt(mse(entropy_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, dif, avg_i],
                                                                        entropy_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, dif, avg_j])), 2)                                   # Calculate RMSE between ith and jth average
    ########## Assess respiratory motion across averages and resort averages ######################################################################
    for dif in range(directions):                                                                                                                   # Iterate through diffusion directions
        for slc in range(slices):                                                                                                                       # Iterate through slices
            acquisition_index = np.where(SSIM[:, :, dif, slc] == SSIM[:, :, dif, slc].min())                                                                # Identify index (indices) of minimum SSIM (largest difference between averages)
            if len(acquisition_index[0]) > 2:                                                                                                               # If more than one pair of indices is identified ...
                print('For diffusion direction %i on slice %i, SSIM is not enough: evaluating RSME' %(dif, slc))                                                # Print warning that RMSE will be the tie breaker
                filtered_RMSE = RMSE[:, :, dif, slc]                                                                                                            # Create filtered RMSE matrix for current case
                filtered_RMSE[SSIM[:, :, dif, slc] > SSIM[:, :, dif, slc].min()] = 0                                                                            # Filter out non-minimum SSIM values
                acquisition_index = np.where(filtered_RMSE == filtered_RMSE.max())                                                                              # Identify index of acquisition with maximum RMSE (largest difference between averages)
                if acquisition_index[0][0] == acquisition_index[0][1]:                                                                                          # If first two indicies are the same
                    acquisition_index[0][1] = acquisition_index[0][2]                                                                                               # Take the third index instead
            image_index_1 = acquisition_index[0][0]                                                                                                         # Define image index 1
            image_index_2 = acquisition_index[0][1]                                                                                                         # Define image index 2
            image_sum_1   = np.nansum(temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, dif, image_index_1])                                                               # Take the total pixel sum of average for image index 1
            image_sum_2   = np.nansum(temporary_matrix[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, dif, image_index_2])                                                               # Take the total pixel sum of average for image index 2
            if image_sum_1 > image_sum_2:                                                                                                                   # If the total pixel sum of average for image index 1 is greater ...
                expiration_index  = image_index_2                                                                                                               # Expiration index is defined as the average for image index 2
                inspiration_index = image_index_1                                                                                                               # Inspiration index is defined as the average for image index 1
            if image_sum_1 < image_sum_2:                                                                                                                   # If the total pixel sum of average for image index 2 is greater ...
                expiration_index  = image_index_1                                                                                                               # Expiration index is defined as the average for image index 1
                inspiration_index = image_index_2                                                                                                               # Inspiration index is defined as the average for image index 2
            if image_sum_1 == image_sum_2:                                                                                                                  # If the total pixel sum of average for image index 1 and 2 are the same ...
                print('For diffusion direction %i on slice %i, no respiratory differences identified. ' %(dif, slc))                                            # Print warning that data is the same
                expiration_index  = image_index_1                                                                                                               # Expiration index is defined as the average for image index 1
                inspiration_index = image_index_2                                                                                                               # Inspiration index is defined as the average for image index 2
            expiration_to_inspiration       = SSIM[expiration_index, :, dif, slc].tolist()                                                                  # Create expiration to inspiration list with expiration index
            expiration_to_inspiration_index = np.flip(np.argsort(expiration_to_inspiration))                                                                # Reverse sort expiration to inspiration list to start with expiration average
            for avg in range(averages):                                                                                                                     # Iterate through averages
                respiratory_matrix[:, :, slc, dif, avg] = original_matrix[:, :, slc, dif, expiration_to_inspiration_index[avg]]                                 # Store original matrix averages using the expiration to inspiration list
    ########## Tie-up-loose-ends ... ##############################################################################################################
    respiratory_bvals = original_bvals                                                                                                              # Store b-value information
    respiratory_bvecs = original_bvecs                                                                                                              # Store b-vector information
    return [respiratory_matrix, respiratory_bvals, respiratory_bvecs, Slice_Crop_Coordinates]
