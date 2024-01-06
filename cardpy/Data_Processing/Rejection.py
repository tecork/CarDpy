def shot_rejection(original_matrix, original_bvals, original_bvecs, NRMSE_threshold = 0.75, NSSIM_threshold = 0.75, zoom = 'ON', IntERACT_zoom = 'ON', organ = 'Heart', operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix       : Sorted diffusion data (5D).
    original_bvals        : Sorted b-values.
    original_bvecs        : Sorted b vectors.
    zoom                  :
    IntERACT_zoom         :
    organ                 :
    operation_type        : Identify the type of operation for original matrix (Optional).
                            Default operation type is magnitude.
                            Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    accepted_matrix       : Accepted matrix (5D).
    accepted_bvals        : Accepted b-values.
    accepted_bvecs        : Accepted b vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules #####################################################################################################################
    from   cardpy.Data_Sorting       import sorted2stacked, stacked2sorted                                                          # Import sorted to stacked and stacked to sorted from CarDpy
    from   cardpy.GUI_Tools.IntERACT import INTERACT_GUI, execute_crop, next_slice, finish_program, update_plots                    #
    import numpy                     as     np                                                                                      #
    import cv2                                                                                                                      #
    from   skimage.metrics           import structural_similarity as ssim                                                           # Import strutural similarity metric from skimage module
    from   skimage.metrics           import mean_squared_error    as mse                                                            #
    from   yellowbrick.cluster       import KElbowVisualizer                                                                        #
    import matplotlib.style                                                                                                         #
    import matplotlib                as mpl                                                                                         #
    import matplotlib.pyplot         as plt                                                                                         #
    from   sklearn.cluster           import KMeans                                                                                  # Import K means clustering from sklearn module
    import itertools                                                                                                                #
    import math                                                                                                                     #
    mpl.style.use('default')                                                                                                        #

    ########## Initialize accepted matrix and address data type of stacked matrix #################################################################
    [original_matrix_stacked, original_bvals_stacked, original_bvecs_stacked] = sorted2stacked(original_matrix,
                                                                                               original_bvals,
                                                                                               original_bvecs)
    accepted_matrix                                = np.zeros(original_matrix.shape)                                                # Initialize accepted matrix
    if original_matrix_stacked.dtype == 'complex128':                                                                                        # If data type is complex ...
        if operation_type == 'Complex':                                                                                                 # If operation type is complex ...
            accepted_matrix = accepted_matrix.astype(np.complex128)                                                                         # Cast accepted matrix for complex data
        if operation_type == 'Magnitude':                                                                                               # If operation type is magnitude ...
            accepted_matrix = accepted_matrix.astype(np.float64)                                                                            # Cast accepted matrix for magnitude data
            original_matrix = abs(original_matrix_stacked)                                                                                           # Convert complex original matrix to magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                                           # Print warning
        temporary_matrix = abs(original_matrix_stacked)                                                                                          # Create temporary matrix variable using magnitude data
    else:                                                                                                                          # Otherwise ...
        accepted_matrix  = accepted_matrix.astype(np.float64)                                                                           # Cast accepted matrix for magnitude data
        temporary_matrix = original_matrix_stacked                                                                                              # Create temporary matrix variable
    ########## ROI Cropping for Image #############################################################################################################
    slices                 = original_matrix_stacked.shape[2]                                                                      #
    Slice_Crop_Coortinates = []                                                                                                    #
    if zoom == 'ON':                                                                                                               #
        print(zoom)
        print(slices)
        print(original_matrix_stacked.shape)
        if IntERACT_zoom == 'ON':
            [x_start, x_end, y_start, y_end] = INTERACT_GUI(original_matrix_stacked, organ)
            Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]
            print(Slice_Crop_Coordinates)
        if IntERACT_zoom == 'OFF':
            print('Interact is off')
            x_start = []                                                                                                                   #
            x_end   = []                                                                                                                   #
            y_start = []                                                                                                                   #
            y_end   = []                                                                                                                   #
            temporary_matrix_list = []                                                                                                     #
            for slc in range(slices):                                                                            # Iterate through slices
                print(slc)
                image               = np.max(original_matrix_stacked[:, :, slc, :], axis = 2)                                                  #
                image_normalization = image / image.max()                                                                                      #
                # Select ROI
                title_pre = 'Select area for slice ' + str(slc + 1)                                                                            #
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
    ########## Identify high and low b-values #####################################################################################################
    (unique, index, counts) = np.unique(original_bvals_stacked, return_counts = True, return_index = True, axis = 0)                                                        # Extract number of different diffusion directions
    numBvals                = len(unique)                                                                                                                           # Identify number of b-values in stacked b-values
    bval_low                = min(unique)                                                                                                                           # Identify lower b-value (typically b = 0)
    bval_low_indicies       = np.where(original_bvals_stacked == bval_low)[0]                                                                                               # Identify index of lower b-value in stacked b-values
    bval_high               = max(unique)                                                                                                                           # Identify higher b-value
    bval_high_indicies      = np.where(original_bvals_stacked == bval_high)[0]                                                                                              # Identify indices of higher b-values in stacked b-values
    ########## Low b-value rejection ##############################################################################################################
    SSIM_bvl_v_bvl  = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize structure similarity index measure (SSIM) for low b-value comparison matrix
    NSSIM_bvl_v_bvl = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize normalized SSIM (NSSIM) for low b-value comparison matrix
    RMSE_bvl_v_bvl  = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize root mean squared error (RMSE) for low b-value comparison matrix
    NRMSE_bvl_v_bvl = np.zeros([len(bval_low_indicies), len(bval_low_indicies), slices])                                                                                # Initialize normalized root mean squared error (NRMSE) for low b-value
    ##### SSIM low b-value across averages #####
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        for avg_i in range(len(bval_low_indicies)):                                                                                                                                   # Iterate through averages (ith average)
            for avg_j in range(len(bval_low_indicies)):                                                                                                                                   # Iterate through averages (jth average)
                SSIM_bvl_v_bvl[avg_i, avg_j, slc] = ssim(original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_i]],
                                                         original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_j]])                           # Compute SSIM between ith and jth low b-value averages
                RMSE_bvl_v_bvl[avg_i, avg_j, slc] = np.sqrt(mse(original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_i]],
                                                                original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_low_indicies[avg_j]]))                   # Compute RMSE between ith and jth low b-value averages
        NSSIM_bvl_v_bvl[:, :, slc] = SSIM_bvl_v_bvl[:, :, slc] / SSIM_bvl_v_bvl[:, :, slc].max()                                          # Normalize SSIM across all averages
        NRMSE_bvl_v_bvl[:, :, slc] = RMSE_bvl_v_bvl[:, :, slc] / RMSE_bvl_v_bvl[:, :, slc].max()                                          # Normalize RMSE across all averages
        ###### ######
    NSSIM_AoA_pre = np.zeros((len(bval_low_indicies), slices))                                                                                            # Initialize NSSIM assesment of averages (AoA) for pre automatic acquisition rejection
    NRMSE_AoA_pre = np.zeros((len(bval_low_indicies), slices))                                                                                            # Initialize NRMSE assesment of averages (AoA) for pre automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        #for baseline in range(len(bval_low_indicies)):                                                                                                                  # Iterate through low b-values
        NSSIM_AoA_pre[:, slc] = np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0).max()         # Calcualte NSSIM AoA for pre automatic acquisition rejection
        NRMSE_AoA_pre[:, slc] = np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0).max()         # Calcualte NRMSE AoA for pre automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
            NSSIM_AoA_pre_list = NSSIM_AoA_pre[:, slc].tolist()                                                                                                   # Convert NSSIM AoA for pre automatic acquisition rejection to a list
            NRMSE_AoA_pre_list = NRMSE_AoA_pre[:, slc].tolist()                                                                                                   # Convert NRMSE AoA for pre automatic acquisition rejection to a list
            if  all(x >= NSSIM_threshold for x in NSSIM_AoA_pre_list) and all(x >= NRMSE_threshold for x in NRMSE_AoA_pre_list):                                            # If both lists are greater than or equal to respective thresholds ...
                continue                                                                                                                                                        # Continue on to next block of code
            else:                                                                                                                                                           # Otherwise ...
                for avg in range(len(NSSIM_AoA_pre_list)):                                                                                                                      # Iterate through averages
                    if NRMSE_AoA_pre_list[avg] > NRMSE_threshold and NSSIM_AoA_pre_list[avg] < NSSIM_threshold:                                                                     # If NRMSE AoA pre is greater than NRMSE threshold and NSSIM AoA pre is less than NSSIM threshold ...
                        NSSIM_bvl_v_bvl[avg,   :, slc] = 0                                                                                                                    # Set current average columns of NSSIM for low b-value comparison matrix to zero (automatically reject acquisition)
                        NSSIM_bvl_v_bvl[  :, avg, slc] = 0                                                                                                                    # Set current average rows of NSSIM for low b-value comparison matrix to zero (automatically reject acquisition)
                        NRMSE_bvl_v_bvl[avg,   :, slc] = 0                                                                                                                    # Set current average columns of NRMSE for low b-value comparison matrix to zero (automatically reject acquisition)
                        NRMSE_bvl_v_bvl[  :, avg, slc] = 0                                                                                                                    # Set current average rows of NRMSE for low b-value comparison matrix to zero (automatically reject acquisition)
    NSSIM_AoA_post = np.zeros((len(bval_low_indicies), slices))                                                                                           # Initialize NSSIM assesment of averages (AoA) for post automatic acquisition rejection
    NRMSE_AoA_post = np.zeros((len(bval_low_indicies), slices))                                                                                           # Initialize NRMSE assesment of averages (AoA) for post automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_post[:, slc] = np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NSSIM_bvl_v_bvl[:, :, slc], axis = 0).max()        # Calcualte NSSIM AoA for post automatic acquisition rejection
        NRMSE_AoA_post[:, slc] = np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0) / np.sum(NRMSE_bvl_v_bvl[:, :, slc], axis = 0).max()        # Calcualte NRMSE AoA for post automatic acquisition rejection
        if np.isnan(NRMSE_AoA_post[:, slc]).any():
            NRMSE_AoA_post[:, slc] = 1
    kmeans_labels   = np.zeros(NSSIM_AoA_post.shape)                                                                                                                # Initialize k means labels matrix
    keep_bvl_matrix = np.zeros(NSSIM_AoA_post.shape)

    k_means_cluster = []
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_post_list = NSSIM_AoA_post[:, slc].tolist()                                                                                                 # Convert NSSIM AoA for post automatic acquisition rejection to a list
        NRMSE_AoA_post_list = NRMSE_AoA_post[:, slc].tolist()                                                                                                 # Convert NRMSE AoA for post automatic acquisition rejection to a list
        data = np.hstack((NSSIM_AoA_post[:, slc, np.newaxis], NRMSE_AoA_post[:, slc, np.newaxis]))                                                  # Combine NSSIM AoA post and NRMSE AoA post into an [avg,2] shape
        tile_size = int(np.floor(30 / data.shape[0]))
        data_estimate = np.tile(data, (tile_size, 1))
        if data.shape[0] > 1:
            model = KMeans()
            visualizer = KElbowVisualizer(model, k = (1, data_estimate.shape[0] + 1), timings = False)
            visualizer.fit(data_estimate)        # Fit data to visualizer
            k_means_cluster.append(visualizer.elbow_value_)
            plt.title('K-Means Estimate (Low $\it{b-value}$) for Slice %i' %int(slc + 1))
            plt.legend(loc = 1)
            plt.show()
            NSSIM_AoA_post_list_temp = [i for i in NSSIM_AoA_post_list if i != 0]
            NRMSE_AoA_post_list_temp = [i for i in NRMSE_AoA_post_list if i != 0]
            if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list_temp) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list_temp):                                           # If both lists are greater than or equal to respective thresholds ...
                if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list):
                    k_mean_initialization        = np.array(([1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                    k_means                      = KMeans(n_clusters = 1, algorithm = 'elkan', n_init = 1).fit(data)                                                                # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                    keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] == 0).astype('int')                                                                                  # Store keep low b-values based off k-means labels
                else:
                    k_mean_initialization        = np.array(([0, 0], [1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                    k_means                      = KMeans(n_clusters = 2, algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                                                # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                    keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                         # Store keep low b-values based off k-means labels
            else:                                                                                                                                                           # Otherwise ...
                distance_max = 0
                for ii in range(len(NSSIM_AoA_post_list_temp)):
                    for jj in range(len(NRMSE_AoA_post_list_temp)):
                        distance = math.dist([NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]], [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]])
                        if distance > distance_max:
                            distance_max = distance
                            point_1_temp = [NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]]
                            point_2_temp = [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]]
                if point_1_temp[0] >  point_2_temp[0]:
                    point_1 = point_2_temp
                    point_2 = point_1_temp
                else:
                    point_1 = point_1_temp
                    point_2 = point_2_temp
                X = [point_1[0], point_2[0]]
                Y = [point_1[1], point_2[1]]
                a, b = np.polyfit(X, Y, 1)                                                                                                                                      # Calculate linear fit of X and Y data
                X_vals = np.linspace(min(X), max(X), k_means_cluster[slc])
                Y_vals = a * X_vals + b
                Fit_Centers = np.hstack((X_vals[np.newaxis, :].T, Y_vals[np.newaxis, :].T))
                if (0 in NSSIM_AoA_post_list and 0 in NRMSE_AoA_post_list):                                                                                                   # Else if automatic acquisition rejection is detected ...
                    Fit_Centers = np.insert(Fit_Centers, 0, [0, 0], axis = 0)
                    k_mean_initialization        = Fit_Centers                                                    # Initialize four cluster centroids (one for automatic acquisiton rejection)
                    k_means                      = KMeans(n_clusters = k_means_cluster[slc] + 1, algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                  # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]        = k_means.labels_                                                                                                                  # Store k-means labels
                    if len(np.unique(k_means_labels)) == 2:
                        keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
                    else:
                        keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] >= 2).astype('int')
                else:                                                                                                                                                           # Otherwise ...
                    k_mean_initialization        = Fit_Centers                                                              # Initialize three cluster centroids
                    k_means                      = KMeans(n_clusters = k_means_cluster[slc], algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                  # Run k-means algorithm
                    k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                    kmeans_labels[:, slc]        = k_means.labels_                                                                                                                  # Store k-means labels
                    keep_bvl_matrix[:, slc]      = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
        else:
            k_means_cluster.append(1)
            k_mean_initialization             = np.array(([1.0, 1.0]))                                                                                                      # Initalize only one cluster centroid
            k_means                           = KMeans(n_clusters = 1, algorithm = 'elkan', n_init = 1).fit(data)                                                           # Run k-means algorithm
            k_means_labels                    = k_means.labels_                                                                                                             # Extract k-means labels
            kmeans_labels[:, slc]   = k_means.labels_                                                                                                             # Store k-means labels
            keep_bvl_matrix[:, slc] = (kmeans_labels[:, slc] == 0).astype('int')
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        plt.scatter(NSSIM_AoA_post[:, slc], NRMSE_AoA_post[:, slc], c = kmeans_labels[:, slc], marker = 'o')

        plt.axhline(NRMSE_threshold, color = 'red', label = 'NRMSE Threshold')
        plt.axvline(NSSIM_threshold, color = 'blue', label = 'NSSIM Threshold')
        plt.title('K-Means Cluster Assignemnts (Low $\it{b-value}$) for Slice %i' %int(slc + 1))
        plt.plot(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], 'gx', label = 'Cluster Centers')
        plt.legend(loc = 8)
        plt.xlabel('NSSIM')
        plt.ylabel('NRMSE')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.show()
    ########## High b-value rejection ##############################################################################################################
    SSIM_bvh_v_bvh  = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize structure similarity index measure (SSIM) for low b-value comparison matrix
    NSSIM_bvh_v_bvh = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize normalized SSIM (NSSIM) for low b-value comparison matrix
    RMSE_bvh_v_bvh  = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize root mean squared error (RMSE) for low b-value comparison matrix
    NRMSE_bvh_v_bvh = np.zeros([len(bval_high_indicies), len(bval_high_indicies), slices])                                                                                # Initialize normalized root mean squared error (NRMSE) for low b-value
    ##### SSIM high b-value across all diffusion data #####
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        for avg_i in range(len(bval_high_indicies)):                                                                                                                                   # Iterate through averages (ith average)
            for avg_j in range(len(bval_high_indicies)):                                                                                                                                   # Iterate through averages (jth average)
                SSIM_bvh_v_bvh[avg_i, avg_j, slc] = ssim(original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_i]],
                                                         original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_j]])                           # Compute SSIM between ith and jth low b-value averages
                RMSE_bvh_v_bvh[avg_i, avg_j, slc] = np.sqrt(mse(original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_i]],
                                                                original_matrix_stacked[y_start[slc]:y_end[slc], x_start[slc]:x_end[slc], slc, bval_high_indicies[avg_j]]))                   # Compute RMSE between ith and jth low b-value averages
        NSSIM_bvh_v_bvh[:, :, slc] = SSIM_bvh_v_bvh[:, :, slc] / SSIM_bvh_v_bvh[:, :, slc].max()                                          # Normalize SSIM across all averages
        NRMSE_bvh_v_bvh[:, :, slc] = RMSE_bvh_v_bvh[:, :, slc] / RMSE_bvh_v_bvh[:, :, slc].max()                                          # Normalize RMSE across all averages
    ###### ######
    NSSIM_AoA_pre = np.zeros((len(bval_high_indicies), slices))                                                                                            # Initialize NSSIM assesment of averages (AoA) for pre automatic acquisition rejection
    NRMSE_AoA_pre = np.zeros((len(bval_high_indicies), slices))                                                                                            # Initialize NRMSE assesment of averages (AoA) for pre automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_pre[:, slc] = np.sum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0) / np.sum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0).max()         # Calcualte NSSIM AoA for pre automatic acquisition rejection
        NRMSE_AoA_pre[:, slc] = np.sum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0) / np.sum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0).max()         # Calcualte NRMSE AoA for pre automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_pre_list = NSSIM_AoA_pre[:, slc].tolist()                                                                                                      # Convert NSSIM AoA for pre automatic acquisition rejection to a list
        NRMSE_AoA_pre_list = NRMSE_AoA_pre[:, slc].tolist()                                                                                                      # Convert NSSIM AoA for pre automatic acquisition rejection to a list
        if  all(x >= NSSIM_threshold for x in NSSIM_AoA_pre_list) and all(x < NRMSE_threshold for x in NRMSE_AoA_pre_list):                                                # If both lists are greater than or equal to respective thresholds ...
            continue                                                                                                                                                        # Continue on to next block of code
        else:                                                                                                                                                           # Otherwise ...
            for avg in range(len(NSSIM_AoA_pre_list)):                                                                                                                      # Iterate through averages
                if NRMSE_AoA_pre_list[avg] >= NRMSE_threshold and NSSIM_AoA_pre_list[avg] < NSSIM_threshold:                                                                         # If NMI AoA pre is greater than NMI threshold and NSSIM AoA pre is less than NSSIM threshold ...
                    NSSIM_bvh_v_bvh[avg,   :, slc] = np.nan                                                                                                                         # Set current average columns of NSSIM for high b-value comparison matrix to zero (automatically reject acquisition)
                    NSSIM_bvh_v_bvh[  :, avg, slc] = np.nan                                                                                                                         # Set current average rows of NSSIM for high b-value comparison matrix to zero (automatically reject acquisition)
                    NRMSE_bvh_v_bvh[avg,   :, slc] = np.nan                                                                                                                         # Set current average columns of NRMSE for high b-value comparison matrix to zero (automatically reject acquisition)
                    NRMSE_bvh_v_bvh[  :, avg, slc] = np.nan                                                                                                                         # Set current average rows of NRMSE for high b-value comparison matrix to zero (automatically reject acquisition)
            NRMSE_AoA_pre_list = NRMSE_bvh_v_bvh[:, :, slc].tolist()
            NRMSE_AoA_pre_list = list(itertools.chain.from_iterable(NRMSE_AoA_pre_list))
            if  sum(NRMSE_AoA_pre_list) == 0:
                for avg in range(len(NSSIM_AoA_pre_list)):
                    if  NSSIM_bvh_v_bvh[avg, avg, dif, slc] == 1:
                        NRMSE_bvh_v_bvh[avg, avg, dif, slc] = 1
    NSSIM_AoA_post = np.zeros((len(bval_high_indicies), slices))                                                                                          # Initialize NSSIM AoA for post automatic acquisition rejection
    NRMSE_AoA_post = np.zeros((len(bval_high_indicies), slices))                                                                                          # Initialize NRMSE AoA for post automatic acquisition rejection
    for slc in range(slices):                                                                                                                                       # Iterate through slices
        for dif in range(len(bval_high_indicies)):                                                                                                                      # Iterate through high b-values
            NSSIM_AoA_post[:, slc] = np.nansum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0) / np.nansum(NSSIM_bvh_v_bvh[:, :, slc], axis = 0).max()                       # Calcualte NSSIM AoA for post automatic acquisition rejection
            NRMSE_AoA_post[:, slc] = np.nansum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0) / np.nansum(NRMSE_bvh_v_bvh[:, :, slc], axis = 0).max()                       # Calcualte NRMSE AoA for post automatic acquisition rejection
    kmeans_labels   = np.zeros(NSSIM_AoA_post.shape)                                                                                                                # Initialize k means labels matrix
    keep_bvh_matrix = np.zeros(NSSIM_AoA_post.shape)
    k_means_cluster = []

    for slc in range(slices):                                                                                                                                       # Iterate through slices
        NSSIM_AoA_post_list = NSSIM_AoA_post[:, slc].tolist()                                                                                                 # Convert NSSIM AoA for post automatic acquisition rejection to a list
        NRMSE_AoA_post_list = NRMSE_AoA_post[:, slc].tolist()                                                                                                 # Convert NRMSE AoA for post automatic acquisition rejection to a list
        data = np.hstack((NSSIM_AoA_post[:, slc, np.newaxis], NRMSE_AoA_post[:, slc, np.newaxis]))                                                  # Combine NSSIM AoA post and NRMSE AoA post into an [avg,2] shape
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1, len(NSSIM_AoA_post_list) + 1), timings = False)
        visualizer.fit(data)        # Fit data to visualizer
        if visualizer.elbow_value_ < 6:
            k_means_cluster.append(6)
            print('Overridding estimated number of clusters to be 5 clusters.')
        else:
            k_means_cluster.append(visualizer.elbow_value_)
        plt.title('K-Means Estimate (High $\it{b-value}$) for Slice %i' %int(slc + 1))
        plt.legend(loc = 1)
        plt.show()
        NSSIM_AoA_post_list_temp = [i for i in NSSIM_AoA_post_list if i != 0]
        NRMSE_AoA_post_list_temp = [i for i in NRMSE_AoA_post_list if i != 0]
        if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list_temp) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list_temp):                                           # If both lists are greater than or equal to respective thresholds ...
            if all(x >= NSSIM_threshold for x in NSSIM_AoA_post_list) and all(x >= NRMSE_threshold for x in NRMSE_AoA_post_list):
                k_mean_initialization        = np.array(([1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                k_means                      = KMeans(n_clusters = 1, algorithm = 'elkan', n_init = 1).fit(data)                                                                # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] == 0).astype('int')                                                                                  # Store keep low b-values based off k-means labels
            else:
                k_mean_initialization        = np.array(([0, 0], [1.0, 1.0]))                                                                                                           # Initalize only one cluster centroid
                k_means                      = KMeans(n_clusters = 2, algorithm = 'elkan', init = k_mean_initialization, n_init = 1).fit(data)                                                                # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
        else:                                                                                                                                                           # Otherwise ...
            distance_max = 0
            for ii in range(len(NSSIM_AoA_post_list_temp)):
                for jj in range(len(NRMSE_AoA_post_list_temp)):
                    distance = math.dist([NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]], [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]])
                    if distance > distance_max:
                        distance_max = distance
                        point_1_temp = [NSSIM_AoA_post_list_temp[ii], NRMSE_AoA_post_list_temp[ii]]
                        point_2_temp = [NSSIM_AoA_post_list_temp[jj], NRMSE_AoA_post_list_temp[jj]]
            if point_1_temp[0] >  point_2_temp[0]:
                point_1 = point_2_temp
                point_2 = point_1_temp
            else:
                point_1 = point_1_temp
                point_2 = point_2_temp
            X = [point_1[0], point_2[0]]
            Y = [point_1[1], point_2[1]]
            a, b = np.polyfit(X, Y, 1)                                                                                                                                      # Calculate linear fit of X and Y data
            X_vals = np.linspace(min(X), max(X), k_means_cluster[slc])
            Y_vals = a * X_vals + b
            Fit_Centers = np.hstack((X_vals[np.newaxis, :].T, Y_vals[np.newaxis, :].T))
            if (0 in NSSIM_AoA_post_list and 0 in NRMSE_AoA_post_list):                                                                                                   # Else if automatic acquisition rejection is detected ...
                Fit_Centers = np.insert(Fit_Centers, 0, [0, 0], axis = 0)
                k_mean_initialization        = Fit_Centers                                                  # Initialize four cluster centroids (one for automatic acquisiton rejection)
                k_means                      = KMeans(n_clusters = k_means_cluster[slc] + 1, algorithm = 'elkan', init = k_mean_initialization, n_init = 1, verbose = 1).fit(data)                                  # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                if len(np.unique(k_means_labels)) == 2:
                    keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
                else:
                    keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 2).astype('int')
            else:                                                                                                                                                           # Otherwise ...
                k_mean_initialization        = Fit_Centers                                                                   # Initialize three cluster centroids
                print(Fit_Centers)
                k_means                      = KMeans(n_clusters = k_means_cluster[slc], algorithm = 'elkan', init = k_mean_initialization, n_init = 1, verbose = 1).fit(data)                                  # Run k-means algorithm
                k_means_labels               = k_means.labels_                                                                                                                  # Extract k-means labels
                kmeans_labels[:, slc]   = k_means.labels_                                                                                                                  # Store k-means labels
                keep_bvh_matrix[:, slc] = (kmeans_labels[:, slc] >= 1).astype('int')                                                                                  # Store keep low b-values based off k-means labels
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        plt.scatter(NSSIM_AoA_post[:, slc], NRMSE_AoA_post[:, slc], c = kmeans_labels[:, slc], marker = 'o')
        plt.axhline(NRMSE_threshold, color = 'red', label = 'NRMSE Threshold')
        plt.axvline(NSSIM_threshold, color = 'blue', label = 'NSSIM Threshold')
        plt.title('K-Means Cluster Assignemnts (High $\it{b-value}$) for Slice %i' %int(slc + 1))
        plt.plot(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], 'gx', label = 'Cluster Centers')
        plt.legend(loc=8)
        plt.xlabel('NSSIM')
        plt.ylabel('NRMSE')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.show()
    keep_matrix_stacked = np.zeros([original_matrix_stacked.shape[2], original_matrix_stacked.shape[3]])
    for slc in range(keep_bvl_matrix.shape[1]):
        for acquisition in range(keep_bvl_matrix.shape[0]):
            keep_matrix_stacked[slc, bval_low_indicies[acquisition]] = keep_bvl_matrix[acquisition, slc]
    for slc in range(keep_bvh_matrix.shape[1]):
        for acquisition in range(keep_bvh_matrix.shape[0]):
            keep_matrix_stacked[slc, bval_high_indicies[acquisition]] = keep_bvh_matrix[acquisition, slc]
    ########## Identify new matrix dimensions ######################################################################################################
    numRow                  = original_matrix_stacked.shape[0]                                                                   # Define number of rows (x)
    numCol                  = original_matrix_stacked.shape[1]                                                                   # Define number of columns (y)
    numSlc                  = original_matrix_stacked.shape[2]                                                                   # Define number of slices (z)
    (unique, index, counts) = np.unique(original_bvecs_stacked, return_counts = True, return_index = True, axis = 0)             # Extract number of different diffusion directions
    unique                  = unique[np.argsort(index)]                                                                 # Retain stacked diffusion direction order
    numDif                  = len(unique)                                                                               # Define number of diffusion directions
    numAvg                  = max(counts)                                                                               # Define number of averages
    keep_matrix_sorted = np.zeros([numSlc, numDif, numAvg])
    stacked_index_list = []                                                                                        # Initialize stacked index list
    for dif in range(numDif):                                                                                           # Iterate through diffusion directions
        idxAvg = -1                                                                                                         # Initialize average index
        for idx in range(keep_matrix_stacked.shape[1]):                                                                          # Iterate through stacked diffusion dimension
            current_search = unique[dif]                                                                                        # Define current diffusion direction
            if np.array_equal(current_search, original_bvecs_stacked[idx, :]) == True:                                                   # Check if current diff dir matches desired diff dir ...
                idxAvg                                 = idxAvg + 1                                                                 # Index current average
                idxDif                                 = np.where(np.all(unique == original_bvecs_stacked[idx, :],axis = 1))[0][0]           # Index current diffuion direction
                keep_matrix_sorted[:, idxDif, idxAvg]  = keep_matrix_stacked[:, idx]                                               # Store sorted matrix from current index of
    stats = np.zeros([keep_matrix_sorted.shape[0], keep_matrix_sorted.shape[1]])
    for slc in range(stats.shape[0]):
        for dif in range(stats.shape[1]):
            stats[slc, dif] = (np.sum(keep_matrix_sorted[slc, dif, :]) / keep_matrix_sorted.shape[2]) * 100
            print('Acceptance rate for slice %i, direction %i: %.1f%%' %(int(slc + 1), int(dif + 1), stats[slc, dif]))
    accepted_matrix = np.zeros(original_matrix.shape)
    accepted_bvals  = []
    accepted_bvecs  = []
    averages = keep_matrix_sorted.shape[2]
    if keep_matrix_sorted.shape[2] > 1:
        accepted_bvals  = original_bvals
        accepted_bvecs  = original_bvecs
        for slc in range(keep_matrix_sorted.shape[0]):
            for dif in range(keep_matrix_sorted.shape[1]):
                all_averages   = np.arange(0, averages)                                                                                                                         # Define indices all averages
                bad_averages   = np.where((keep_matrix_sorted[slc, dif, :] == 0))[0]                                                                                          # Identify indices of bad averages
                good_averages  = np.delete(all_averages, bad_averages)                                                                                                          # Identify indices of good averages
                final_averages = all_averages
                for idx in range(len(bad_averages)):                                                                                                                            # Iterate through indices of bad averages
                    replacement_average               = np.random.choice(good_averages, 1)                                                                                          # Select random replacement index from indices of good averages
                    final_averages[bad_averages[idx]] = replacement_average                                                                                                         # Overwrite bad average index with random replacement average index
                for avg in range(averages):                                                                                                                                     # Iterate through averages
                    accepted_matrix[:, :, slc, dif, avg]  = original_matrix[:, :, slc, dif, final_averages[avg]]                    # Overwrite bad average data with replacement average data in accepted matrix
    else:
        print('Only 1 average detected.')
        #print('Output will put each value in a list with each slice nested within the respective list.')
#        accepted_matrix = []
#        for slc in range(keep_matrix_sorted.shape[0]):
#            accepted_bvals.append([])
#            accepted_bvecs.append([])
#            matrix_list = []
#            for dif in range(keep_matrix_sorted.shape[1]):
#                if keep_matrix_sorted[slc, dif] == 1:
#                    temp_image = original_matrix[:, :, slc, dif, 0]
#                    temp_image = temp_image[:, :, np.newaxis]
#                    matrix_list.append(temp_image)
#                    accepted_bvals[slc].append(original_bvals[dif])
#                    accepted_bvecs[slc].append(original_bvecs[dif])
#                else:
#
#                    if dif == 0:
#                        print('On slice %i, direction %i was rejected, but non-diffusion weighted images cannot be removed.')
#                        temp_image = original_matrix[:, :, slc, dif, 0]
#                        temp_image = temp_image[:, :, np.newaxis]
#                        matrix_list.append(temp_image)
#                        accepted_bvals[slc].append(original_bvals[dif])
#                        accepted_bvecs[slc].append(original_bvecs[dif])
#                    else:
#                        print('On slice %i, direction %i was rejected and removed from dataset.' %(int(slc), int(dif)))
#            accepted_matrix.append(np.concatenate(matrix_list, axis = 2))
        for slc in range(keep_matrix_sorted.shape[0]):
            for dif in range(keep_matrix_sorted.shape[1]):
                if keep_matrix_sorted[slc, dif] == 1:
                    accepted_matrix[:, :, slc, dif, 0] = original_matrix[:, :, slc, dif, 0]
                else:

                    if dif == 0:
                        print('On slice %i, direction %i was rejected, but non-diffusion weighted images cannot be removed.')
                        accepted_matrix[:, :, slc, dif, 0] = original_matrix[:, :, slc, dif, 0]
                    else:
                        print('On slice %i, direction %i was rejected and replaced with NaN values.' %(int(slc), int(dif)))
#                        temp_image    = np.empty([original_matrix.shape[0], original_matrix.shape[1]])
#                        temp_image[:] = 0
#                        accepted_matrix[:, :, slc, dif, 0] = temp_image
                        temp_image        = original_matrix[:, :, slc, dif, 0]
                        temp_image_max    = temp_image.max()
                        temp_image_offset = temp_image - temp_image_max
                        accepted_matrix[:, :, slc, dif, 0] = temp_image_offset
        accepted_bvals  = original_bvals
        accepted_bvecs  = original_bvecs
    return[accepted_matrix, accepted_bvals, accepted_bvecs, Slice_Crop_Coortinates]
