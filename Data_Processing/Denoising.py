def denoise(original_matrix, original_bvals, original_bvecs, denoising_algorithm = 'LocalPCA', numCoils = 20, operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values.
    original_bvecs          : Sorted b-vectors.
    denoising_algorithm     : Select type of denoising to perform (Recommended).
                              Default denoising algorithm is local PCA.
                              Denoising algorithm options include Patch2Self, Non-Local Means, and Local PCA.
    numCoils                : Number of coils used to aquire image data (Recommended if using Non-Local Means).
                              This is only used for the Non-Local Means denoising algorithm.
                              Default number of coils is 20.
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    denoised_matrix         : Sorted denoised diffusion data (5D - [rows, columns, slices, directions, averages]).
    denoised_bvals          : Sorted b-values.
    denoised_bvecs          : Sorted b-vectors.
    ########## References #########################################################################################################################
    Please cite the following articles if using Patch2Self denoising:
    [1] Fadnavis S, Chowdhury A, Batson J, Drineas P, Garyfallidis E.
        Patch2self denoising of diffusion MRI with self-supervision and matrix sketching.
        Information Processing Systems 2022;33:16293–303.
        doi: 10.1101/2022.03.15.484539

    Please cite the following articles if using Non-Local Means denoising:
    [2] Descoteaux M, Wiest-Daesslé N, Prima S, Barillot C, Deriche R.
        Impact of Rician adapted non-local means filtering on Hardi.
        Medical Image Computing and Computer-Assisted Intervention – MICCAI 2008 2008:122–30.
        doi: 10.1007/978-3-540-85990-1_15
    
    Please cite the following articles if using Local PCA denoising:
    [3] Manjón JV, Coupé P, Concha L, Buades A, Collins DL, Robles M.
        Diffusion weighted image denoising using overcomplete local PCA.
        PLoS ONE 2013;8.
        doi: 10.1371/journal.pone.0073021
    [4] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers J, Fieremans E.
        Denoising of diffusion MRI using random matrix theory.
        NeuroImage 2016;142:394–406.
        doi: 10.1016/j.neuroimage.2016.08.016
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    import numpy                           as     np                                                                                # Import numpy module
    from   cardpy.Data_Sorting             import sorted2stacked, stacked2sorted                                                    # Import sorted to stacked and stacked to sorted from CarDpy
    from   dipy.denoise.patch2self         import patch2self                                                                        # Import patch to self-denoising from DiPy
    from   dipy.denoise.noise_estimate     import estimate_sigma                                                                    # Import non-local means denoising sigma estimate from DiPy
    from   dipy.denoise.nlmeans            import nlmeans                                                                           # Import non-local means denoising from DiPy
    from   dipy.core.gradients             import gradient_table                                                                    # Import gradient table from DiPy
    from   dipy.denoise.pca_noise_estimate import pca_noise_estimate                                                                # Import local PCA denoising noise estimate from DiPy
    from   dipy.denoise.localpca           import localpca                                                                          # Import local PCA denoising from DiPy
    ########## Address data type of stacked matrix ################################################################################################
    [stacked_matrix, stacked_bvals, stacked_bvecs] = sorted2stacked(original_matrix, original_bvals, original_bvecs)                # Convert sorted data into stacked data
    if stacked_matrix.dtype == 'complex128':                                                                                        # If data type is complex ...
        if operation_type == 'Complex':                                                                                                 # If operation type is complex ...
            real_matrix     = np.real(stacked_matrix)                                                                                       # Separate real part from complex data
            imag_matrix     = np.imag(stacked_matrix)                                                                                       # Separate imaginary part from complex data
            denoised_matrix = np.zeros(stacked_matrix.shape)                                                                                # Initialize denoised matrix
            denoised_matrix = denoised_matrix.astype(np.complex128)                                                                         # Cast denoised matrix for complex data
        if operation_type == 'Magnitude':                                                                                               # If operation type is magnitude ...
            stacked_matrix  = np.abs(stacked_matrix)                                                                                        # Convert complex original matrix to magnitude data
            denoised_matrix = np.zeros(stacked_matrix.shape)                                                                                # Initialize denoised matrix
            denoised_matrix = denoised_matrix.astype(np.float64)                                                                            # Cast denoised matrix for magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                                           # Print warning
    else:                                                                                                                           # Otherwise ...
        denoised_matrix = np.zeros(stacked_matrix.shape)                                                                                # Initialize denoised matrix
        denoised_matrix = denoised_matrix.astype(np.float64)                                                                            # Cast denoised matrix for magnitude data
    ########## Self2Patch denoising ###############################################################################################################
    if denoising_algorithm == 'Patch2Self':                                                                                         # If Patch2Self is selected ...
        print('Denoising images using Patch2Self.')                                                                                     # Print which algorithm is selected
        if operation_type == 'Complex':                                                                                                 # If data type is complex ...
            ########## Real matrix operations ##########
            denoised_real_matrix = patch2self(real_matrix, stacked_bvals)                                                                   # Store denoised real matrix
            ########## Imaginary matrix operations ##########
            denoised_imag_matrix = patch2self(imag_matrix, stacked_bvals)                                                                   # Store denoised imaginary matrix
            ########## Combine real and imaginary data ##########
            denoised_matrix      = denoised_real_matrix + (1j * denoised_imag_matrix)                                                       # Store denoised complex matrix
        if operation_type == 'Magnitude':                                                                                               # If data type is magnitude ...
            denoised_matrix      = patch2self(stacked_matrix, stacked_bvals)                                                                # Store denoised matrix
    ########## Non-local means denoising ##########################################################################################################
    if denoising_algorithm == 'NLMeans':                                                                                            # If Non-Local Means is selected ...
        print('Denoising images using Non-Local Means (NLMEANS).')                                                                      # Print which algorithm is selected
        if operation_type == 'Complex':                                                                                                 # If data type is complex ...
            ########## Real matrix operations ##########
            sigma_real           = estimate_sigma(real_matrix, N = numCoils)                                                                # Set sigma for Non-Local Means denoising using real matrix
            denoised_real_matrix = nlmeans(real_matrix, sigma = sigma_real, patch_radius = 1, block_radius = 2, rician = False)             # Store denoised real matrix (rician = False ==> Gaussian noise)
            ########## Imaginary matrix operations ##########
            sigma_imag           = estimate_sigma(imag_matrix, N = numCoils)                                                                # Set sigma for Non-Local Means denoising using imaginary matrix
            denoised_imag_matrix = nlmeans(imag_matrix, sigma = sigma_imag, patch_radius = 1, block_radius = 2, rician = False)             # Store denoised imaginary matrix (rician = False ==> Gaussian noise)
            ########## Combine real and imaginary data ##########
            denoised_matrix      = denoised_real_matrix + (1j * denoised_imag_matrix)                                                       # Store denoised complex matrix
        if operation_type == 'Magnitude':                                                                                                # If data type is magnitude ...
            sigma_mag            = estimate_sigma(stacked_matrix, N = numCoils)                                                             # Set sigma for Non-Local Means denoising
            denoised_matrix      = nlmeans(stacked_matrix, sigma = sigma_mag, patch_radius = 1, block_radius = 2, rician = True)            # Store denoised matrix (rician = True ==> Rician noise)
    ########## Local PCA denoising ################################################################################################################
    if denoising_algorithm == 'LocalPCA':                                                                                           # If Local PCA is selected ...
        print('Denoising images using Local PCA via empirical thresholds.')                                                             # Print which algorithm is selected
        gtab   = gradient_table(stacked_bvals, stacked_bvecs)                                                                           # Create gradient table from b-values and b-vectors
        slices = stacked_matrix.shape[2]                                                                                                # Extract number of slices
        if operation_type == 'Complex':                                                                                                 # If data type is complex ...
            for slc in range(slices):                                                                                                       # Iterate through slices
                ########## Real matrix operations ##########
                tmp_real_matrix   = real_matrix[:, :, slc, :]                                                                                   # Select slice you want to denoise
                tmp_real_matrix   = tmp_real_matrix[:, :, np.newaxis, :]                                                                        # Insert new axis as real matrix has compressed
                tmp_real_matrix   = np.repeat(tmp_real_matrix, 5, axis = 2)                                                                     # Pad array with repeats of the selected slice
                sigma_real        = pca_noise_estimate(tmp_real_matrix, gtab, correct_bias = True, smooth = 2)                                  # Set sigma for local PCA denoising using real matrix
                tmp_real_denoised = localpca(tmp_real_matrix, sigma, tau_factor = 2.3, patch_radius = 2)                                        # Run Local PCA for real matrix
                ########## Imaginary matrix operations ##########
                tmp_imag_matrix   = imag_matrix[:, :, slc, :]                                                                                   # Select slice you want to denoise
                tmp_imag_matrix   = tmp_imag_matrix[:, :, np.newaxis, :]                                                                        # Insert new axis as imaginary matrix has compressed
                tmp_imag_matrix   = np.repeat(tmp_imag_matrix, 5, axis = 2)                                                                     # Pad array with repeats of the selected slice
                sigma_imag        = pca_noise_estimate(tmp_imag_matrix, gtab, correct_bias = True, smooth = 2)                                  # Set sigma for local PCA denoising using imaginary matrix
                tmp_imag_denoised = localpca(tmp_imag_matrix, sigma, tau_factor = 2.3, patch_radius = 2)                                        # Run Local PCA for imaginary matrix
                ########## Combine real and imaginary data ##########
                denoised_matrix[:, :, slc, :] = tmp_real_denoised[:, :, 2, :] + (1j * tmp_imag_denoised[:, :, 2, :] )                           # Store selected denoised complex slice
        if operation_type == 'Magnitude':                                                                                               # If data type is magnitude ...
            for slc in range(slices):                                                                                                       # Iterate through slices
                tmp_matrix   = stacked_matrix[:, :, slc, :]                                                                                     # Select slice you want to denoise
                tmp_matrix   = tmp_matrix[:, :, np.newaxis, :]                                                                                  # Insert new axis as magnitude matrix has compressed
                tmp_matrix   = np.repeat(tmp_matrix, 5, axis = 2)                                                                               # Pad array with repeats of the selected slice
                sigma_mag    = pca_noise_estimate(tmp_matrix, gtab, correct_bias = True, smooth = 2)                                            # Set sigma for local PCA denoising using magnitude matrix
                tmp_denoised = localpca(tmp_matrix, sigma_mag, tau_factor = 2.3, patch_radius = 2)                                              # Run Local PCA for magnitude data
                denoised_matrix[:, :, slc, :] = tmp_denoised[:, :, 2, :]                                                                        # Store selected denoised magnitude slice
    ########## Tie-up-loose-ends ... ##############################################################################################################
    [denoised_matrix, _, _] = stacked2sorted(denoised_matrix, stacked_bvals, stacked_bvecs)                                         # Convert stacked data into sorted data
    denoised_bvals          = original_bvals                                                                                        # Store b-value information
    denoised_bvecs          = original_bvecs                                                                                        # Store b-vector information
    return [denoised_matrix, denoised_bvals, denoised_bvecs]
