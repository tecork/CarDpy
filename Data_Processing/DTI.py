def DTI_recon(original_matrix, original_bvals, original_bvecs, tensor_fit = 'NLLS'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values.
    original_bvecs          : Sorted b-vectors.
    tensor_fit              : Diffusion Tensor fit model. Default is non-linear least squares (NLLS).
                              Options included can be seen below.
                              - Ordinary least-squares (OLS)
                              - Weighted least squeares (WLS)
                              - Non-linear least squares (NLLS)
    ########## Definition Outputs #################################################################################################################
    Tensor                  : Diffusion tensor
    Eigenvalues             : Python dictionary containing all of the eigenvalues, which can be seen directly below.
                              - Eigenvalue 1 (λ 1)
                              - Eigenvalue 2 (λ 2)
                              - Eigenvalue 3 (λ 3)
    Eigenvectors            : Python dictionary containing all of the eigenvectors, which can be seen directly below.
                              - Eigenvector 1 (v 1)
                              - Eigenvector 2 (v 2)
                              - Eigenvector 3 (v 3)
    Standard_DTI_Metrics    : Python dictionary containing all of the standard DTI quantitative metrics, which can be seen directly below.
                              - Mean diffusivity (MD) [scaled to mm^2 / μs]
                              - Trace (TR) [scaled to mm^2 / μs]
                              - Fractional anisotropy (FA)
                              - Mode (MO)
                              - Axial diffusivity (AD) [scaled to mm^2 / μs]
                              - Radial diffusivity (RD) [scaled to mm^2 / μs]
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    import numpy                         as np                                                                          # Import numpy module
    from   cardpy.Data_Sorting           import sorted2stacked, stacked2sorted                                          # Import sorted to stacked and stacked to sorted from CarDpy
    from   dipy.core.gradients           import gradient_table                                                          # Import gradient table from DiPy
    import dipy.reconst.dti              as dti                                                                         # Import DTI reconstruction from DiPy
    ########## Address data type of original data #################################################################################################
    if original_matrix.dtype == 'complex128':                                                                           # If data type is complex ...
            original_matrix = np.abs(original_matrix)                                                                       # Convert data to magnitude
    ########## Convert sorted data into stacked data ##############################################################################################
    [stacked_matrix, stacked_bvals, stacked_bvecs] = sorted2stacked(original_matrix, original_bvals, original_bvecs)    # Convert sorted data into stacked data
    slices     = stacked_matrix.shape[2]
    directions = stacked_matrix.shape[3]
    matrix_slices_list = []
    bvals_slices_list  = []
    bvecs_slices_list  = []
    for slc in range(slices):
        bvals_slices_list.append([])
        bvecs_slices_list.append([])
        matrix_list = []
        for dif in range(directions):
            if np.sum(stacked_matrix[:, :, slc, dif]) > 0:
                temp_image = stacked_matrix[:, :, slc, dif]
                temp_image = temp_image[:, :, np.newaxis]
                matrix_list.append(temp_image)
                bvals_slices_list[slc].append(stacked_bvals[dif])
                bvecs_slices_list[slc].append(stacked_bvecs[dif])
        matrix_slices_list.append(np.concatenate(matrix_list, axis = 2))
    Tensor_list                = []
    eigenvalue_1_list          = []
    eigenvalue_2_list          = []
    eigenvalue_3_list          = []
    eigenvector_1_list         = []
    eigenvector_2_list         = []
    eigenvector_3_list         = []
    mean_diffusivity_list      = []
    fractional_anisotropy_list = []
    trace_list                 = []
    mode_list                  = []
    axial_diffusivity_list     = []
    radial_diffusivity_list    = []
    for slc in range(slices):
        ########## Model, fit, and derive diffusion tensor ############################################################################################
        gtable   = gradient_table(bvals_slices_list[slc], bvecs_slices_list[slc])                                           # Create gradient table from b-values and b-vectors
        tenmodel = dti.TensorModel(gtab = gtable, fit_method = tensor_fit)                                                  # Create tensor model from gradient table and tensor fit
        tenfit   = tenmodel.fit(matrix_slices_list[slc])                                                                    # Fit diffusion data to tensor model
        Tensor   = tenfit.quadratic_form                                                                                    # Extract tensor from fitted tensor model
        ########## Extract eigenvalues and eigenvectors ################################################################################################
        [eigenvalues, eigenvectors] = dti.decompose_tensor(Tensor)                                                          # Decompose tensor into Eigenvalues and Eigenvectors
        eigenvalue_1                = eigenvalues[:, :, np.newaxis, 0]                                                      # Extract eigenvalue 1 from tensor fit
        eigenvalue_2                = eigenvalues[:, :, np.newaxis, 1]                                                      # Extract eigenvalue 2 from tensor fit
        eigenvalue_3                = eigenvalues[:, :, np.newaxis, 2]                                                      # Extract eigenvalue 3 from tensor fit
        eigenvector_1               = eigenvectors[:, :, np.newaxis, :, 0]                                                  # Extract eigenvector 1 from tensor fit
        eigenvector_2               = eigenvectors[:, :, np.newaxis, :, 1]                                                  # Extract eigenvector 2 from tensor fit
        eigenvector_3               = eigenvectors[:, :, np.newaxis, :, 2]                                                  # Extract eigenvector 3 from tensor fit
        ########## Extract mean diffusivity, fractional anisotropy, trace, mode, axial diffusivity, and radial diffusivity #############################
        mean_diffusivity                                       = dti.mean_diffusivity(tenfit.evals) * 1000                  # Extract mean diffusivity and scale to mm^2 / μs
        mean_diffusivity                                       = np.clip(mean_diffusivity, 0, 4)                            # Clip mean diffusivity values to be between 0 and 4
        fractional_anisotropy                                  = dti.fractional_anisotropy(tenfit.evals)                    # Extract fractional anisotropy
        fractional_anisotropy[np.isnan(fractional_anisotropy)] = 0                                                          # Make NaN fractional anisotropy set to 0
        fractional_anisotropy                                  = np.clip(fractional_anisotropy, 0, 1)                       # Clip fractional anisotropy values to be between 0 and 1
        trace                                                  = dti.trace(tenfit.evals) * 1000                             # Extract trace and scale to mm^2 / μs
        trace                                                  = np.clip(trace, 0, 12)                                      # Clip trace values to be between 0 and 12
        mode                                                   = dti.mode(Tensor)                                           # Extract mode
        mode                                                   = np.clip(mode, -1, 1)                                       # Clip mode values to be between -1 and 1
        axial_diffusivity                                      = dti.axial_diffusivity(tenfit.evals) * 1000                 # Extract axial diffusivity and scale to mm^2 / μs
        radial_diffusivity                                     = dti.radial_diffusivity(tenfit.evals) * 1000                # Extract radial diffusivity and scale to mm^2 / μs
        Tensor_list.append(Tensor[:, :, np.newaxis, :, :])
        eigenvalue_1_list.append(eigenvalue_1)
        eigenvalue_2_list.append(eigenvalue_2)
        eigenvalue_3_list.append(eigenvalue_3)
        eigenvector_1_list.append(eigenvector_1)
        eigenvector_2_list.append(eigenvector_2)
        eigenvector_3_list.append(eigenvector_3)
        mean_diffusivity_list.append(mean_diffusivity[:, :, np.newaxis])
        fractional_anisotropy_list.append(fractional_anisotropy[:, :, np.newaxis])
        trace_list.append(trace[:, :, np.newaxis])
        mode_list.append(mode[:, :, np.newaxis])
        axial_diffusivity_list.append(axial_diffusivity[:, :, np.newaxis])
        radial_diffusivity_list.append(radial_diffusivity[:, :, np.newaxis])
    Tensor                = np.concatenate(Tensor_list, axis = 2)
    eigenvalue_1          = np.concatenate(eigenvalue_1_list, axis = 2)
    eigenvalue_2          = np.concatenate(eigenvalue_2_list, axis = 2)
    eigenvalue_3          = np.concatenate(eigenvalue_3_list, axis = 2)
    eigenvector_1         = np.concatenate(eigenvector_1_list, axis = 2)
    eigenvector_2         = np.concatenate(eigenvector_2_list, axis = 2)
    eigenvector_3         = np.concatenate(eigenvector_3_list, axis = 2)
    mean_diffusivity      = np.concatenate(mean_diffusivity_list, axis = 2)
    fractional_anisotropy = np.concatenate(fractional_anisotropy_list, axis = 2)
    trace                 = np.concatenate(trace_list, axis = 2)
    mode                  = np.concatenate(mode_list, axis = 2)
    axial_diffusivity     = np.concatenate(axial_diffusivity_list, axis = 2)
    radial_diffusivity    = np.concatenate(radial_diffusivity_list, axis = 2)
    ########## Store standard DTI metrics in a dictionary ##########################################################################################
    Standard_DTI_Metrics = dict()                                                                                       # Initialize DTI metrics dictionary
    Standard_DTI_Metrics['MD'] = mean_diffusivity                                                                       # Store mean diffusivity in DTI metrics dictionary
    Standard_DTI_Metrics['TR'] = trace                                                                                  # Store trace in DTI metrics dictionary
    Standard_DTI_Metrics['FA'] = fractional_anisotropy                                                                  # Store fractional anisotropy in DTI metrics dictionary
    Standard_DTI_Metrics['MO'] = mode                                                                                   # Store mode in DTI metrics dictionary
    Standard_DTI_Metrics['AD'] = axial_diffusivity                                                                      # Store axial diffusivity in DTI metrics dictionary
    Standard_DTI_Metrics['RD'] = radial_diffusivity                                                                     # Store radial diffusivity in DTI metrics dictionary
    Eigenvalues = dict()                                                                                                # Initialize eigenvalues dictionary
    Eigenvalues['Lambda 1'] = eigenvalue_1                                                                              # Store eigenvalue 1 in eigenvalues dictionary
    Eigenvalues['Lambda 2'] = eigenvalue_2                                                                              # Store eigenvalue 2 in eigenvalues dictionary
    Eigenvalues['Lambda 3'] = eigenvalue_3                                                                              # Store eigenvalue 3 in eigenvalues dictionary
    Eigenvectors = dict()                                                                                               # Initialize eigenvectors dictionary
    Eigenvectors['E1'] = eigenvector_1                                                                                  # Store eigenvector 1 in eigenvalues dictionary
    Eigenvectors['E2'] = eigenvector_2                                                                                  # Store eigenvector 2 in eigenvalues dictionary
    Eigenvectors['E3'] = eigenvector_3                                                                                  # Store eigenvector 3 in eigenvalues dictionary
    return [Tensor, Eigenvalues, Eigenvectors, Standard_DTI_Metrics]
