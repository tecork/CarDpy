def register(original_matrix, original_bvals, original_bvecs, registration_algorithm = 'Rigid', temporary_denoising = 'OFF', operation_type = 'Magnitude'):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values (Recommended if using temporary denoising).
    original_bvecs          : Sorted b-vectors (Optional).
    registration_algorithm  : Select type of registration to perform (Recommended).
                              Default registration algorithm is rigid.
                              Registration algorithm options include Translation, Rigid, Affine, and Elastic.
    temporary_denoising     : Temporary densoing to help improve registration accuracy (Optional).
                              Registration will be applied on the orignal matrix to preserve original data.
                              The default status of denoising is turned off.
                              If turned on, the denoising algorithm is Patch2Self is used for increased SNR gain
    operation_type          : Identify the type of operation for original matrix (Optional).
                              Default operation type is magnitude.
                              Operation type options include Magnitude and Complex.
    ########## Definition Outputs #################################################################################################################
    registered_matrix         : Sorted registered diffusion data (5D - [rows, columns, slices, directions, averages]).
    registered_bvals          : Sorted b-values.
    registered_bvecs          : Sorted b-vectors.    
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules #####################################################################################################################
    import numpy                            as     np                                                                                               #
    from   cardpy.Data_Processing.Denoising import denoise                                                                                          #
    from   dipy.align.imaffine              import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)              #
    from   dipy.align.transforms            import (TranslationTransform2D, RigidTransform2D, AffineTransform2D)                                    #
    from   dipy.align.imwarp                import SymmetricDiffeomorphicRegistration                                                               #
    from   dipy.align.metrics               import SSDMetric, CCMetric, EMMetric                                                                    #
    ########## Address data type of original matrix ################################################################################################
    slices                          = original_matrix.shape[2]                                                                                      # Extract number of slices
    directions                      = original_matrix.shape[3]                                                                                      # Extract number of directions
    averages                        = original_matrix.shape[4]                                                                                      # Extract number of averages
    if original_matrix.dtype == 'complex128':                                                                                                       # If data type is complex ...
        if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
            real_matrix                        = np.real(original_matrix)                                                                                   # Separate real part from complex data
            imag_matrix                        = np.imag(original_matrix)                                                                                   # Separate imaginary part from complex data
            registered_diffusion_original_real = np.zeros(original_matrix.shape)                                                                            # Initialize registered diffusion original real matrix
            registered_diffusion_original_imag = np.zeros(original_matrix.shape)                                                                            # Initialize registered diffusion original imaginary matrix
            registered_average_original_real   = np.zeros(original_matrix.shape)                                                                            # Initialize registered average original real matrix
            registered_average_original_imag   = np.zeros(original_matrix.shape)                                                                            # Initialize registered average original imaginary matrix
            registered_matrix                  = np.zeros(original_matrix.shape)                                                                            # Initialize registered matrix
            registered_matrix                  = registered_matrix.astype(np.complex128)                                                                    # Cast registered matrix for complex data
        if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
            original_matrix                   = np.abs(original_matrix)                                                                                     # Convert complex original matrix to magnitude data
            registered_diffusion_original_mag = np.zeros(original_matrix.shape)                                                                             # Initialize registered diffusion original magnitude matrix
            registered_average_original_mag   = np.zeros(original_matrix.shape)                                                                             # Initialize registered average original magnitude matrix
            registered_matrix                 = np.zeros(original_matrix.shape)                                                                             # Initialize registered matrix
            registered_matrix                 = registered_matrix.astype(np.float64)                                                                        # Cast registered matrix for magnitude data
            print('Input data type is complex, but magnitude is being executed.')                                                                           # Print warning
    else:                                                                                                                                               # Otherwise ...
        registered_diffusion_original_mag = np.zeros(original_matrix.shape)                                                                                 # Initialize registered diffusion original magnitude matrix
        registered_average_original_mag   = np.zeros(original_matrix.shape)                                                                                 # Initialize registered average original magnitude matrix
        registered_matrix                 = np.zeros(original_matrix.shape)                                                                                 # Initialize registered matrix
        registered_matrix                 = registered_matrix.astype(np.float64)                                                                            # Cast registered matrix for magnitude data
    ########## Implement temporary denoising #######################################################################################################
    if temporary_denoising == 'ON':                                                                                                                 # If temporary denoising is on ...
        print('Temporary denoising is turned on.')                                                                                                      # Print warning
        [denoised_matrix, _, _]       = denoise(original_matrix, original_bvals, original_bvecs, numCoils = 20, denoising_algorithm = 'Patch2Self')     # Denoise original matrix
        registered_diffusion_denoised = np.zeros(original_matrix.shape)                                                                                 # Initialize registered diffusion denoised matrix
        temporary_matrix              = np.abs(denoised_matrix)                                                                                         # Create temporary matrix
    if temporary_denoising == 'OFF':                                                                                                                # If temporary denoising is off ...
        print('Temporary denoising is turned off.')                                                                                                     # Print warning
        registered_diffusion_original = np.zeros(original_matrix.shape)                                                                                 # Initialize registered diffusion original matrix
        temporary_matrix              = np.abs(original_matrix)                                                                                         # Create temporary matrix
    ########## Registering diffusion directions to b = 0 for each average ##########################################################################
    print('Registering diffusion directions to first diffusion direction (b = 0) for each average.')                                                # Print warning
    for avg in range(averages):                                                                                                                     # Iterate through averages
        for slc in range(slices):                                                                                                                       # Iterate through slices
            for dif in range(directions):                                                                                                                   # Iterate through diffusion directions
                static            = temporary_matrix[:, :, slc,   0, avg]                                                                                       # Choose static image
                moving            = temporary_matrix[:, :, slc, dif, avg]                                                                                       # Choose moving image
                ##### Center of mass in 2D #####
                static_grid2world = None                                                                                                                        # Define static coordinates
                moving_grid2world = None                                                                                                                        # Define moving coordinates
                c_of_mass         = transform_centers_of_mass(static, static_grid2world, moving, moving_grid2world)                                             # Initialize and perform center of mass registration
                ##### Initialize registration metric #####
                nbins         = 32                                                                                                                              # Set number of bins for mutual information metric
                sampling_prop = None                                                                                                                            # Set sampling property
                metric        = MutualInformationMetric(nbins, sampling_prop)                                                                                   # Define registration metric
                level_iters   = [10000, 1000, 100]                                                                                                           # Define pyramid for iterations
                sigmas        = [3.0,    2.0,   0.0]                                                                                                            # Define pyramid for sigmas
                factors       = [2,      1,     1]                                                                                                              # Define pyramid for factors
                affreg        = AffineRegistration(metric = metric, level_iters = level_iters, sigmas = sigmas, factors = factors, verbosity = 0)               # Initialize non-elastic registration parameters
                ##### Translation transform in 2D #####
                transform       = TranslationTransform2D()                                                                                                      # Define transformation as translation
                params0         = None                                                                                                                          # Define parameters
                starting_affine = c_of_mass.affine                                                                                                              # Choose starting registration transformation from center of mass transformation
                translation     = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                                  starting_affine = starting_affine)                                                                            # Initialize and perform translation registration
                if registration_algorithm == 'Translation':                                                                                                     # If registration algorithm is translation ...
                    if temporary_denoising == 'ON':                                                                                                                 # If temporary denoising is on ...
                        transformed_denoised = translation.transform(temporary_matrix[:, :, slc, dif, avg])                                                             # Apply and store translation registered denoised image
                    if temporary_denoising == 'OFF':                                                                                                                # If temporary denoising is off ...
                        transformed_original = translation.transform(temporary_matrix[:, :, slc, dif, avg])                                                             # Apply and store translation registered original image
                    if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                        transformed_original_real = translation.transform(real_matrix[:, :, slc, dif, avg])                                                             # Apply and store translation registered real image
                        transformed_original_imag = translation.transform(imag_matrix[:, :, slc, dif, avg])                                                             # Apply and store translation registered imaginary image
                    if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                        transformed_original_mag  = translation.transform(original_matrix[:, :, slc, dif, avg])                                                         # Apply and store translation registered magnitude image
                ##### Rigid transform in 2D #####
                if (registration_algorithm == 'Rigid' or registration_algorithm == 'Affine' or registration_algorithm == 'Elastic'):                            # If registration algorithm is rigid, affine, or elastic ...
                    transform       = RigidTransform2D()                                                                                                            # Define transformation as rigid
                    params0         = None                                                                                                                          # Define parameters
                    starting_affine = translation.affine                                                                                                            # Choose starting registration transformation from translation transformation
                    rigid           = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                                      starting_affine = starting_affine)                                                                            # Initialize and perform rigid registration
                    if registration_algorithm == 'Rigid':                                                                                                           # If registration algorithm is rigid ...
                        if temporary_denoising == 'ON':                                                                                                                 # If temporary denoising is on ...
                            transformed_denoised = rigid.transform(temporary_matrix[:, :, slc, dif, avg])                                                                   # Apply and store rigid registered denoised image
                        if temporary_denoising == 'OFF':                                                                                                                # If temporary denoising is off ...
                            transformed_original = rigid.transform(temporary_matrix[:, :, slc, dif, avg])                                                                   # Apply and store rigid registered original image
                        if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                            transformed_original_real = rigid.transform(real_matrix[:, :, slc, dif, avg])                                                                   # Apply and store rigid registered real image
                            transformed_original_imag = rigid.transform(imag_matrix[:, :, slc, dif, avg])                                                                   # Apply and store rigid registered imaginary image
                        if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                            transformed_original_mag  = rigid.transform(original_matrix[:, :, slc, dif, avg])                                                               # Apply and store rigid registered magnitude image
                    ##### Affine transform in 2D #####
                    if (registration_algorithm == 'Affine' or registration_algorithm == 'Elastic'):                                                                 # If registration algorithm is affine or elastic ...
                        transform       = AffineTransform2D()                                                                                                           # Define transformation as affine
                        params0         = None                                                                                                                          # Define parameters
                        starting_affine = rigid.affine                                                                                                                  # Choose starting registration transformation from rigid transformation
                        affine          = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                                          starting_affine = starting_affine)                                                                            # Initialize and perform affine registration
                        if temporary_denoising == 'ON':                                                                                                                 # If temporary denoising is on ...
                            transformed_denoised = affine.transform(temporary_matrix[:, :, slc, dif, avg])                                                                  # Apply and store affine registered denoised image
                        if temporary_denoising == 'OFF':                                                                                                                # If temporary denoising is off ...
                            transformed_original = affine.transform(temporary_matrix[:, :, slc, dif, avg])                                                                  # Apply and store affine registered original image
                        if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                            transformed_original_real = affine.transform(real_matrix[:, :, slc, dif, avg])                                                                  # Apply and store affine registered real image
                            transformed_original_imag = affine.transform(imag_matrix[:, :, slc, dif, avg])                                                                  # Apply and store affine registered imaginary image
                        if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                            transformed_original_mag  = affine.transform(original_matrix[:, :, slc, dif, avg])                                                              # Apply and store affine registered magnitude image
                    ##### Elastic registrations in 2D #####
                    if (registration_algorithm == 'Elastic'):                                                                                                       # If registration algorithm is elastic ...
                        metric               = EMMetric(2)                                                                                                              # Define registration metric
                        level_iters          = [100, 50, 25]                                                                                                            # Define pyramid for iterations
                        moving_elastic       = affine.transform(temporary_matrix[:, :, slc, dif, avg])                                                                  # Set moving image as affine registered temporary image
                        sdr                  = SymmetricDiffeomorphicRegistration(metric, level_iters)                                                                  # Initialize elastic registration
                        mapping              = sdr.optimize(static, moving_elastic)                                                                                     # Perform elastic registration
                        if temporary_denoising == 'ON':                                                                                                                 # If temporary denoising is on ...
                            transformed_denoised = mapping.transform(temporary_matrix[:, :,  slc, dif, avg])                                                                # Apply and store elastic registered denoised image
                        if temporary_denoising == 'OFF':                                                                                                                # If temporary denoising is off ...
                            transformed          = mapping.transform(temporary_matrix[:, :,  slc, dif, avg])                                                                # Apply and store elastic registered denoised image
                        if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                            transformed_original_real = mapping.transform(real_matrix[:, :,  slc, dif, avg])                                                                # Apply and store elastic registered real image
                            transformed_original_imag = mapping.transform(imag_matrix[:, :,  slc, dif, avg])                                                                # Apply and store elastic registered imaginary image
                        if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                            transformed_original_mag = mapping.transform(original_matrix[:, :,  slc, dif, avg])                                                             # Apply and store elastic registered imaginary image
                    ##### Store transformations #####
                    if temporary_denoising == 'ON':                                                                                                                 # If temporary denoising is on ...
                        registered_diffusion_denoised[:, :, slc, dif, avg] = transformed_denoised                                                                       # Store transformed denoised image
                    if temporary_denoising == 'OFF':                                                                                                                # If temporary denoising is off ...
                        registered_diffusion_original[:, :, slc, dif, avg] = transformed_original                                                                       # Store transformed original image
                    if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                        registered_diffusion_original_real[:, :, slc, dif, avg] = transformed_original_real                                                             # Store transformed real image
                        registered_diffusion_original_imag[:, :, slc, dif, avg] = transformed_original_imag                                                             # Store transformed imaginary image
                    if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                        registered_diffusion_original_mag[:, :, slc, dif, avg] = transformed_original_mag                                                               # Store transformed magnitude image
    ########## Reinitialize temporary matrix for average registration process ######################################################################
    if temporary_denoising == 'ON':                                                                                                                 # If temporary denoising is on ...
        registered_average_denoised   = np.zeros(original_matrix.shape)                                                                                 # Initialize registered average denoised matrix
        temporary_matrix              = np.abs(registered_diffusion_denoised)                                                                           # Create temporary matrix
    if temporary_denoising == 'OFF':                                                                                                                # If temporary denoising is off ...
        registered_average_original   = np.zeros(original_matrix.shape)                                                                                 # Initialize registered average original matrix
        temporary_matrix              = np.abs(registered_diffusion_original)                                                                           # Create temporary matrix
    ########## Registering b = 0 images for each average to first b = 0 image ######################################################################
    print('Registering first diffusion direction (b = 0) to each average first diffusion direction (b = 0).')                                       # Print warning
    for avg in range(averages):                                                                                                                     # Iterate through averages
        for slc in range(slices):                                                                                                                       # Iterate through slices
            static            = temporary_matrix[:, :, slc, 0,   0]                                                                                         # Choose static image
            moving            = temporary_matrix[:, :, slc, 0, avg]                                                                                         # Choose moving image
            ##### Center of mass in 2D #####
            static_grid2world = None                                                                                                                        # Define static coordinates
            moving_grid2world = None                                                                                                                        # Define moving coordinates
            c_of_mass         = transform_centers_of_mass(static, static_grid2world, moving, moving_grid2world)                                             # Initialize and perform center of mass registration
            ##### Initialize registration metric #####
            nbins         = 32                                                                                                                              # Set number of bins for mutual information metric
            sampling_prop = None                                                                                                                            # Set sampling property
            metric        = MutualInformationMetric(nbins, sampling_prop)                                                                                   # Define registration metric
            level_iters   = [10000, 1000, 100]                                                                                                           # Define pyramid for iterations
            sigmas        = [3.0,    2.0,   0.0]                                                                                                            # Define pyramid for sigmas
            factors       = [2,      1,     1]                                                                                                              # Define pyramid for factors
            affreg        = AffineRegistration(metric = metric, level_iters = level_iters, sigmas = sigmas, factors = factors, verbosity = 0)               # Initialize non-elastic registration parameters
            ##### Translation transform in 2D #####
            transform       = TranslationTransform2D()                                                                                                      # Define transformation as translation
            params0         = None                                                                                                                          # Define parameters
            starting_affine = c_of_mass.affine                                                                                                              # Choose starting registration transformation from center of mass transformation
            translation     = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                              starting_affine = starting_affine)                                                                            # Initialize and perform translation registration
            ##### Rigid transform in 2D #####
            if (registration_algorithm == 'Rigid' or registration_algorithm == 'Affine' or registration_algorithm == 'Elastic'):                            # If registration algorithm is rigid, affine, or elastic ...
                transform       = RigidTransform2D()                                                                                                            # Define transformation as rigid
                params0         = None                                                                                                                          # Define parameters
                starting_affine = translation.affine                                                                                                            # Choose starting registration transformation from translation transformation
                rigid           = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                                  starting_affine = starting_affine)                                                                            # Initialize and perform rigid registration
            ##### Affine transform in 2D #####
            if (registration_algorithm == 'Affine' or registration_algorithm == 'Elastic'):                                                                 # If registration algorithm is affine or elastic ...
                transform       = AffineTransform2D()                                                                                                           # Define transformation as affine
                params0         = None                                                                                                                          # Define parameters
                starting_affine = rigid.affine                                                                                                                  # Choose starting registration transformation from rigid transformation
                affine          = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world,
                                                  starting_affine = starting_affine)                                                                            # Initialize and perform affine registration
                if (registration_algorithm == 'Elastic'):                                                                                                       # If registration algorithm is elastic ...
                    metric               = SSDMetric(2)                                                                                                             # Define registration metric
                    level_iters          = [100, 50, 25]                                                                                                            # Define pyramid for iterations
                    moving_elastic       = affine.transform(temporary_matrix[:, :, slc, dif, avg])                                                                  # Set moving image as affine registered temporary image
                    sdr                  = SymmetricDiffeomorphicRegistration(metric, level_iters)                                                                  # Initialize elastic registration
                    mapping              = sdr.optimize(static, moving_elastic)                                                                                     # Perform elastic registration
            ##### Apply registration across diffusion directions #####
            for dif in range(directions):                                                                                                                   # Iterate through diffusion directions
                if (registration_algorithm == 'Translation'):                                                                                                   # If registration algorithm is translation ...
                    if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                        transformed_original_real = translation.transform(registered_diffusion_original_real[:, :, slc, dif, avg])                                      # Apply and store translation registered real image
                        transformed_original_imag = translation.transform(registered_diffusion_original_imag[:, :, slc, dif, avg])                                      # Apply and store translation registered imaginary image
                    if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                        transformed_original_mag = translation.transform(registered_diffusion_original_mag[:, :, slc, dif, avg])                                        # Apply and store translation registered magnitude image
                if (registration_algorithm == 'Rigid'):                                                                                                         # If registration algorithm is rigid ...
                    if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                        transformed_original_real = rigid.transform(registered_diffusion_original_real[:, :, slc, dif, avg])                                            # Apply and store rigid registered real image
                        transformed_original_imag = rigid.transform(registered_diffusion_original_imag[:, :, slc, dif, avg])                                            # Apply and store rigid registered imaginary image
                    if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                        transformed_original_mag = rigid.transform(registered_diffusion_original_mag[:, :, slc, dif, avg])                                              # Apply and store rigid registered magnitude image
                if (registration_algorithm == 'Affine'):                                                                                                        # If registration algorithm is affine ...
                    if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                        transformed_original_real = affine.transform(registered_diffusion_original_real[:, :, slc, dif, avg])                                           # Apply and store affine registered real image
                        transformed_original_imag = affine.transform(registered_diffusion_original_imag[:, :, slc, dif, avg])                                           # Apply and store affine registered imaginary image
                    if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                        transformed_original_mag = affine.transform(registered_diffusion_original_mag[:, :, slc, dif, avg])                                             # Apply and store affine registered magnitude image
                if (registration_algorithm == 'Elastic'):                                                                                                       # If registration algorithm is elastic ...
                    if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                        transformed_original_real = mapping.transform(registered_diffusion_original_real[:, :,  slc, dif, avg])                                         # Apply and store elastic registered real image
                        transformed_original_imag = mapping.transform(registered_diffusion_original_imag[:, :,  slc, dif, avg])                                         # Apply and store elastic registered imaginary image
                    if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                        transformed_original_mag = mapping.transform(registered_diffusion_original_mag[:, :,  slc, dif, avg])                                           # Apply and store elastic registered magnitude image
                if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
                    registered_average_original_real[:, :, slc, dif, avg] = transformed_original_real                                                               # Store transformed original real in registered average real matrix
                    registered_average_original_imag[:, :, slc, dif, avg] = transformed_original_imag                                                               # Store transformed original imaginary in registered average imaginary matrix
                if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
                    registered_average_original_mag[:, :, slc, dif, avg] = transformed_original_mag                                                                 # Store transformed original magnitude in registered average magnitude matrix
    print('Finished registration.')                                                                                                                 # Print warning
    ########## Tie-up-loose-ends ... ##############################################################################################################
    if operation_type == 'Complex':                                                                                                                 # If operation type is complex ...
        registered_matrix = registered_average_original_real + (1j * registered_average_original_imag)                                                  # Store registered matrix and convert to complex data
    if operation_type == 'Magnitude':                                                                                                               # If operation type is magnitude ...
        registered_matrix = registered_average_original_mag                                                                                             # Store registered matrix
    registered_bvals  = original_bvals                                                                                                              # Store b-values
    registered_bvecs  = original_bvecs                                                                                                              # Store b-vectors
    return [registered_matrix, registered_bvals, registered_bvecs]
