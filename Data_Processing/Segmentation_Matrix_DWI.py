def DWI_segmentation_matrix(original_matrix, original_bvals, original_bvecs):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvals          : Sorted b-values.
    original_bvecs          : Sorted b-vectors.
    ########## Definition Outputs #################################################################################################################
    segmentation_matrix     : Annotated stacked extended diffusion data (4D - [rows, columns, slices, maps)
                              Maps include the following:
                              - Low b-value images [Not to scale]
                              - High b-value images [Not to scale]
                              - Apparent Diffusion Coefficient (ADC) [Not to scale]
    segmentation_bvals      : Dummy b-values list required for MITK to read as Diffusion data for segmenatation (Reccomended).
    segmentation_bvecs      : Dummy b-vectors list required for MITK to read as Diffusion data for segmenatation (Reccomended).
    """
    ########## Definition Information ##############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import modules ######################################################################################################################
    from   cardpy.Data_Processing.DWI import DWI_recon
    from   cardpy.Data_Sorting        import sorted2stacked, stacked2sorted                                                   # Import sorted to stacked and stacked to sorted from CarDpy
    import numpy                         as np
    import cv2
    ########## Initialize interpolated matrix ######################################################################################################
    [Standard_DWI_Metrics]       = DWI_recon(original_matrix, original_bvals, original_bvecs)
    ########## Convert sorted data into stacked data ##############################################################################################
    [stacked_matrix, stacked_bvals, stacked_bvecs] = sorted2stacked(original_matrix, original_bvals, original_bvecs)            # Convert sorted data into stacked data
    
    rows                         = stacked_matrix.shape[0]
    columns                      = stacked_matrix.shape[1]
    slices                       = stacked_matrix.shape[2]
    directions                   = stacked_matrix.shape[3]
    segmentation_matrix          = np.zeros((rows, columns, slices, directions + 1))
    text_mask                    = np.zeros((rows, columns))
    text_mask[int(rows / 6):, :] = 1

    for slc in range(slices):
        for dif in range(directions):
            blank_matrix                        = np.zeros((rows, columns), dtype = np.uint8)
            cv2.putText(blank_matrix,
                        text      = 'DWI Image: ' + str(dif + 1),
                        org       = (rows // 20, columns // 10),
                        fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.65,
                        color     = (1, 1, 1),
                        thickness = 2,
                        lineType  = cv2.FILLED)
            image_matrix                        = stacked_matrix[:, :, slc, dif] * text_mask
            image_matrix_norm                   = (image_matrix / image_matrix.max()) + blank_matrix
            image_matrix_scaled                 = image_matrix_norm * 400
            segmentation_matrix[:, :, slc, dif] = image_matrix_scaled
        ##### Apparent Diffusion Coefficient #####
        blank_matrix                                   = np.zeros((rows, columns), dtype = np.uint8)
        cv2.putText(blank_matrix,
                    text      = 'ADC',
                    org       = (rows // 20, columns // 10),
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.65,
                    color     = (1, 1, 1),
                    thickness = 2,
                    lineType  = cv2.FILLED)
        image_matrix                                   = Standard_DWI_Metrics['ADC'][:, :, slc] * text_mask
        image_matrix_norm                              = (image_matrix / image_matrix.max()) + blank_matrix
        image_matrix_scaled                            = image_matrix_norm * 400
        segmentation_matrix[:, :, slc, directions + 0] = image_matrix_scaled
    ########## Tie-up-loose-ends ... ##############################################################################################################
    segmentation_bvecs = np.zeros((segmentation_matrix.shape[3], 3))
    segmentation_bvals = np.zeros(segmentation_matrix.shape[3])
    return [segmentation_matrix, segmentation_bvals, segmentation_bvecs]
