def cDTI_recon(myocardial_mask, Eigenvectors, num_interp_points = 200, smoothness_level = 'Low', Helix_Angle_Filter_Settings = 'Default'):
    """
    ########## Definition Inputs ##################################################################################################################
    # myocardial_mask               : Left or right ventricular binary mask.
    # Eigenvectors                  : Python dictonary containing eigenvector 1, eigenvector 2, and eigenvector 3.
    # num_interp_points             : Number of interpolated points to be generated. Default is set to 200 points.
    # smoothness_level              : Smoothing factor for B-spline data. Default is set to low.
                                      Smoothless level options inclue Native (no smoothing), Low, Medium, High, and Extreme.
    # helix_angle_outlier_criteria  :
    ########## Definition Outputs #################################################################################################################
    # Cardiac_DTI_Metrics           : Python dictionary containing all of the cardiac DTI quantitative metrics, which can be seen directly below.
    #                                 - helix angle (HA)
    #                                 - helix angle filtered (HAF)
    #                                 - E2 angle (E2A)
    #                                 - transverse angle (TA)
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules #####################################################################################################################
    import scipy
    import math
    import numpy                       as     np
    from   cardpy.Data_Processing.cDTI import Microstructure_Angle_Projections, Linear_Helix_Angle_Filtering, Spatial_Helix_Angle_Filtering
    from   cardpy.Tools.Contours       import Myocardial_Mask_Contour_Extraction, Angular_Contour_Organization
    from   cardpy.Tools.Contours       import Contour_BSpline, Myocardial_Mask_Contour_Filler
    from   cv2                         import dilate, getStructuringElement, MORPH_ELLIPSE
    
    if Helix_Angle_Filter_Settings == 'Default':
        Helix_Angle_Filter_Settings = dict()
        Helix_Angle_Filter_Settings['Linear Filter: Outlier StDev']      = 1
        Helix_Angle_Filter_Settings['Spatial Filter: Wall Depth Factor'] = 0.25
        Helix_Angle_Filter_Settings['Spatial Filter: Kernel Size']       = 5
    
    ### Dialte oringal mask
    myocardial_mask_dialated = np.zeros(myocardial_mask.shape)
    kernel                   = getStructuringElement(MORPH_ELLIPSE, (5,5))
    for slc in range(myocardial_mask_dialated.shape[2]):
        temp_matrix                         = np.copy(myocardial_mask[:, :, slc])
        myocardial_mask_dialated[:, :, slc] = dilate(temp_matrix, kernel, iterations = 1)
    ###  Extract Contours from Dilated LV Mask
    slices                                      = myocardial_mask_dialated.shape[2]
    epicardial_cordinates_dilated               = []
    endocardial_cordinates_dilated              = []
    epicardial_cordinates_interpolated_dilated  = []
    endocardial_cordinates_interpolated_dilated = []
    for slc in range(slices):
#        [epicardial_contour_dilated,      endocardial_contour_dilated]          = myocardial_mask_contour_extractor(myocardial_mask_dialated, slc)
        [Epicardium_Contours_Dialated, Endocardium_Contours_Dialated, _]         = Myocardial_Mask_Contour_Extraction(myocardial_mask_dialated, slc)
        epicardial_contour_dilated                                               = Epicardium_Contours_Dialated[0]
        endocardial_contour_dilated                                              = Endocardium_Contours_Dialated[0]
        [endocardial_y_original_dilated,  endocardial_x_original_dilated]        = np.where(endocardial_contour_dilated == 1)
        [endocardial_x_organized_dilated, endocardial_y_organized_dilated, _, _] = Angular_Contour_Organization(endocardial_x_original_dilated,
                                                                                                                endocardial_y_original_dilated,
                                                                                                                x_offset = 'None',
                                                                                                                y_offset = 'None')
        [epicardial_y_original_dilated,   epicardial_x_original_dilated]         = np.where(epicardial_contour_dilated == 1)
        [epicardial_x_organized_dilated,  epicardial_y_organized_dilated, _, _]  = Angular_Contour_Organization(epicardial_x_original_dilated,
                                                                                                                epicardial_y_original_dilated,
                                                                                                                x_offset = 'None',
                                                                                                                y_offset = 'None')
        epicardial_cordinates_dilated.append( [epicardial_x_organized_dilated,  epicardial_y_organized_dilated])
        endocardial_cordinates_dilated.append([endocardial_x_organized_dilated, endocardial_y_organized_dilated])
        if len(endocardial_cordinates_dilated[slc][0]) > 1:
            [epicardial_interpolated_x_dilated,  epicardial_interpolated_y_dilated]  = Contour_BSpline(epicardial_cordinates_dilated[slc][0],
                                                                                                       epicardial_cordinates_dilated[slc][1],
                                                                                                       num_interp_points,
                                                                                                       smoothness_level)
            epicardial_cordinates_interpolated_dilated.append([epicardial_interpolated_x_dilated, epicardial_interpolated_y_dilated])
            [endocardial_interpolated_x_dilated, endocardial_interpolated_y_dilated] = Contour_BSpline(endocardial_cordinates_dilated[slc][0],
                                                                                                       endocardial_cordinates_dilated[slc][1],
                                                                                                       num_interp_points,
                                                                                                       smoothness_level)
            endocardial_cordinates_interpolated_dilated.append([endocardial_interpolated_x_dilated, endocardial_interpolated_y_dilated])
            

        else:
            epicardial_cordinates_interpolated_dilated.append( ['No Data'])
            endocardial_cordinates_interpolated_dilated.append(['No Data'])
        
    
    ### Extract Contours from Original LV Mask
    slices                              = myocardial_mask.shape[2]
    epicardial_cordinates               = []
    endocardial_cordinates              = []
    epicardial_cordinates_interpolated  = []
    endocardial_cordinates_interpolated = []
    interpolated_mask                   = np.zeros(myocardial_mask.shape)

    for slc in range(slices):
#        [epicardial_contour,      endocardial_contour]           = myocardial_mask_contour_extractor(myocardial_mask, slc)
        [Epicardium_Contours, Endocardium_Contours, _]           = Myocardial_Mask_Contour_Extraction(myocardial_mask, slc)
        epicardial_contour                                       = Epicardium_Contours[0]
        endocardial_contour                                      = Endocardium_Contours[0]
        [epicardial_y_original,   epicardial_x_original]         = np.where(epicardial_contour == 1)
        [epicardial_x_organized,  epicardial_y_organized, _, _]  = Angular_Contour_Organization(epicardial_x_original,
                                                                                                epicardial_y_original,
                                                                                                x_offset = 'None',
                                                                                                y_offset = 'None')
        [endocardial_y_original,  endocardial_x_original]        = np.where(endocardial_contour == 1)
        [endocardial_x_organized, endocardial_y_organized, _, _] = Angular_Contour_Organization(endocardial_x_original,
                                                                                                endocardial_y_original,
                                                                                                x_offset = 'None',
                                                                                                y_offset = 'None')


        epicardial_cordinates.append( [epicardial_x_organized,  epicardial_y_organized])
        endocardial_cordinates.append([endocardial_x_organized, endocardial_y_organized])
        if len(endocardial_cordinates[slc][0]) > 1:
            [epicardial_interpolated_x,  epicardial_interpolated_y]  = Contour_BSpline(epicardial_cordinates[slc][0],
                                                                                       epicardial_cordinates[slc][1],
                                                                                       num_interp_points,
                                                                                       smoothness_level)
            [endocardial_interpolated_x, endocardial_interpolated_y] = Contour_BSpline(endocardial_cordinates[slc][0],
                                                                                       endocardial_cordinates[slc][1],
                                                                                       num_interp_points,
                                                                                       smoothness_level)
            endocardial_cordinates_interpolated.append([endocardial_interpolated_x, endocardial_interpolated_y])
            epicardial_cordinates_interpolated.append( [epicardial_interpolated_x,  epicardial_interpolated_y])
            interpolated_mask[:, :, slc] = Myocardial_Mask_Contour_Filler(myocardial_mask[:, :, slc],
                                                                          endocardial_cordinates_interpolated[slc],
                                                                          epicardial_cordinates_interpolated[slc])
        else:
            epicardial_cordinates_interpolated.append(['No Data'])
            endocardial_cordinates_interpolated.append(['No Data'])
    
    ### Calculate Microstrucural Angle Analysis
    helix_angle                  = np.empty(myocardial_mask.shape)                                                                                                            # Initialize empty helix angle matrix
    helix_angle.fill(np.nan)                                                                                                                                # Fill helix angle matrix with np.nan
    helix_angle_linear_filtered  = np.empty(myocardial_mask.shape)
    helix_angle_linear_filtered.fill(np.nan)                                                                                                                                # Fill helix angle matrix with np.nan
    helix_angle_spatial_filtered = np.empty(myocardial_mask.shape)
    helix_angle_spatial_filtered.fill(np.nan)                                                                                                                                # Fill helix angle matrix with np.nan
    e2_angle                     = np.empty(myocardial_mask.shape)                                                                                                            # Initialize empty e2 angle matrix
    e2_angle.fill(np.nan)                                                                                                                                   # Fill e2 angle matrix with np.nan
    transverse_angle             = np.empty(myocardial_mask.shape)                                                                                                            # Initialize empty transverse angle matrix
    transverse_angle.fill(np.nan)                                                                                                                           # Fill transverse angle matrix with np.nan
    Epicardial_Points_List  = []
    Endocardial_Points_List = []
    for slc in range(slices):
        if (epicardial_cordinates_interpolated[slc] == ['No Data'] or endocardial_cordinates_interpolated[slc] == ['No Data']):
            print('Helix Angle cannot be calculated on slice %i.' %int(slc))
            Epicardial_Points_List.append([])
            Epicardial_Points_List.append([])
        else:
            points_epicardium                         = np.stack(epicardial_cordinates_interpolated_dilated[slc], axis=1)
            points_endocardium                        = np.stack(endocardial_cordinates_interpolated_dilated[slc], axis=1)
            [Helix_Angle, E2_Angle, Transverse_Angle] = Microstructure_Angle_Projections(interpolated_mask[:, :, slc],
                                                                                         Eigenvectors['E1'][:, :, slc, :],
                                                                                         Eigenvectors['E2'][:, :, slc, :],
                                                                                         points_epicardium,
                                                                                         points_endocardium)
            Helix_Angle_Linear_Filtered               = Linear_Helix_Angle_Filtering(Helix_Angle,
                                                                                     points_epicardium,
                                                                                     points_endocardium,
                                                                                     Helix_Angle_Filter_Settings['Linear Filter: Outlier StDev'] )
            Helix_Angle_Spatial_Filtered              = Spatial_Helix_Angle_Filtering(Helix_Angle,
                                                                                      interpolated_mask[:, :, slc],
                                                                                      Helix_Angle_Filter_Settings['Spatial Filter: Wall Depth Factor'],
                                                                                      Helix_Angle_Filter_Settings['Spatial Filter: Kernel Size'])
            helix_angle[:, :, slc]                  = Helix_Angle
            helix_angle_linear_filtered[:, :, slc]  = Helix_Angle_Linear_Filtered
            helix_angle_spatial_filtered[:, :, slc] = Helix_Angle_Spatial_Filtered
            e2_angle[:, :, slc]                     = E2_Angle
            transverse_angle[:, :, slc]             = Transverse_Angle
            Epicardial_Points_List.append(points_epicardium)
            Endocardial_Points_List.append(points_endocardium)
    Cardiac_DTI_Metrics = dict()
    Cardiac_DTI_Metrics['HA']   = helix_angle
    Cardiac_DTI_Metrics['HALF'] = helix_angle_linear_filtered
    Cardiac_DTI_Metrics['HASF'] = helix_angle_spatial_filtered
    Cardiac_DTI_Metrics['E2A']  = e2_angle
    Cardiac_DTI_Metrics['TA']   = transverse_angle
    return [Cardiac_DTI_Metrics, Epicardial_Points_List, Endocardial_Points_List, interpolated_mask]
    
def Microstructure_Angle_Projections(myocardial_mask, Eigenvector_1, Eigenvector_2, points_epicardium, points_endocardium):
    """
    ########## Definition Inputs ##################################################################################################################
    # myocardial_mask    : Left or right ventricular binary mask
    # Eigenvector_1      :
    # Eigenvector_2      :
    # points_epicardium  :
    # points_endocardium :
    ########## Definition Outputs #################################################################################################################
    # helix_angle        :
    # e2_angle           :
    # transverse_angle   :
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2021
    ########## Import Modules #####################################################################################################################
    import scipy
    import math
    import numpy as np
    ########## Import Modules #####################################################################################################################
    rows             = myocardial_mask.shape[0]                                                                                                             # Extract number of rows
    columns          = myocardial_mask.shape[1]                                                                                                             # Extract number of columns
    helix_angle      = np.empty([rows, columns])                                                                                                            # Initialize empty helix angle matrix
    helix_angle.fill(np.nan)                                                                                                                                # Fill helix angle matrix with np.nan
    e2_angle         = np.empty([rows, columns])                                                                                                            # Initialize empty e2 angle matrix
    e2_angle.fill(np.nan)                                                                                                                                   # Fill e2 angle matrix with np.nan
    transverse_angle = np.empty([rows, columns])                                                                                                            # Initialize empty transverse angle matrix
    transverse_angle.fill(np.nan)                                                                                                                           # Fill transverse angle matrix with np.nan
    ########## Import Modules #####################################################################################################################
    vector_epicardium       = np.zeros(points_epicardium.shape)                                                                                             # Initialize epicardium vector matrix with zeros
    vector_epicardium[0, :] = points_epicardium[0, :]  - points_epicardium[-1, :]                                                                           # Define first value of epicardium vector matrix
    for idx in range(points_epicardium.shape[0] - 1):                                                                                                       # Iterate through indicies
        vector_epicardium[idx + 1, :] = points_epicardium [idx + 1, :] - points_epicardium [idx, :]                                                             # Define index + 1 value of epicardium vector matrix
    
    vector_endocardium       = np.zeros(points_endocardium.shape)                                                                                           # Initialize endocardium vector matrix with zeros
    vector_endocardium[0, :] = points_endocardium[0, :] - points_endocardium[-1, :]                                                                         # Define first value of endocardium vector matrix
    for idx in range(points_endocardium.shape[0] - 1):                                                                                                      # Iterate through indicies
        vector_endocardium[idx + 1, :] = points_endocardium[idx + 1, :] - points_endocardium[idx, :]                                                            # Define index + 1 value of endocardium vector matrix
    
    positions      = np.vstack((points_epicardium, points_endocardium))                                                                                     # Define poisitons of epicardium and endocardium in one matrix
    vectors        = np.vstack((vector_epicardium, vector_endocardium))                                                                                     # Define vectors of epicardium and endocardium in one matrix
    x              = np.linspace(0, rows - 1,    rows)                                                                                                      # Define x vector for meshgrid
    y              = np.linspace(0, columns - 1, columns)                                                                                                   # Define y vector for meshgrid
    xi, yi         = np.meshgrid(y, x)                                                                                                                      # Create meshgrid
    Vx             = scipy.interpolate.griddata(positions, vectors[:, 0], (xi, yi), method = 'linear')                                                      # Combine x poition data and x vector data for both epicardium and endocardium
    Vy             = scipy.interpolate.griddata(positions, vectors[:, 1], (xi, yi), method = 'linear')                                                      # Combine y poition data and y vector data for both epicardium and endocardium
    ########## Import Modules #####################################################################################################################
    for x in range(rows):                                                                                                                                   # Iterate through rows
        for y in range(columns):                                                                                                                                # Iterate through columns
            if myocardial_mask[x, y] != 0:                                                                                                                          # If myocardial mask is not eqaul to 0 ...
                ##### Define and normalize vectors of intrest #####
                E1   = np.squeeze(Eigenvector_1[x, y, :])                                                                                                               # Extract primary eigenvector
                E1   = E1 / np.linalg.norm(E1, ord = 2)                                                                                                                 # Verify primary eigenvector is normalized
                E2   = np.squeeze(Eigenvector_2[x, y, :])                                                                                                               # Extract secondary eigenvector
                E2   = E2 / np.linalg.norm(E2, ord = 2)                                                                                                                 # Verify secondary eigenvector is normalized
                Circ = np.array([Vx[x, y], Vy[x, y], 0])                                                                                                                # Define circunferiential vector
                Circ = Circ / np.linalg.norm(Circ, ord = 2)                                                                                                             # Verify circunferiential vector is normalized
                Long = np.array([0, 0, 1])                                                                                                                              # Define longitudinal vector
                Rad  = np.cross(Circ / np.linalg.norm(Circ, ord = 2), Long / np.linalg.norm(Long, ord = 2))                                                             # Define radial vector
                ##### Calculate Helix Angle #####
                E1proj    = np.dot(E1, Circ) * Circ / (np.linalg.norm(Circ, ord = 2)**2)                                                                                # Calculate projection of the fiber vector (e1) onto the circunferential direction
                FiberProj = np.array([E1proj[0], E1proj[1], E1[2]])                                                                                                     # Extract fiber projection
                FiberProj = FiberProj / np.linalg.norm(FiberProj, ord = 2)                                                                                              # Verify fiber projection is normalized
                helix_angle[x, y] = math.asin(FiberProj[2] / np.linalg.norm(FiberProj, ord = 2)) * 180 / np.pi                                                          # Calculate helix angle
                if np.dot(Circ, E1) < 0:                                                                                                                                # Check if polarity of helix angle needs to be changed
                    helix_angle[x, y] = -helix_angle[x, y]                                                                                                                  # Change polarity of helix angle
                ##### Calculate E2 Angle #####
                MidFiber       = np.cross(FiberProj / np.linalg.norm(FiberProj, ord = 2), Rad / np.linalg.norm(Rad, ord = 2))                                           # Calculate middle of the fiber
                E2Proj         = np.dot(E2, Rad) * Rad / (np.linalg.norm(Rad, ord = 2)**2)                                                                              # Calculate projection of the Sheet Vector onto the Radial direction
                tProj3         = np.dot(E2, MidFiber) * MidFiber / (np.linalg.norm(MidFiber, ord = 2)**2)                                                               # Calculate projection of the Sheet Vector onto the MidFiber direction
                e2_angle[x, y] = math.atan2(np.linalg.norm(E2Proj, ord = 2 ), np.linalg.norm(tProj3, ord = 2)) * 180 / np.pi                                            # Calculate e2 angle
                ##### Calculate Transverse Angle #####
                vector_2               = np.array([np.squeeze(Eigenvector_1[x, y, 0]), np.squeeze(Eigenvector_1[x, y, 1]), 0])                                          #
                transverse_angle[x, y] = math.acos(np.dot(vector_2, Circ) / (np.linalg.norm(Circ, ord = 2) * np.linalg.norm(vector_2, ord = 2))) * 180 / np.pi          # Calculate transverse angle
                if np.isnan(transverse_angle[x, y]) == True:
                    transverse_angle[x, y] = 90
                if np.dot(vector_2, Circ) < 0:                                                                                                                          # Check if polarity of transverse angle needs to be changed
                    transverse_angle[x, y] = transverse_angle[x, y] - 180                                                                                                   # Change polarity of transverse angle
    return [helix_angle, e2_angle, transverse_angle]
    
def Linear_Helix_Angle_Filtering(helix_angle, points_epicardium, points_endocardium, outlier_standard_deviation = 1):
    """
    ########## Definition Inputs ##################################################################################################################
    # myocardial_mask    : Left or right ventricular binary mask
    # Eigenvector_1      :
    # Eigenvector_2      :
    # points_epicardium  :
    # points_endocardium :
    ########## Definition Outputs #################################################################################################################
    # helix_angle        :
    # e2_angle           :
    # transverse_angle   :
    """
    ########## Import Modules #####################################################################################################################
    import numpy                as np
    import scipy
    from   sklearn.linear_model import LinearRegression

    helix_angle_filtered      = np.empty(helix_angle.shape)
    helix_angle_filtered.fill(np.nan)
    
    helix_angle_coordinates = np.argwhere(~np.isnan(helix_angle))
    endocardial_line        = np.zeros(points_endocardium.shape)
    epicardial_line         = np.ones(points_epicardium.shape)
    ROI_position            = np.vstack((points_epicardium, points_endocardium))
    ROI_line                = np.vstack((epicardial_line, endocardial_line))
    rows                    = helix_angle.shape[0]                                                                                                               # Extract number of rows
    columns                 = helix_angle.shape[1]                                                                                                               # Extract number of columns
    x                       = np.linspace(0, rows - 1,    rows)                                                                                                      # Define x vector for meshgrid
    y                       = np.linspace(0, columns - 1, columns)                                                                                                   # Define y vector for meshgrid
    xi, yi                  = np.meshgrid(y, x)
    endo2epi_coordinates    = scipy.interpolate.griddata(ROI_position, ROI_line[:, 0], (xi, yi), method = 'linear')

    endo2epi_position    = endo2epi_coordinates[~np.isnan(helix_angle)]
    endo2epi_position    = endo2epi_position[:, np.newaxis]
    helix_angle_measured = helix_angle[~np.isnan(helix_angle)]
    helix_angle_measured = helix_angle_measured[:, np.newaxis]

    model                 = LinearRegression().fit(endo2epi_position, helix_angle_measured)
    helix_angle_predicted = model.intercept_ + model.coef_ * endo2epi_position

    incorrect_data  = (np.abs(helix_angle_predicted - helix_angle_measured) > outlier_standard_deviation * np.nanstd(helix_angle_measured)) * -1
    correct_data    = incorrect_data + 1
    data_correction = correct_data + incorrect_data

    for idx in range(helix_angle_coordinates.shape[0]):
        x = helix_angle_coordinates[idx, 0]
        y = helix_angle_coordinates[idx, 1]
        helix_angle_filtered[x, y] = helix_angle_measured[idx] * data_correction[idx]
    return helix_angle_filtered
    
def Spatial_Helix_Angle_Filtering(helix_angle, mask, wall_depth_factor = 0.25, kernel_size = 5):
    from   cardpy.Tools.Contours       import Myocardial_Mask_Contour_Extraction, Comprehensive_Outter_Contour, Moving_Window_2D
    from   cardpy.Data_Processing.cDTI import Endo2Epi_Grid
    import math
    import pandas as pd
    import numpy  as np
    
    ###
    helix_angle_original       = helix_angle
    mask                       = mask[:, :, np.newaxis]
    depth_factor               = wall_depth_factor
    kernel_size                = kernel_size
    rows                       = mask.shape[0]
    cols                       = mask.shape[1]
    slc                        = 0
    mean_endo_y                = np.zeros(mask.shape[2])
    mean_endo_x                = np.zeros(mask.shape[2])
    filled_mask                = np.zeros(mask.shape)
    concentric_contours_list   = []
    spatial_helix_angle_filter = np.copy(helix_angle_original)
    grid                       = Endo2Epi_Grid(mask)
    grid_nan                   = np.copy(grid)
    grid_nan[grid_nan == 0]    = np.nan
    ###
    [Epicardial_Contour,      Endocardial_Contour, _] = Myocardial_Mask_Contour_Extraction(mask, slc)
    endocardial_contour                               = Endocardial_Contour[0]
    epicardial_contour                                = Epicardial_Contour[0]
    [endocardial_x_original,  endocardial_y_original] = np.where(endocardial_contour == 1)
    mean_endo_y[slc]                                  = np.round(np.nanmean(endocardial_x_original))
    mean_endo_x[slc]                                  = np.round(np.nanmean(endocardial_y_original))
    temporary_mask                                    = np.copy(mask[:, :, slc])
    append_condition                                  = np.copy(mask[:, :, slc])
    while np.nansum(append_condition) > 0:
        epicardial_contour = Comprehensive_Outter_Contour(temporary_mask)
        epicardial_contour = epicardial_contour * 1
        append_condition   = (mask[:, :, slc]) * epicardial_contour
        if np.nansum(append_condition) > 0:
            concentric_contours_list.append(epicardial_contour * (mask[:, :, slc]))
        temporary_mask     = temporary_mask - epicardial_contour
    ###
    unsorted_data_list = []
    sorted_data_list   = []
    unsorted_data      = np.zeros([len(concentric_contours_list), 5])
    for depth in range(len(concentric_contours_list)):
        [x_original,  y_original] = np.where(concentric_contours_list[depth] == 1)
        transmurality             = np.zeros(x_original.shape)
        helix_angle               = np.zeros(x_original.shape)
        theta                     = np.zeros(x_original.shape)
        radius                    = np.zeros(x_original.shape)
        x_offset                  = np.int(mean_endo_x[slc])
        y_offset                  = np.int(mean_endo_y[slc])
        for index in range(x_original.shape[0]):
            point_static  = np.array([0, 50])                                                                                           # Static (reference) point
            point_origin  = np.array([0, 0])                                                                                            # Origin point
            point_moving  = np.array([x_original[index] - x_offset, y_original[index] - y_offset])                                      # Moving point
            vector_static = point_static - point_origin                                                                                 # Static (reference) vector
            vector_moving = point_moving - point_origin                                                                                 # Moving vector
            cosine_angle  = np.dot(vector_static, vector_moving) / (np.linalg.norm(vector_static) * np.linalg.norm(vector_moving))      # Cosine angle
            angle         = np.arccos(cosine_angle)                                                                                         # Calculate angle
            ########## Identify orientation of angle ##################################################################################################
            if (x_original[index] - x_offset < 0):                                                                                      # If moving point (x) is below reference (x offset) point ...
                angle = -angle                                                                                                              # Change angle polarity
            theta[index]         = np.degrees(angle)
            radius[index]        = math.dist(point_origin, point_moving)
            transmurality[index] = grid_nan[x_original[index],    y_original[index], slc]
            helix_angle[index]   = helix_angle_original[x_original[index], y_original[index]]
        unsorted_data = np.hstack((x_original[:, np.newaxis],
                                   y_original[:, np.newaxis],
                                   transmurality[:, np.newaxis],
                                   helix_angle[:, np.newaxis],
                                   radius[:, np.newaxis],
                                   theta[:, np.newaxis]))
        unsorted_data_list.append(unsorted_data)

        df = pd.DataFrame(unsorted_data_list[depth])
        df = df.sort_values(by = 5, axis = 0)

        sorted_data = df.to_numpy()
        if depth < int(np.round(len(concentric_contours_list) * depth_factor)):
            print("Autoadjust Epicardium")
            test = np.nanmean(sorted_data[:, 3]) + 2 * np.std(sorted_data[:, 3])
            if test > 15:
                test = 15
            for index in range(sorted_data.shape[0]):
                if sorted_data[index, 3] > test:
                    sorted_data[index, 3] = sorted_data[index, 3] * -1
        if depth > len(concentric_contours_list) - int(np.round(len(concentric_contours_list) * depth_factor)) - 1:
            print("Autoadjust Endocardium")
            test = np.nanmean(sorted_data[:, 3]) -  2 * np.std(sorted_data[:, 3])
            if test < -15:
                test = -15
            for index in range(sorted_data.shape[0]):
                if sorted_data[index, 3] < test:
                    sorted_data[index, 3] = sorted_data[index, 3] * -1
        sorted_data_list.append(sorted_data)

        rows = 0
        sorted_data_stacked = np.zeros([1, 6])
        for index in range(len(sorted_data_list)):
            sorted_data_stacked = np.vstack((sorted_data_stacked, sorted_data_list[index]))
            rows = rows + sorted_data_list[index].shape[0]
        sorted_data_stacked = np.delete(sorted_data_stacked, 0, 0)

        helix_angle_updated = np.copy(helix_angle_original)
        for index in range(sorted_data_stacked.shape[0]):
            x    = np.int(sorted_data_stacked[index, 0])
            y    = np.int(sorted_data_stacked[index, 1])
            if (helix_angle_updated[x, y] != sorted_data_stacked[index, 3]):
                helix_angle_updated[x, y] = sorted_data_stacked[index, 3]

        counter     = 0
        spatial_helix_angle_filter  = np.copy(helix_angle_updated)
        temp_matrix = np.copy(helix_angle_updated)
        for index in range(sorted_data_stacked.shape[0]):
            new_x  = np.int(sorted_data_stacked[index, 0])
            new_y  = np.int(sorted_data_stacked[index, 1])
            x      = np.int(np.floor((kernel_size - 1)/2))
            y      = np.int(np.floor((kernel_size - 1)/2))
            window = Moving_Window_2D(new_x, new_y, temp_matrix, kernel_size)
            if index < np.int(sorted_data_stacked.shape[0] / 2):
                mean_epi  = np.int(np.nanmean(sorted_data_list[0][:, 3]))
                window    = np.nan_to_num(window, nan = mean_epi)
            if index > np.int(sorted_data_stacked.shape[0] / 2):
                mean_endo = np.int(np.nanmean(sorted_data_list[-1][:, 3]))
                window    = np.nan_to_num(window, nan = mean_endo)
            new_window   = np.copy(window)
            window_mean  = np.nanmean(new_window)
            window_stdev = np.nanstd(new_window)
            if ((new_window[x, y] > 0 and window_mean > 0) or (new_window[x, y] < 0 and window_mean < 0)):
                continue
            else:
                if window_stdev > window_mean:
                    spatial_helix_angle_filter[new_x, new_y] = sorted_data_stacked[index, 3] * -1
                    counter                  = counter + 1
                else:
                    if ((new_window[x, y] > window_mean + 2 * window_stdev) or (new_window[x, y] < window_mean - 2 * window_stdev)):
                        spatial_helix_angle_filter[new_x, new_y] = sorted_data_stacked[index, 3] * -1
                        counter                  = counter + 1
                    else:
                        spatial_helix_angle_filter[new_x, new_y] = sorted_data_stacked[index, 3]
            temp_matrix[new_x, new_y] = spatial_helix_angle_filter[new_x, new_y]
    return spatial_helix_angle_filter

def Endo2Epi_Grid(LV_Mask):
    import skimage
    import numpy as np
    from scipy.interpolate import griddata

    rows    = LV_Mask.shape[0]
    columns = LV_Mask.shape[1]
    slices  = LV_Mask.shape[2]
    grid    = np.zeros([rows, columns, slices])

    for slc in range(slices):

        tmp_mask = np.ones([rows, columns]) - LV_Mask[:, :, slc]
        labels   = skimage.measure.label(tmp_mask, connectivity = 1, background = 0, return_num = True)

        Epi_Outside = np.zeros([rows, columns])
        Endo_inside = np.zeros([rows, columns])

        Epi_Outside[labels[0] == 1] = 1
        Endo_inside[labels[0] == 2] = 1

        [epi_x,  epi_y]             = np.where(Epi_Outside == 1)
        [endo_x, endo_y]            = np.where(Endo_inside == 1)
        epi    = np.stack((epi_y, epi_x), axis=1)
        endo   = np.stack((endo_y, endo_x), axis=1)
        points = np.vstack((epi,endo))

        epi_val  = np.ones((epi_x.shape[0], 1))
        endo_val = np.zeros((endo_x.shape[0], 1))
        values   = np.vstack((epi_val,endo_val))

        x = np.linspace(0, rows - 1,    rows)
        y = np.linspace(0, columns - 1, columns)
        xi, yi = np.meshgrid(y, x)
        tmp_grid = np.squeeze(griddata(points, values, (xi, yi), method='cubic')) * LV_Mask[:, :, slc]
        grid[:, :, slc] = tmp_grid
    return grid
