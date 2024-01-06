def Myocardial_Mask_Contour_Extractor(myocardial_mask, slice_number):
    """
    ########## Definition Inputs ##################################################################################################################
    # myocardial_mask       : Left or right ventricular binary mask (3D - [rows, columns, slices])
    # slice_number          : Index of slice to evaluate.
    ########## Definition Outputs #################################################################################################################
    # epicardial_contour    : Contour of the epicardium (outtermost layer) for current slice.
    # endocardial_contour   : Contour of the endocardium (innermost layer) for current slice.
    ########## Notes ##############################################################################################################################
    # This fucntion has been archived. Please use the more accurate definition named 'Myocardial_Mask_Contour_Extraction'.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2021
    ########## Import Modules #####################################################################################################################
    import numpy   as     np                                                    # Import numpy module
    from   skimage import measure                                               # Import measure from skimage module
    import cv2                                                                  # Import open CV module
    import warnings                                                             # Import warning module
    ########## Print Warning ######################################################################################################################
    warnings.warn('This fucntion has been archived. It is reccomended to use the more accurate definition named myocardial_mask_contour_extraction.')
    ########## Initalize data #####################################################################################################################
    rows                = myocardial_mask.shape[0]                              # Number of rows
    columns             = myocardial_mask.shape[1]                              # Number of columns
    slc                 = slice_number                                          # Index of current slice
    binary              = (myocardial_mask[:, :, slc] * 255).astype('uint8')    # Make image compadible with OpenCV
    epicardial_contour  = np.zeros(([rows, columns]))                           # Initialize epicardium
    endocardial_contour = np.zeros(([rows, columns]))                           # Initialize endocardium
    ########## Detect edges from input myocardial mask ############################################################################################
    if np.mean(binary) != 0:                                                    # If slice contains myocardium ...
        edges      = cv2.Canny(binary, 0 , 2)                                       # Edge detection with Canny filter
        all_labels = measure.label(edges)                                           # Label edges detected
        if 2 in all_labels:                                                         # If 2 labels were detected ...
            print('Slice %i contains endocardium and epicardium.' %int(slc))            # Print information statment about input data
            epicardial_contour  = all_labels == 1                                       # Identify epicardium contour
            endocardial_contour = all_labels == 2                                       # Identify endocardium contour
        else:                                                                       # Otherwise only 1 label was detected and slice is ignored
            print('Slice %i contains only epicardium.' %int(slc))                       # Print information statment about input data
    else:                                                                       # Otherwise myocardium was not detected and slice is ignored
        print('Slice %i contains no myocardium.'  %int(slc))                        # Print information statment about input data
    return [epicardial_contour, endocardial_contour]

def Myocardial_Mask_Contour_Extraction(mask, slice_number):
    """
    ########## Definition Inputs ##################################################################################################################
    # myocardial_mask      : Left or right ventricular binary mask (3D - [rows, columns, slices])
    # slice_number         : Index of slice to evaluate.
    ########## Definition Outputs #################################################################################################################
    # epicardial_contours  : Contour of the epicardium (outtermost layer) for current slice.
    # endocardial_contours : Contour of the endocardium (innermost layer) for current slice.
    # endocardial_centers  :
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules #####################################################################################################################
    import numpy                 as     np                                                                       #
    import skimage                                                                                               #
    from   cardpy.Tools.Contours import Comprehensive_Inner_Contour, Comprehensive_Outter_Contour                #
    ########## Initialize Mask Parameters #########################################################################################################
    rows                 = mask.shape[0]                                                                         #
    cols                 = mask.shape[1]                                                                         #
    slc                  = slice_number                                                                          #
    background_mask      = np.zeros([rows, cols])                                                                #
    myocardial_mask      = np.zeros([rows, cols])                                                                #
    blood_pool_mask      = np.zeros([rows, cols])                                                                #
    endocardial_centers  = []                                                                                    #
    epicardium_contours  = []                                                                                    #
    endocardium_contours = []                                                                                    #
    ########## Generate Masks #####################################################################################################################
    background_temp                   = np.zeros([rows, cols])                                                   #
    myocardium_temp                   = np.zeros([rows, cols])                                                   #
    blood_pool_temp                   = np.zeros([rows, cols])                                                   #
    mask_labels                       = skimage.measure.label(abs(mask[:, :, slc] - 1),
                                                              background   = None,
                                                              return_num   = False,
                                                              connectivity = 1)                                  #
    background_temp[mask_labels == 1] = 1                                                                        #
    myocardium_temp[mask_labels == 0] = 1                                                                        #
    blood_pool_temp[mask_labels == 2] = 1                                                                        #
    background_mask                   = background_temp                                                          #
    myocardial_mask                   = myocardium_temp                                                          #
    blood_pool_mask                   = blood_pool_temp                                                          #
    ########## Generate Masks #####################################################################################################################
    ########## Extract Epicardial and Endocardial Contours ########################################################################################
    [x_coordinates_endo,  y_coordinates_endo] = np.where(blood_pool_temp == 1)                                   #
    mean_y_coordinates_endo                   = int(np.round(np.mean(y_coordinates_endo)))                       #
    mean_x_coordinates_endo                   = int(np.round(np.mean(x_coordinates_endo)))                       #
    endocardial_centers.append([mean_x_coordinates_endo, mean_y_coordinates_endo])                               #
    epicardium_contour                        = Comprehensive_Outter_Contour(myocardial_mask)
    endocardium_contour                       = Comprehensive_Inner_Contour(myocardial_mask,
                                                                            int(endocardial_centers[0][0]),
                                                                            int(endocardial_centers[0][1]))      #
    endocardium_contour                       = endocardium_contour - (endocardium_contour * epicardium_contour) #
    epicardium_contours.append(epicardium_contour)                                                               #
    endocardium_contours.append(endocardium_contour)                                                             #
    return [epicardium_contours, endocardium_contours, endocardial_centers]

def Comprehensive_Inner_Contour(mask, offset_x, offset_y):
    import numpy as np
    rows = mask.shape[0]
    cols = mask.shape[1]

    pos_x_pos_y = np.zeros([rows, cols])
    pos_x_pos_y = np.zeros([rows, cols])
    for x in range(rows - offset_x):
        stop_flag = 'OFF'
        if mask[x + offset_x, 0 + offset_y] == 1:
            stop_flag = 'ON'
        for y in range(cols - offset_y):
            if stop_flag == 'OFF':
                if mask[x + offset_x, y + offset_y] == 0:
                    pos_x_pos_y[x + offset_x, y + offset_y] = 0
                if mask[x + offset_x, y + offset_y] == 1:
                    pos_x_pos_y[x + offset_x, y + offset_y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    pos_x_neg_y = np.zeros([rows, cols])
    for x in range(rows - offset_x):
        stop_flag = 'OFF'
        if mask[x + offset_x, offset_y - 0] == 1:
            stop_flag = 'ON'
        for y in range(offset_y + 1):
            if stop_flag == 'OFF':
                if mask[x + offset_x, offset_y - y] == 0:
                    pos_x_neg_y[x + offset_x, offset_y - y] = 0
                if mask[x + offset_x, offset_y - y] == 1:
                    pos_x_neg_y[x + offset_x, offset_y - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_x_neg_y = np.zeros([rows, cols])
    for x in range(offset_x + 1):
        stop_flag = 'OFF'
        if mask[offset_x - x, offset_y - 0] == 1:
            stop_flag = 'ON'
        for y in range(offset_y + 1):
            if stop_flag == 'OFF':
                if mask[offset_x - x, offset_y - y] == 0:
                    neg_x_neg_y[offset_x - x, offset_y - y] = 0
                if mask[offset_x - x, offset_y - y] == 1:
                    neg_x_neg_y[offset_x - x, offset_y - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_x_pos_y = np.zeros([rows, cols])
    for x in range(offset_x + 1):
        stop_flag = 'OFF'
        if mask[offset_x - x, 0 + offset_y] == 1:
            stop_flag = 'ON'
        for y in range(cols - offset_y):
            if stop_flag == 'OFF':
                if mask[offset_x - x, y + offset_y] == 0:
                    neg_x_pos_y[offset_x - x, y + offset_y] = 0
                if mask[offset_x - x, y + offset_y] == 1:
                    neg_x_pos_y[offset_x - x, y + offset_y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    ###
    pos_y_pos_x = np.zeros([rows, cols])
    for y in range(cols - offset_y):
        stop_flag = 'OFF'
        if mask[0 + offset_x, y + offset_y] == 1:
            stop_flag = 'ON'
        for x in range(rows - offset_x):
            if stop_flag == 'OFF':
                if mask[x + offset_x, y + offset_y] == 0:
                    pos_y_pos_x[x + offset_x, y + offset_y] = 0
                if mask[x + offset_x, y + offset_y] == 1:
                    pos_y_pos_x[x + offset_x, y + offset_y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_y_pos_x = np.zeros([rows, cols])
    for y in range(offset_y + 1):
        stop_flag = 'OFF'
        if mask[0 + offset_x, offset_y - y] == 1:
            stop_flag = 'ON'
        for x in range(rows - offset_x):
            if stop_flag == 'OFF':
                if mask[x + offset_x, offset_y - y] == 0:
                    neg_y_pos_x[x + offset_x, offset_y - y] = 0
                if mask[x + offset_x, offset_y - y] == 1:
                    neg_y_pos_x[x + offset_x, offset_y - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_y_neg_x = np.zeros([rows, cols])
    for y in range(offset_y + 1):
        stop_flag = 'OFF'
        if mask[offset_x - 0, offset_y - y] == 1:
            stop_flag = 'ON'
        for x in range(offset_x + 1):
            if stop_flag == 'OFF':
                if mask[offset_x - x, offset_y - y] == 0:
                    neg_y_neg_x[offset_x - x, offset_y - y] = 0
                if mask[offset_x - x, offset_y - y] == 1:
                    neg_y_neg_x[offset_x - x, offset_y - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    pos_y_neg_x = np.zeros([rows, cols])
    for y in range(cols - offset_y):
        stop_flag = 'OFF'
        if mask[offset_x - 0, y + offset_y] == 1:
            stop_flag = 'ON'
        for x in range(offset_x + 1):
            if stop_flag == 'OFF':
                if mask[offset_x - x, y + offset_y] == 0:
                    pos_y_neg_x[offset_x - x, y + offset_y] = 0
                if mask[offset_x - x, y + offset_y] == 1:
                    pos_y_neg_x[offset_x - x, y + offset_y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break

    x_all = pos_x_pos_y + pos_x_neg_y + neg_x_neg_y + neg_x_pos_y
    y_all = pos_y_pos_x + pos_y_neg_x + neg_y_neg_x + neg_y_pos_x
    contour = x_all + y_all
    contour[contour >= 1] = 1
    return contour

def Comprehensive_Outter_Contour(mask):
    import numpy as np

    rows = mask.shape[0]
    cols = mask.shape[1]

    pos_x_pos_y = np.zeros([rows, cols])
    for x in range(rows):
        stop_flag = 'OFF'
        for y in range(cols):
            if stop_flag == 'OFF':
                if mask[x, y] == 0:
                    pos_x_pos_y[x, y] = 0
                if mask[x, y] == 1:
                    pos_x_pos_y[x, y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    pos_x_neg_y = np.zeros([rows, cols])
    for x in range(rows):
        stop_flag = 'OFF'
        for y in range(cols):
            if stop_flag == 'OFF':
                if mask[x, cols - 1 - y] == 0:
                    pos_x_pos_y[x, cols - 1 - y] = 0
                if mask[x, cols - 1 - y] == 1:
                    pos_x_pos_y[x, cols - 1 - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_x_neg_y = np.zeros([rows, cols])
    for x in range(rows):
        stop_flag = 'OFF'
        for y in range(cols):
            if stop_flag == 'OFF':
                if mask[rows - 1 - x, cols - 1 - y] == 0:
                    pos_x_pos_y[rows - 1 - x, cols - 1 - y] = 0
                if mask[rows - 1 - x, cols - 1 - y] == 1:
                    pos_x_pos_y[rows - 1 - x, cols - 1 - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_x_pos_y = np.zeros([rows, cols])
    for x in range(rows):
        stop_flag = 'OFF'
        for y in range(cols):
            if stop_flag == 'OFF':
                if mask[rows - 1 - x, y] == 0:
                    pos_x_pos_y[rows - 1 - x, y] = 0
                if mask[rows - 1 - x, y] == 1:
                    pos_x_pos_y[rows - 1 - x, y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    ###
    pos_y_pos_x = np.zeros([rows, cols])
    for y in range(cols):
        stop_flag = 'OFF'
        for x in range(rows):
            if stop_flag == 'OFF':
                if mask[x, y] == 0:
                    pos_y_pos_x[x, y] = 0
                if mask[x, y] == 1:
                    pos_y_pos_x[x, y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    pos_y_neg_x = np.zeros([rows, cols])
    for y in range(cols):
        stop_flag = 'OFF'
        for x in range(rows):
            if stop_flag == 'OFF':
                if mask[rows - 1 - x, y] == 0:
                    pos_y_neg_x[rows - 1 - x, y] = 0
                if mask[rows - 1 - x, y] == 1:
                    pos_y_neg_x[rows - 1 - x, y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_y_neg_x = np.zeros([rows, cols])
    for y in range(cols):
        stop_flag = 'OFF'
        for x in range(rows):
            if stop_flag == 'OFF':
                if mask[rows - 1 - x, cols - 1 - y] == 0:
                    neg_y_neg_x[rows - 1 - x, cols - 1 - y] = 0
                if mask[rows - 1 - x, cols - 1 - y] == 1:
                    neg_y_neg_x[rows - 1 - x, cols - 1 - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    neg_y_pos_x = np.zeros([rows, cols])
    for y in range(cols):
        stop_flag = 'OFF'
        for x in range(rows):
            if stop_flag == 'OFF':
                if mask[x, cols - 1 - y] == 0:
                    neg_y_pos_x[x, cols - 1 - y] = 0
                if mask[x, cols - 1 - y] == 1:
                    neg_y_pos_x[x, cols - 1 - y] = 1
                    stop_flag = 'ON'
            if stop_flag == 'ON':
                break
    x_all = pos_x_pos_y + pos_x_neg_y + neg_x_neg_y + neg_x_pos_y
    y_all = pos_y_pos_x + pos_y_neg_x + neg_y_neg_x + neg_y_pos_x
    contour = x_all + y_all
    contour[contour >= 1] = 1
    return contour


def Angular_Contour_Organization(x_cordinates, y_cordinates, x_offset = 'None', y_offset = 'None'):
    """
    ########## Definition Inputs ##################################################################################################################
    # x_cordinates       : Cartersian x-coordinates of contour.
    # y_cordinates       : Cartersian y-coordinates of contour.
    # x_offset           : Center offset for x corrdinates (typically mean unless specified).
    # y_offset           : Center offset for y corrdinates (typically mean unless specified).
    ########## Definition Outputs #################################################################################################################
    # x_cordinates_polar : Sorted x-coordinates of contour based off angle from reference point.
    # y_cordinates_polar : Sorted y-coordinates of contour based off angle from reference point.
    # x_offset           : Center offset for x corrdinates (typically mean unless specified).
    # y_offset           : Center offset for y corrdinates (typically mean unless specified).
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2021
    ########## Import Modules #####################################################################################################################
    import numpy as np                                                                                                          #
    ########## Initialize angle array and calculate x and y offset ################################################################################
    angles   = np.zeros(len(x_cordinates))                                                                                      # Initialize angles array
    if x_offset == 'None':
        x_offset = np.mean(x_cordinates)                                                                                            # Calcualte x offset
    if y_offset == 'None':
        y_offset = np.mean(y_cordinates)                                                                                            # Calcualte y offset
    ########## Calcualte angle of x and y coodinates ##############################################################################################
    for idx in range(len(x_cordinates)):                                                                                        # Iterate through x and y coordinates
        point_static  = np.array([0, 50])                                                                                           # Static (reference) point
        point_origin  = np.array([0, 0])                                                                                            # Origin point
        point_moving  = np.array([x_cordinates[idx] - x_offset, y_cordinates[idx] - y_offset])                                      # Moving point
        vector_static = point_static - point_origin                                                                                 # Static (reference) vector
        vector_moving = point_moving - point_origin                                                                                 # Moving vector
        cosine_angle  = np.dot(vector_static, vector_moving) / (np.linalg.norm(vector_static) * np.linalg.norm(vector_moving))      # Cosine angle
        angle         = np.arccos(cosine_angle)                                                                                     # Calculate angle
        ########## Identify orientation of angle ##################################################################################################
        if (x_cordinates[idx] - x_offset < 0):                                                                                      # If moving point (x) is below reference (x offset) point ...
            angle = -angle                                                                                                              # Change angle polarity
        angles[idx] = np.degrees(angle)                                                                                             # Store angle
    ########## finalize  #####################################################################################################################
    polar_index        = angles.argsort()                                                                                       # Resort angles
    x_cordinates_polar = x_cordinates[polar_index[::-1]]                                                                        # Resorted x cordinates
    y_cordinates_polar = y_cordinates[polar_index[::-1]]                                                                        # Resorted y cordinates
    return [x_cordinates_polar, y_cordinates_polar, x_offset, y_offset]
    
def Contour_BSpline(x_data, y_data, num_interp_points = 200, smoothness_level = 'Low'):
    """
    ########## Definition Inputs ##################################################################################################################
    # x_data                : Cartersian x-coordinates of contour.
    # y_data                : Cartersian x-coordinates of contour.
    # num_interp_points     : Number of interpolated points to be generated. Default is set to 200 points.
    # smoothness_level      : Smoothing factor for B-spline data. Default is set to low.
                              Smoothless level options inclue Native (no smoothing), Low, Medium, High, and Extreme.
    ########## Definition Outputs #################################################################################################################
    # interpolated_x        : Interpolated cartersian x-coordinates of contour after B-Spline.
    # interpolated_y        : Interpolated cartersian y-coordinates of contour after B-Spline.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2021
    ########## Import Modules #####################################################################################################################
    import scipy.interpolate as interpolate                                                             #
    import numpy             as np                                                                      #
    ########## Assign Smoothness Factor ###########################################################################################################
    if smoothness_level == 'Native':                                                                    #
        smoothness_factor = 0                                                                               #
    if smoothness_level == 'Low':                                                                       #
        smoothness_factor = 10                                                                              #
    if smoothness_level == 'Medium':                                                                    #
        smoothness_factor = 20                                                                              #
    if smoothness_level == 'High':                                                                      #
        smoothness_factor = 40                                                                              #
    if smoothness_level == 'Extreme':                                                                   #
        smoothness_factor = 80                                                                              #
    ########## Initialize Interpolation ###########################################################################################################
    num_interp_points = num_interp_points + 1                                                           # Add additional datapoint to complete cicle for BSpline, removed at the end
    temporary_x       = x_data                                                                          # Temporary x data
    temporary_x       = np.append(temporary_x, temporary_x[0])                                          # Copy first x point to end of array to complete cicle for BSpline
    temporary_y       = y_data                                                                          # Temporary y data
    temporary_y       = np.append(temporary_y, temporary_y[0])                                          # Copy first y point to end of array to complete cicle for BSpline
    temporary_z       = np.linspace(0, len(x_data), len(x_data) + 1)                                    # Common axis for x and y data
    ########## Interpolate and Smooth x and y points ##############################################################################################
    interpolated_z = np.linspace(0, len(x_data + 1), num_interp_points)                                 # Common axis interpolated to desired number of points
    BSpline_x      = interpolate.splrep(temporary_z, temporary_x, s = smoothness_factor, per = True)    # Interpolate function with smoothing for x
    BSpline_y      = interpolate.splrep(temporary_z, temporary_y, s = smoothness_factor, per = True)    # Interpolate function with smoothing for y
    interpolated_x = interpolate.splev(interpolated_z, BSpline_x, der = 0)                              # X data interpolated to desired number of points
    interpolated_y = interpolate.splev(interpolated_z, BSpline_y, der = 0)                              # Y data interpolated to desired number of points
    interpolated_x = interpolated_x[:-1]                                                                # Remove last x data point that was initally added
    interpolated_y = interpolated_y[:-1]                                                                # Remove last x data point that was initally added
    return [interpolated_x, interpolated_y]
    
def Myocardial_Mask_Contour_Filler(myocardial_mask, endocardial_cordinates, epicardial_cordinates):
    import cv2
    import numpy                 as     np
    from   cardpy.Tools.Contours import Point2Pixel
    
    new_endocardial_x   = []
    new_endocardial_y   = []
    new_epicardial_x    = []
    new_epicardial_y    = []
    rows                = myocardial_mask.shape[0]
    columns             = myocardial_mask.shape[1]
    endocardial_contour = np.zeros([rows, columns])
    for index in range(endocardial_cordinates[0].shape[0]):
        [new_x_coordinate, new_y_coordinate] = Point2Pixel(epicardial_cordinates[0][index],  epicardial_cordinates[1][index])
        new_epicardial_x.append(new_x_coordinate)
        new_epicardial_y.append(new_y_coordinate)
        [new_x_coordinate, new_y_coordinate] = Point2Pixel(endocardial_cordinates[0][index], endocardial_cordinates[1][index])
        new_endocardial_x.append(new_x_coordinate)
        new_endocardial_y.append(new_y_coordinate)

    new_endocardium_points = [np.array(new_endocardial_x), np.array(new_endocardial_y)]
    new_epicardium_points  = [np.array(new_epicardial_x),  np.array(new_epicardial_y)]

    endocardial_points = np.hstack((new_endocardium_points[0][:, np.newaxis], new_endocardium_points[1][:, np.newaxis]))
    epicardial_points  = np.hstack((new_epicardium_points[0][:, np.newaxis],  new_epicardium_points[1][:, np.newaxis]))
    temp_matrix        = np.zeros([rows, columns])
    endocardial_mask   = cv2.fillPoly(temp_matrix, np.int32([endocardial_points]), color=(255, 255, 255))
    endocardial_mask   = endocardial_mask / 255
    temp_matrix        = np.zeros([rows, columns])
    epicardial_mask    = cv2.fillPoly(temp_matrix, np.int32([epicardial_points]),  color=(255, 255, 255))
    epicardial_mask    = epicardial_mask / 255
    
    for index in range(endocardial_points.shape[0]):
        x                         = endocardial_points[index, 0]
        y                         = endocardial_points[index, 1]
        endocardial_contour[y, x] = 1
    filled_myocardial_mask = epicardial_mask - endocardial_mask + endocardial_contour
    return filled_myocardial_mask
    
def Point2Pixel(x_coordinate, y_coordinate):
    import numpy as np
    import math
    point_x = x_coordinate
    point_y = y_coordinate
    
    floor_x = np.floor(point_x)
    floor_y = np.floor(point_y)
    ceil_x  = np.ceil(point_x)
    ceil_y  = np.ceil(point_y)
    
    point  = np.array([point_x, point_y])
    quad_1 = np.array([ceil_x,  floor_y])
    quad_2 = np.array([ceil_x,  ceil_y])
    quad_3 = np.array([floor_x, ceil_y])
    quad_4 = np.array([floor_x, floor_y])
    
    quadrant_list = [np.array([ceil_x,  floor_y]), np.array([ceil_x,  ceil_y]), np.array([floor_x, ceil_y]), np.array([floor_x, floor_y])]
    distance_list = []
    for quadrant in range(len(quadrant_list)):
        distance_list.append(math.dist(point, quadrant_list[quadrant]))
    
    quadrant_index = distance_list.index(min(distance_list))
    new_x_coordinate = int(quadrant_list[quadrant_index][0])
    new_y_coordinate = int(quadrant_list[quadrant_index][1])
    return[new_x_coordinate, new_y_coordinate]

def Moving_Window_2D(original_x, original_y, original_data, kernel_size):
    import numpy as np
    kernel_size = np.int(np.round(kernel_size))
    skip_flag   = 'OFF'
    if kernel_size < 3:
        kernel_size = int(3)
        skip_flag = 'ON'
#         print('Too small of kernel size selected. Set Kernal to minimum size of 3 (3x3 window kernel).')
    if kernel_size > 9:
        kernel_size = int(9)
        skip_flag = 'ON'
#         print('Too large of kernel size selected. Set Kernal to maximum size of 9 (9x9 window kernel).')
    if (kernel_size % 2) == 0:
        kernel_size_new = int(kernel_size + 1)
        skip_flag = 'ON'
#         print('Even Kernel Size selected. Increased Kernel from %i (%ix%i) to %i (%ix%i).' %(kernel_size, kernel_size, kernel_size, kernel_size_new, kernel_size_new, kernel_size_new))
        kernel_size = kernel_size_new
#    if (kernel_size % 2) == 1:
#        if skip_flag == 'OFF':
#            continue
#             print('Usable kernel size %i (%ix%i) has been detected.'%(kernel_size, kernel_size, kernel_size))
    
    temporary_matrix = np.copy(original_data)
    padded_array     = np.pad(temporary_matrix,
                              (kernel_size - np.int(np.ceil( kernel_size / 2)), kernel_size - np.int(np.ceil( kernel_size / 2))),
                              'constant',
                              constant_values = np.nan)
    
    new_x   = original_x + kernel_size - np.int(np.ceil( kernel_size / 2))
    new_y   = original_y + kernel_size - np.int(np.ceil( kernel_size / 2))

    minus_x = kernel_size - np.int(np.ceil( kernel_size / 2))
    plus_x  = kernel_size - np.int(np.floor(kernel_size / 2))
    minus_y = kernel_size - np.int(np.ceil( kernel_size / 2))
    plus_y  = kernel_size - np.int(np.floor(kernel_size / 2))

    window  = padded_array[new_x - minus_x:new_x + plus_x, new_y - minus_y:new_y + plus_y]
    
    return window
