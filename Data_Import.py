def DICOM_Reader(dcm_path, info = 'ON'):
    """
    ########## Definition Inputs ##################################################################################################################
    dcm_path      : Path leading to DICOMs.
    info          : Information flag to show DICOM related information. Default is set to on.
    ########## Definition Outputs #################################################################################################################
    matrix        : 4D Matrix (Rows, Columns, Slices, Diffusion Directions) from DICOM folder.
    b_vals        : List containing all b values from DICOM folder.
    b_vecs        : List containing all b vectors from DICOM folder.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import glob
    import os
    import numpy              as     np
    import pydicom
    import nibabel            as     nib
    from   cardpy.Data_Import import VendorHeaders
    ########## Find DICOMs in Input Directory ######################################################################################################
    dcmPath = []                                                                                            # Initialize DICOM path list
    dcmEXT = '*.dcm'                                                                                        # DICOM extension
    for name in glob.glob(os.path.join(dcm_path, dcmEXT)):                                                  # Search for .dcm files in DICOM folder path
        dcmPath.append(name)                                                                                    # Append .dcm files to DICOM path list
    if dcmPath == []:                                                                                       # if DICOM path list is empty ...
        if info == 'ON':                                                                                        # If information flag is turned on ...
            print('No .dcm found: Trying .IMA')                                                                    # Print DCM not found
        imaEXT = '*.IMA'                                                                                        # IMA (Siemens) extension
        for name in glob.glob(os.path.join(dcm_path, imaEXT)):                                                  # Search for .IMA files in DICOM folder path
            dcmPath.append(name)                                                                                    # Append .IMA files to DICOM path list
    dcmPath.sort()                                                                                          # Sort DICOM path list
    
    ds     = pydicom.dcmread(dcmPath[0])                                                                      # Load Nth (ii) DICOM
    Header = VendorHeaders(ds)
    if len(Header['Slice Location']) > 1:
        numCols       = Header['Total Columns']                                                                                       # Initialize number of columns (y)
        numRows       = Header['Total Rows']
        numSlc        = len(Header['Slice Location'])
        slc_loc       = np.empty([len(dcmPath) * numSlc])                                                                # Initialize slice location array
        slc_loc[:]    = np.nan                                                                                  # Set slice location array to NaN
        slc_loc_list  = Header['Slice Location']
        b_val         = np.empty([len(dcmPath) * numSlc])                                                                # Initialize b value array
        b_val[:]      = np.nan                                                                                  # Set b value array to NaN
        b_vec         = np.empty([len(dcmPath) * numSlc, 3])                                                             # Initialize b vector array
        b_vec[:]      = np.nan                                                                                  # Set b vector array to NaN
        pe_pol        = np.empty([len(dcmPath) * numSlc])                                                                # Initialize phase encode polarity array
        pe_pol[:]     = np.nan                                                                                  # Set phase encode polarity array to NaN
        HeaderInfo    = np.empty([len(dcmPath) * numSlc, 6])                                                             # Initialize header information array
        HeaderInfo[:] = np.nan
        
        for ii in range(len(dcmPath)):                                                                          # Iterate through DICOMs
            ds  = pydicom.dcmread(dcmPath[ii])                                                                      # Load Nth (ii) DICOM
            Header = VendorHeaders(ds)                                                                              # Extract all meaningful DICOM tags for DTI
            for slc in range(numSlc):
                if Header['Slice Location'][slc] not in slc_loc:                                                             # Check if slice location information exists ...
                    slc_loc_list.append(Header['Slice Location'][slc])                                                           # If not, add slice location to slice location list
                index = (ii * numSlc) + slc
                slc_loc[index]    = Header['Slice Location'][slc]                                                               # Extract slice location
                b_val[index]      = Header['B Value']                                                                      # Store b value
                b_vec_1        = Header['B Vector 1']                                                                   # Extract b vector 1
                b_vec_2        = Header['B Vector 2']                                                                   # Extract b vector 2
                b_vec_3        = Header['B Vector 3']                                                                   # Extract b vector 3
                b_vec[index,:]    = [b_vec_1, b_vec_2, b_vec_3]                                                            # Store b vector information
                pe_pol[index]     = int(Header['Phase Encode Polarity'])                                                   # Extract phase encode polarity
                HeaderInfo[index] = [slc_loc[index], pe_pol[index], b_val[index], b_vec_1, b_vec_2, b_vec_3]                        # Store header information
            Diffusion_Information = HeaderInfo[:, 1::]                                                              # Extract diffusion information from header information
            Diffusion_Index       = np.unique(Diffusion_Information, axis = 0, return_index = True)[1]              # Identify orignal order of diffusion information by indexing
            UniDif                = np.array([Diffusion_Information[index] for index in sorted(Diffusion_Index)])   # Extract number of different diffusion directions with preserved order
            Diff_Dir_Counts       = np.unique(HeaderInfo[:, 1::], axis = 0, return_counts = True)[1]                # Extract number of counts (averages) for each diffusion direction
            numDir = int(len(dcmPath))                                                                     # Extract number of diffusion directions
            numAvg = int(min(Diff_Dir_Counts / numSlc))                                                             # Extract number of averages
        ########## Organize data into one matrix from DICOMs ###########################################################################################
        matrix = np.zeros([numRows, numCols, numSlc, numDir])                                                   # Initialize image matrix
        idxAvg = -1                                                                                             # Initialize average index
        idxDir = -1                                                                                             # Initialize direction index
        for ii in range(len(dcmPath)):                                                                          # Iterate through DICOMs
            ds             = pydicom.dcmread(dcmPath[ii])                                                           # Load Nth (ii) DICOM
            index          = (ii * numSlc)
            idxAvg_counter = np.all(HeaderInfo[0] == HeaderInfo[0:index + 1], axis = 1)                               # Identify current average from header information
            idxAvg         = np.count_nonzero(idxAvg_counter) - 1                                                   # Index current average
            for slc in range(numSlc):
                idxSlc = slc_loc_list.index(slc_loc[slc])
                index2 = index + slc
                idxDir = np.where((UniDif == HeaderInfo[index2, 1::]).all(axis = 1))[0][0] + (len(UniDif) * idxAvg)         # Index current diffuion direction
                matrix[:, :, idxSlc, idxDir] = ds.pixel_array[slc, :, :]                                                           # Extract Nth (idxSlc) slice from DICOM
        b_vals  = np.tile(UniDif[:, 1],   reps = numAvg)                                                        # Store b value information
        b_vecs  = np.tile(UniDif[:, 2::], reps = (numAvg,1))                                                    # Store b vector information
        info = 'ON'
        if info == 'ON':                                                                                        # If information flag is turned on ...
            print('Number of Diffusion Directions:', str(numDir))                                                   # Print dimension 4 (Diffusion Directions)
            print('Number of Slices:', str(numSlc))                                                                 # Print dimension 3 (Slices)
            print('Number of Columns:', str(numCols))                                                               # Print dimension 2 (Columns)
            print('Number of Rows:', str(numRows))                                                                  # Print dimension 1 (Rows)
        
    else:
        slc_loc       = np.empty([len(dcmPath)])                                                                # Initialize slice location array
        slc_loc[:]    = np.nan                                                                                  # Set slice location array to NaN
        slc_loc_list  = []                                                                                      # Initialize slice location list
        b_val         = np.empty([len(dcmPath)])                                                                # Initialize b value array
        b_val[:]      = np.nan                                                                                  # Set b value array to NaN
        b_vec         = np.empty([len(dcmPath), 3])                                                             # Initialize b vector array
        b_vec[:]      = np.nan                                                                                  # Set b vector array to NaN
        pe_pol        = np.empty([len(dcmPath)])                                                                # Initialize phase encode polarity array
        pe_pol[:]     = np.nan                                                                                  # Set phase encode polarity array to NaN
        HeaderInfo    = np.empty([len(dcmPath), 6])                                                             # Initialize header information array
        HeaderInfo[:] = np.nan                                                                                  # Set header information array to NaN
        numDir        = 0                                                                                       # Initialize number of diffusion directions (including b0)
        numSlc        = 0                                                                                       # Initialize number of slices
        numCols       = 0                                                                                       # Initialize number of columns (y)
        numRows       = 0
        
        for ii in range(len(dcmPath)):                                                                          # Iterate through DICOMs
            ds  = pydicom.dcmread(dcmPath[ii])                                                                      # Load Nth (ii) DICOM
            Header = VendorHeaders(ds)                                                                              # Extract all meaningful DICOM tags for DTI
            if Header['Slice Location'] not in slc_loc:                                                             # Check if slice location information exists ...
                slc_loc_list.append(Header['Slice Location'])                                                           # If not, add slice location to slice location list
            slc_loc[ii]    = Header['Slice Location'][0]                                                            # Extract slice location
            b_val[ii]      = Header['B Value']                                                                      # Store b value
            b_vec_1        = Header['B Vector 1']                                                                   # Extract b vector 1
            b_vec_2        = Header['B Vector 2']                                                                   # Extract b vector 2
            b_vec_3        = Header['B Vector 3']                                                                   # Extract b vector 3
            b_vec[ii,:]    = [b_vec_1, b_vec_2, b_vec_3]                                                            # Store b vector information
            pe_pol[ii]     = int(Header['Phase Encode Polarity'])                                                   # Extract phase encode polarity
            HeaderInfo[ii] = [slc_loc[ii], pe_pol[ii], b_val[ii], b_vec_1, b_vec_2, b_vec_3]                        # Store header information
        tot_rows          = Header['Total Rows']                                                                # Extract total number of rows in matrix
        tot_cols          = Header['Total Cols']                                                                # Extract total number of columns in matrix
        acq_mat           = Header['Acquisition Matrix']                                                        # Extract acquisition matrix information
        acq_rows          = Header['Acquisition Rows']                                                          # Extract number of rows in acquisition
        acq_cols          = Header['Acquisition Cols']                                                          # Extract number of columns in acquisition
        numRows           = acq_rows                                                                            # Re-define number of rows
        numCols           = acq_cols                                                                            # Re-define number of columns
        Diffusion_Information = HeaderInfo[:, 1::]                                                              # Extract diffusion information from header information
        Diffusion_Index       = np.unique(Diffusion_Information, axis = 0, return_index = True)[1]              # Identify orignal order of diffusion information by indexing
        UniDif                = np.array([Diffusion_Information[index] for index in sorted(Diffusion_Index)])   # Extract number of different diffusion directions with preserved order
        Diff_Dir_Counts       = np.unique(HeaderInfo[:, 1::], axis = 0, return_counts = True)[1]                # Extract number of counts (averages) for each diffusion direction
        if (Header['Mosaic'] == None):                                                                          # Check if one slice per matrix ...
            numSlc = len(np.unique(HeaderInfo[:, 0], axis = 0))                                                     # Extract number of slices
            numDir = int(len(dcmPath) / numSlc)                                                                     # Extract number of diffusion directions
            numAvg = int(min(Diff_Dir_Counts / numSlc))                                                             # Extract number of averages
        else:                                                                                                   # Or if there are multiple slices per matrix ...
            numSlc = Header['Mosaic']                                                                               # Extract number of slices per matrix
            numDir = len(dcmPath)                                                                                   # Extract number of diffusion directions
            numAvg = int(min(Diff_Dir_Counts))                                                                      # Extract number of averages
        ########## Organize data into one matrix from DICOMs ###########################################################################################
        matrix = np.zeros([numRows, numCols, numSlc, numDir])                                                   # Initialize image matrix
        idxAvg = -1                                                                                             # Initialize average index
        idxDir = -1                                                                                             # Initialize direction index
        for ii in range(len(dcmPath)):                                                                          # Iterate through DICOMs
            ds             = pydicom.dcmread(dcmPath[ii])                                                           # Load Nth (ii) DICOM
            idxAvg_counter = np.all(HeaderInfo[ii] == HeaderInfo[0:ii + 1], axis = 1)                               # Identify current average from header information
            idxAvg         = np.count_nonzero(idxAvg_counter) - 1                                                   # Index current average
            if (Header['Mosaic'] == None):                                                                          # Check if one slice per matrix ...
                idxSlc = slc_loc_list.index(slc_loc[ii])                                                                # Index current slice
                idxDir = np.where((UniDif == HeaderInfo[ii, 1::]).all(axis = 1))[0][0] + (len(UniDif) * idxAvg)         # Index current diffuion direction
                matrix[:, :, idxSlc, idxDir] = ds.pixel_array                                                           # Extract Nth (idxSlc) slice from DICOM
            else:                                                                                                   # Or if there are multiple slices (mosaic) per matrix ...
                row_ims = tot_rows / acq_rows                                                                           # Number of sub slices per row
                col_ims = tot_cols / acq_cols                                                                           # Number of sub slices per column
                idxDir  = idxDir + 1                                                                                    # Index current diffusion direction
                for idxSlc in range(numSlc):                                                                            # Index current slice
                    grid_x  = int(np.floor((idxSlc) / col_ims))                                                             # Identify slice's location in x grid of mosaic matrix
                    grid_y  = int(idxSlc - (col_ims * grid_x))                                                              # Identify slice's location in y grid of mosaic matrix
                    x_start = int(0 + acq_rows * grid_x)                                                                    # Define row (x) start point of mosaic matrix
                    x_stop  = int(acq_rows + acq_rows * grid_x)                                                             # Define row (x) stop point of mosaic matrix
                    y_start = int(0 + acq_cols * grid_y)                                                                    # Define column (y) start point of mosaic matrix
                    y_stop  = int(acq_cols + acq_cols * grid_y)                                                             # Define column (y) stop point of mosaic matrix
                    matrix[:, :, idxSlc, idxDir] = ds.pixel_array[x_start:x_stop, y_start:y_stop]                           # Extract Nth (idxSlc) slice from mosaic matrix
        if Header['Image Type'] == 'Phase':                                                                     # If image type is phase ...
            matrix = ((matrix / abs(Header['Rescale Intercept'])) - 0.5) * (Header['Rescale Slope'] * np.pi)        # Rescale and recenter for phase data
        ########## Organize b value and b vector information ###########################################################################################
        b_vals  = np.tile(UniDif[:, 1],   reps = numAvg)                                                        # Store b value information
        b_vecs  = np.tile(UniDif[:, 2::], reps = (numAvg,1))                                                    # Store b vector information
        if info == 'ON':                                                                                        # If information flag is turned on ...
            print('Number of Diffusion Directions:', str(numDir))                                                   # Print dimension 4 (Diffusion Directions)
            print('Number of Slices:', str(numSlc))                                                                 # Print dimension 3 (Slices)
            print('Number of Columns:', str(numCols))                                                               # Print dimension 2 (Columns)
            print('Number of Rows:', str(numRows))                                                                  # Print dimension 1 (Rows)
    return [matrix, b_vals, b_vecs, Header]

def VendorHeaders(dcm):
    """
    ########## Definition Inputs ##################################################################################################################
    # dcm           : DICOM
    ########## Definition Outputs #################################################################################################################
    # HeadersDict   : Dictionary containing header information.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import numpy as np                                                                                                                              # Import numpy module
    HeadersDict = dict()                                                                                                                            # Initialize header dictionary
    Vendor = dcm[0x00080070].value                                                                                                                  # Extract manufacturer from DICOM header
    ########## SIEMENS HEADER EXTRACTION ###########################################################################################################
    if (Vendor =='SIEMENS' or Vendor =='Siemens'):                                                                                                                          # If the vendor is Siemens...
        ### Software Version XA20 ###
        if dcm[0x00181020].value == 'syngo MR XA20':
            HeadersDict['Manufacturer']            = dcm[0x00080070].value                                                                                    # Extract manufacturer from DICOM header
            HeadersDict['Scanner Model']           = dcm[0x00081090].value                                                                                    # Extract scanner model from DICOM header
            HeadersDict['Magnet Strength']         = dcm[0x00180087].value                                                                                    # Extract magnet strength from DICOM header
            HeadersDict['Patient ID']              = dcm[0x00100020].value                                                                                    # Extract patient ID from DICOM header
            if dcm[0x00101010].value == '':
                HeadersDict['Patient Age'] = ''
            else:
                HeadersDict['Patient Age']             = [int(x) for x in dcm[0x00101010].value.split('Y') if x.isdigit()][0]                                 # Extract patient age from DICOM header
            HeadersDict['Patient Sex']             = dcm[0x00100040].value                                                                                    # Extract patient sex from DICOM header
            if dcm[0x00101020].value == None:
                HeadersDict['Patient Height'] = ''
            else:
                HeadersDict['Patient Height']          = float(round(dcm[0x00101020].value, 2))                                                               # Extract patient height from DICOM header
            if dcm[0x00101030].value == None:
                HeadersDict['Patient Weight'] = ''
            else:
                HeadersDict['Patient Weight']          = float(round(dcm[0x00101030].value, 2))                                                               # Extract patient weight from DICOM header
            HeadersDict['Body Part']               = dcm[0x00180015].value                                                                                    # Extract examination region from DICOM header
            HeadersDict['X Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00280030][0]                                                                                       # Extract x resolution from DICOM header
            HeadersDict['Y Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00280030][1]                                                                                        # Extract y resolution from DICOM header
            HeadersDict['Z Resolution']            = dcm[0x52009230][0][0x00289110][0][0x00180050].value                                                                                    # Extract z resolution from DICOM header
            HeadersDict['Echo Time']               = dcm[0x52009230][0][0x00189114][0][0x00189082].value                                                                                   # Extract echo time (TE) from DICOM header
            HeadersDict['Repetition Time']         = dcm[0x52009229][0][0x00189112][0][0x00180080].value                                                                                    # Extract repetition time (TR) from DICOM header
            HeadersDict['Scanner Orientation']     = dcm[0x52009229][0][0x002110fe][0][0x0021100c].value                                                                                  # Extract scanner orientation
            HeadersDict['Pixel Bandwidth']         = dcm[0x52009229][0][0x00189006][0][0x00180095].value
            HeadersDict['Parallel Imaging Factor'] = dcm[0x52009229][0][0x00189115][0][0x00189069].value
            HeadersDict['Parallel Imaging Type']   = dcm[0x52009229][0][0x00189115][0][0x00189078].value
            HeadersDict['Phase Encoding Direction'] = dcm[0x52009229][0][0x00189125][0][0x00181312].value
            if HeadersDict['Phase Encoding Direction'] == 'ROW':
                HeadersDict['Total Rows']    = dcm[0x52009229][0][0x00189125][0][0x00189058].value
                HeadersDict['Total Columns'] = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            else:
                HeadersDict['Total Columns'] = dcm[0x52009229][0][0x00189125][0][0x00189058].value
                HeadersDict['Total Rows']    = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            
            if dcm[0x52009230][0][0x002111fe][0][0x0021111c].value == 0:                                                    # If phase encoding direction is negative ...
                HeadersDict['Phase Encode Polarity'] = -1                                                                                                   # Set phase encode polarity to -1
            if dcm[0x52009230][0][0x002111fe][0][0x0021111c].value == 1:                                                    # If phase encoding direction is positive ...
                HeadersDict['Phase Encode Polarity'] = 1
            try:
                HeadersDict['Trigger Time']        = dcm[0x52009230][0][0x00189118][0][0x00209153].value
            except KeyError:
                HeadersDict['Trigger Time']        = 'N/A'

            HeadersDict['Echo Train Length']       = dcm[0x52009229][0][0x00189112][0][0x00189241].value
            HeadersDict['Phase Encode Steps']      = dcm[0x52009229][0][0x00189125][0][0x00189231].value
            HeadersDict['Partial Fourier']         = (HeadersDict['Echo Train Length'] * HeadersDict['Parallel Imaging Factor']) / HeadersDict['Phase Encode Steps']  # Extract Partial Fourier infromation from DICOM header
            if (HeadersDict['Partial Fourier'] > 3.5 / 8 and HeadersDict['Partial Fourier'] < 4.5 / 8):                                                 # If Partial Fourier is between 3.5/8 and 4.5/8 ...
                HeadersDict['Partial Fourier'] = '4/8'                                                                                                    # Set Partial Fourier to 4/8
            elif (HeadersDict['Partial Fourier'] > 4.5 / 8 and HeadersDict['Partial Fourier'] < 5.5 / 8):                                               # If Partial Fourier is between 4.5/8 and 5.5/8 ...
                HeadersDict['Partial Fourier'] = '5/8'                                                                                                    # Set Partial Fourier to 5/8
            elif (HeadersDict['Partial Fourier'] > 5.5 / 8 and HeadersDict['Partial Fourier'] < 6.5 / 8):                                               # If Partial Fourier is between 5.5/8 and 6.5/8 ...
                HeadersDict['Partial Fourier'] = '6/8'                                                                                                    # Set Partial Fourier to 6/8
            elif (HeadersDict['Partial Fourier'] > 6.5 / 8 and HeadersDict['Partial Fourier'] < 7.5 / 8):                                               # If Partial Fourier is between 6.5/8 and 7.5/8 ...
                HeadersDict['Partial Fourier'] = '7/8'                                                                                                    # Set Partial Fourier to 7/8
            elif (HeadersDict['Partial Fourier'] > 7.5 / 8 and HeadersDict['Partial Fourier'] < 8.5 / 8):                                               # If Partial Fourier is between 7.5/8 and 8.5/8 ...
                HeadersDict['Partial Fourier'] = '8/8'                                                                                                    # Set Partial Fourier to 8/8
            else:                                                                                                                                       # Otherwise ...
                HeadersDict['Partial Fourier'] = 'N/A'

            Slice_Position = []
            for slc in range(len(dcm[0x52009230].value)):
                Slice_Position.append(dcm[0x52009230][slc][0x002111fe][0][0x00211188].value)
            HeadersDict['Slice Location'] = Slice_Position
            HeadersDict['B Value']         =   int(dcm[0x52009230][0][0x00189117][0][0x00189087].value)
            if HeadersDict['B Value'] == 0:                                                                                                               # If b value equals 0 ...
                HeadersDict['B Vector 1'] = 0.0                                                                                                             # Set b vector 1 to 0
                HeadersDict['B Vector 2'] = 0.0                                                                                                             # Set b vector 2 to 0
                HeadersDict['B Vector 3'] = 0.0
            else:
                Orientation               = dcm[0x52009230][0][0x00209116][0][0x00200037].value                                                                                           # Extract patient orientation from DICOM Header
                Im_1                      = np.array(Orientation[0:3])                                                                                      # Extract image orientation 1
                Im_2                      = np.array(Orientation[3:6])                                                                                      # Extract image orientation 2
                Im_3                      = np.cross(Im_1, Im_2)                                                                                            # Cross product image orientaiton 1 and 2
                Diff_Dir                  = dcm[0x52009230][0][0x00189117][0][0x00189076][0][0x00189089].value                                                                                           # Extract diffusion direction from DICOM Header
                Im_1                      = np.expand_dims(Im_1, axis = 1)                                                                                  # Expand dimension of image orientation 1
                Im_2                      = np.expand_dims(Im_2, axis = 1)                                                                                  # Expand dimension of image orientation 2
                Im_3                      = np.expand_dims(Im_3, axis = 1)                                                                                  # Expand dimension of image orientation 3
                ReOr_Diff_Dir             = np.dot(np.hstack((Im_1, Im_2, Im_3)).T, Diff_Dir)                                                               # Correct diffusion directions based of patient orientation
                HeadersDict['B Vector 1'] = float(ReOr_Diff_Dir[0])                                                                                         # Extract b vector 1
                HeadersDict['B Vector 2'] = float(ReOr_Diff_Dir[1])                                                                                         # Extract b vector 2
                HeadersDict['B Vector 3'] = float(ReOr_Diff_Dir[2])

        ### Software Version VE11E ###
        if dcm[0x00181020].value == 'syngo MR E11':
            HeadersDict['Manufacturer']        = dcm[0x00080070].value                                                                                    # Extract manufacturer from DICOM header
            HeadersDict['Scanner Model']       = dcm[0x00081090].value                                                                                    # Extract scanner model from DICOM header
            HeadersDict['Magnet Strength']     = dcm[0x00180087].value                                                                                    # Extract magnet strength from DICOM header
            HeadersDict['Patient ID']          = dcm[0x00100020].value                                                                                    # Extract patient ID from DICOM header
            HeadersDict['Patient Age']         = [int(x) for x in dcm[0x00101010].value.split('Y') if x.isdigit()][0]                                     # Extract patient age from DICOM header
            HeadersDict['Patient Sex']         = dcm[0x00100040].value                                                                                    # Extract patient sex from DICOM header
            HeadersDict['Patient Height']      = float(round(dcm[0x00101020].value, 2))                                                                   # Extract patient height from DICOM header
            HeadersDict['Patient Weight']      = float(round(dcm[0x00101030].value, 2))                                                                   # Extract patient weight from DICOM header
            HeadersDict['Body Part']           = dcm[0x00180015].value                                                                                 # Extract examination region from DICOM header
            HeadersDict['X Resolution']        = dcm[0x00280030][0]                                                                                       # Extract x resolution from DICOM header
            HeadersDict['Y Resolution']        = dcm[0x00280030][1]                                                                                       # Extract y resolution from DICOM header
            HeadersDict['Z Resolution']        = dcm[0x00180050].value                                                                                    # Extract z resolution from DICOM header
            HeadersDict['Echo Time']           = dcm[0x00180081].value                                                                                    # Extract echo time (TE) from DICOM header
            HeadersDict['Repetition Time']     = dcm[0x00180080].value                                                                                    # Extract repetition time (TR) from DICOM header
            HeadersDict['Scanner Orientation'] = dcm[0x00511013].value                                                                                    # Extract scanner orientation

            if (((0x00281052) in dcm) == True):                                                                                                           # Check if data contains phase data ...
                HeadersDict['Image Type']        = 'Phase'                                                                                                  # If true, set data as phase
                HeadersDict['Rescale Intercept'] = dcm[0x00281052].value                                                                                    # Extract rescale intercept information from DICOM header
                HeadersDict['Rescale Slope']     = dcm[0x00281053].value                                                                                    # Extract rescale slope information from DICOM header
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Image Type']        = 'Magnitude'                                                                                              # Set data as magnitude

            if (((0x00180088) in dcm) == True):                                                                                                           # Check if data contains slice spacing ...
                HeadersDict['Slice Spacing']       = dcm[0x00180088].value                                                                                  # If true, extract slice spacing from DICOM header
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Slice Spacing']       = 'N/A'                                                                                                  # Set slice spacing to N/A

            if (((0x00181060) in dcm) == True):                                                                                                           # Check if DICOM header contains trigger time ...
                HeadersDict['Trigger Time'] = dcm[0x00181060].value                                                                                         # If true, extract trigger time from DICOM header
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Trigger Time'] = 'N/A'                                                                                                         # Set trigger time to N/A

            import nibabel.nicom.csareader as csareader                                                                                                   # Import csareader (Siemens) module
            Siemens_CSA_Private_Header1    = csareader.read(dcm[0x00291010].value)                                                                        # Read CSA tag in DICOM Header
            if Siemens_CSA_Private_Header1['tags']['PhaseEncodingDirectionPositive']['items'][0] == 0:                                                    # If phase encoding direction is negative ...
                HeadersDict['Phase Encode Polarity'] = -1                                                                                                   # Set phase encode polarity to -1
            if Siemens_CSA_Private_Header1['tags']['PhaseEncodingDirectionPositive']['items'][0] == 1:                                                    # If phase encoding direction is positive ...
                HeadersDict['Phase Encode Polarity'] = 1                                                                                                    # Set phase encode polarity to 1

            HeadersDict['Slice Location']  = [float(dcm[0x00201041].value)]                                                                               # Extract slice location from DICOM header
            HeadersDict['B Value']         =   int(dcm[0x0019100c].value)                                                                                 # Extract b value from DICOM header

            if HeadersDict['B Value'] == 0:                                                                                                               # If b value equals 0 ...
                HeadersDict['B Vector 1'] = 0.0                                                                                                             # Set b vector 1 to 0
                HeadersDict['B Vector 2'] = 0.0                                                                                                             # Set b vector 2 to 0
                HeadersDict['B Vector 3'] = 0.0                                                                                                             # Set b vector 3 to 0
            else:                                                                                                                                         # Otherwise...
                Orientation               = dcm[0x00200037].value                                                                                           # Extract patient orientation from DICOM Header
                Im_1                      = np.array(Orientation[0:3])                                                                                      # Extract image orientation 1
                Im_2                      = np.array(Orientation[3:6])                                                                                      # Extract image orientation 2
                Im_3                      = np.cross(Im_1, Im_2)                                                                                            # Cross product image orientaiton 1 and 2
                Diff_Dir                  = dcm[0x0019100e].value                                                                                           # Extract diffusion direction from DICOM Header
                Im_1                      = np.expand_dims(Im_1, axis = 1)                                                                                  # Expand dimension of image orientation 1
                Im_2                      = np.expand_dims(Im_2, axis = 1)                                                                                  # Expand dimension of image orientation 2
                Im_3                      = np.expand_dims(Im_3, axis = 1)                                                                                  # Expand dimension of image orientation 3
                ReOr_Diff_Dir             = np.dot(np.hstack((Im_1, Im_2, Im_3)).T, Diff_Dir)                                                               # Correct diffusion directions based of patient orientation
                HeadersDict['B Vector 1'] = float(ReOr_Diff_Dir[0])                                                                                         # Extract b vector 1
                HeadersDict['B Vector 2'] = float(ReOr_Diff_Dir[1])                                                                                         # Extract b vector 2
                HeadersDict['B Vector 3'] = float(ReOr_Diff_Dir[2])                                                                                         # Extract b vector 3

            HeadersDict['Total Rows'] = int(dcm[0x00280010].value)                                                                                        # Extract number of rows from DICOM Header
            HeadersDict['Total Cols'] = int(dcm[0x00280011].value)                                                                                        # Extract number of columns from DICOM Header

            HeadersDict['Acquisition Matrix'] = dcm[0x00181310].value                                                                                     # Extract acquisition matrix from DICOM Header
            HeadersDict['Acquisition Matrix'] = [x for x in HeadersDict['Acquisition Matrix'] if x != 0]                                                  # Remove zeros from acquisition matrix
            HeadersDict['Acquisition Rows']   = int(HeadersDict['Acquisition Matrix'][0])                                                                 # Extract acquisition rows from DICOM Header
            HeadersDict['Acquisition Cols']   = int(HeadersDict['Acquisition Matrix'][1])                                                                 # Extract acquisition columns from DICOM Header

            if (HeadersDict['Acquisition Rows'] > HeadersDict['Acquisition Cols'] and HeadersDict['Total Rows'] < HeadersDict['Total Cols']):             #
                HeadersDict['Acquisition Rows'] = int(HeadersDict['Acquisition Matrix'][1])                                                               #
                HeadersDict['Acquisition Cols'] = int(HeadersDict['Acquisition Matrix'][0])                                                               #

            if ((HeadersDict['Total Rows'] * HeadersDict['Total Cols']) != (HeadersDict['Acquisition Rows'] * HeadersDict['Acquisition Cols'])):          # Check if data is mosaic format ...
                HeadersDict['Mosaic'] = int(dcm[0x0019100a].value)                                                                                          # If so, extract number of mosaic slices from DICOM Header
            else:                                                                                                                                         # Otherwise ...
                if (HeadersDict['Total Rows'] == HeadersDict['Acquisition Cols'] and HeadersDict['Total Cols'] == HeadersDict['Acquisition Rows']):         # Check if single slice rows and columns are correct ...
                    HeadersDict['Acquisition Rows'] = int(HeadersDict['Acquisition Matrix'][1])                                                               # If not, switch rows ...
                    HeadersDict['Acquisition Cols'] = int(HeadersDict['Acquisition Matrix'][0])                                                               # and switch columns ...
                HeadersDict['Mosaic'] = None                                                                                                                # Set Mosaic to none

            if (((0x00511011) in dcm) == True):                                                                                                           # Check if data uses parallel imaging ...
                HeadersDict['Parallel Imaging'] = [int(x) for x in dcm[0x00511011].value.split('p') if x.isdigit()][0]                                      #
            else:                                                                                                                                         # Otherwise ...
                HeadersDict['Parallel Imaging'] = 1                                                                                                         # Set parallel imaging to 1

            HeadersDict['Partial Fourier'] = float((dcm[0x00180091].value * HeadersDict['Parallel Imaging']) / min(HeadersDict['Acquisition Matrix']))  # Extract Partial Fourier infromation from DICOM header
            if (HeadersDict['Partial Fourier'] > 3.5 / 8 and HeadersDict['Partial Fourier'] < 4.5 / 8):                                                 # If Partial Fourier is between 3.5/8 and 4.5/8 ...
                HeadersDict['Partial Fourier'] = '4/8'                                                                                                    # Set Partial Fourier to 4/8
            elif (HeadersDict['Partial Fourier'] > 4.5 / 8 and HeadersDict['Partial Fourier'] < 5.5 / 8):                                               # If Partial Fourier is between 4.5/8 and 5.5/8 ...
                HeadersDict['Partial Fourier'] = '5/8'                                                                                                    # Set Partial Fourier to 5/8
            elif (HeadersDict['Partial Fourier'] > 5.5 / 8 and HeadersDict['Partial Fourier'] < 6.5 / 8):                                               # If Partial Fourier is between 5.5/8 and 6.5/8 ...
                HeadersDict['Partial Fourier'] = '6/8'                                                                                                    # Set Partial Fourier to 6/8
            elif (HeadersDict['Partial Fourier'] > 6.5 / 8 and HeadersDict['Partial Fourier'] < 7.5 / 8):                                               # If Partial Fourier is between 6.5/8 and 7.5/8 ...
                HeadersDict['Partial Fourier'] = '7/8'                                                                                                    # Set Partial Fourier to 7/8
            elif (HeadersDict['Partial Fourier'] > 7.5 / 8 and HeadersDict['Partial Fourier'] < 8.5 / 8):                                               # If Partial Fourier is between 7.5/8 and 8.5/8 ...
                HeadersDict['Partial Fourier'] = '8/8'                                                                                                    # Set Partial Fourier to 8/8
            else:                                                                                                                                       # Otherwise ...
                HeadersDict['Partial Fourier'] = 'N/A'                                                                                                    # Set Patrial Fourier to N/A
    return HeadersDict
    
def NifTi_Reader(NifTi_path, b_values_path = None, b_vectors_path = None, header_path = None, info = 'ON'):
    from dipy.io.image      import load_nifti
    from dipy.io.gradients  import read_bvals_bvecs
    from cardpy.Data_Import import Header_Reader

    matrix, affine_matrix, voxel_resolution = load_nifti(NifTi_path, return_voxsize = True)
    Header                                  = Header_Reader(header_path)
    Header['X Resolution']                  = voxel_resolution[0]
    Header['Y Resolution']                  = voxel_resolution[1]
    Header['Z Resolution']                  = voxel_resolution[2]
    b_vals, b_vecs                          = read_bvals_bvecs(b_values_path, b_vectors_path)
    
    return [matrix, b_vals, b_vecs, Header, voxel_resolution, affine_matrix]
    
def Header_Reader(Header_Path):
    """
    ########## Definition Inputs ##################################################################################################################
    Header_Path   : Path CarDpy header (*.header) file.
    ########## Definition Outputs #################################################################################################################
    Headers_Dict  : Dictionary containing header information from CarDpy header (*.header) file.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    with open(Header_Path) as f:
        lines = f.readlines()
    HeadersDict = dict()
    for idx in range(len(lines)):
        key_word              = lines[idx].split(':')[0]
        if key_word != 'Scanner Model':
            key_value             = lines[idx].split(':')[1]
            key_value             = key_value.strip()
            key_value             = key_value.split(' ')[0]
        else:
            key_value             = lines[idx].split(':')[1]
            key_value             = key_value.strip()
        HeadersDict[key_word] = key_value
    return HeadersDict
