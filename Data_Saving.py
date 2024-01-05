def Save_Diffusion_Image_Data(output_path, file_name, header, original_matrix, original_bvals, original_bvecs):
    """
    ########## Definition Inputs ##################################################################################################################
    output_path           : Path to save the data to.
    file_name             : Name of file to save data as.
    header                : Relavent header infomration from original format of data.
    original_matrix       : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvalues      : Sorted b-values.
    original_bvectors     : Sorted b-vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import os
    from   cardpy.Data_Sorting import sorted2stacked, stacked2sorted                                                                                # Import sorted to stacked and stacked to sorted from CarDpy
    import numpy               as     np
    import nibabel             as     nib
    
    [original_matrix_stacked, original_bvals_stacked, original_bvecs_stacked] = sorted2stacked(original_matrix, original_bvals, original_bvecs)     #

    if os.path.isdir(output_path) == False:
        os.makedirs(output_path)
    ########## Write B-Value File ##################################################################################################################
    bval_string = os.path.join(output_path, file_name + ".bvals")
    f1 = open(bval_string, "w")
    for i in range(len(original_bvals_stacked)):
        f1.write(str(round(original_bvals_stacked[i], 1)) + ' ')
    f1.close()
    ########## Write B-Vector File #################################################################################################################
    bvec_string = os.path.join(output_path, file_name + ".bvecs")
    f2 = open(bvec_string,"w")
    for i in range(len(original_bvecs_stacked)):
        f2.write(str(round(original_bvecs_stacked[i][0], 5)) + ' ')
    f2.write('\n')
    for i in range(len(original_bvecs_stacked)):
        f2.write(str(round(original_bvecs_stacked[i][1], 5)) + ' ')
    f2.write('\n')
    for i in range(len(original_bvecs_stacked)):
        f2.write(str(round(original_bvecs_stacked[i][2], 5)) + ' ')
    f2.close()
    ########## Write Header File ###################################################################################################################
    header_string = os.path.join(output_path, file_name + ".header")
    f3 = open(header_string, "w")
    if 'Manufacturer' in header:
        f3.write('Manufacturer:          ' + str(header['Manufacturer']) + '\n')
    if 'Scanner Model' in header:
        f3.write('Scanner Model:         ' + str(header['Scanner Model']) + '\n')
    if 'Magnet Strength' in header:
        f3.write('Magnet Strength:       ' + str(header['Magnet Strength']) + ' T\n')
    if 'Image Type' in header:
        f3.write('Image Type:            ' + str(header['Image Type']) + '\n')
    if 'Patient ID' in header:
        f3.write('Patient ID:            ' + str(header['Patient ID']) + '\n')
    if 'Patient Age' in header:
        f3.write('Patient Age:           ' + str(header['Patient Age']) + '\n')
    if 'Patient Sex' in header:
        f3.write('Patient Sex:           ' + str(header['Patient Sex']) + '\n')
    if 'Patient Weight' in header:
        if header['Patient Weight'] == '':
            f3.write('Patient Weight:        ' + str(header['Patient Weight']) + '\n')
        else:
            f3.write('Patient Weight:        ' + str(header['Patient Weight']) + ' kg\n')
    if 'Body Part' in header:
        f3.write('Body Part Examined:    ' + str(header['Body Part']) + '\n')
    if 'Slice Spacing' in header:
        f3.write('Slice Spacing:         ' + str(header['Slice Spacing']) + ' mm\n')
    if 'Echo Time' in header:
        f3.write('Echo Time:             ' + str(header['Echo Time']) + ' ms\n')
    if 'Repetition Time' in header:
        f3.write('Repetition Time:       ' + str(header['Repetition Time']) + ' ms\n')
    if 'Trigger Time' in header:
        f3.write('Trigger Time:          ' + str(header['Trigger Time']) + ' ms\n')
    if 'Phase Encode Polarity' in header:
        if header['Phase Encode Polarity'] == -1:
            f3.write('Phase Encode Polarity: ' + 'Negative' + '\n')
        if header['Phase Encode Polarity'] == 1:
            f3.write('Phase Encode Polarity: ' + 'Positive' + '\n')
    if 'Parallel Imaging' in header:
        f3.write('Parallel Imaging:      ' + str(header['Parallel Imaging']) + '\n')
    if 'Partial Fourier' in header:
        f3.write('Partial Fourier:       ' + str(header['Partial Fourier']) + '\n')
    f3.close()
    ########## Write matrix as NifTi ###############################################################################################################
    NifTi_string         = os.path.join(output_path, file_name + ".nii")
    affine_matrix        = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    img                  = nib.Nifti1Image(original_matrix_stacked, affine_matrix)
    img_header           = img.header
    img_header['pixdim'] = [1, header['X Resolution'], header['Y Resolution'], header['Z Resolution'], 1, 1, 1, 1]
    nib.save(img, NifTi_string)
    
def Save_Primary_Eigenvector_Data(output_path, header, original_matrix, original_bvals, original_bvecs):
    """
    ########## Definition Inputs ##################################################################################################################
    output_path           : Path to save the data to.
    header                : Relavent header infomration from original format of data.
    original_matrix       : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    original_bvalues      : Sorted b-values.
    original_bvectors     : Sorted b-vectors.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import os
    from   cardpy.Data_Processing.DTI import DTI_recon
    import numpy                      as     np
    import nibabel                    as     nib
    
    if os.path.isdir(output_path) == False:
        os.makedirs(output_path)
        
    [_, _, Evecs, _] = DTI_recon(original_matrix, original_bvals, original_bvecs)
    shape_3d         = Evecs['E1'].shape[0:3]
    rgb_arr          = (abs(Evecs['E1']) * 256).astype('u1')
    rgb_dtype        = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    rgb_typed        = rgb_arr.view(rgb_dtype).reshape(shape_3d)
    affine_matrix    = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    img                  = nib.Nifti1Image(rgb_typed, affine_matrix)
    img_header           = img.header
    img_header['pixdim'] = [1, header['X Resolution'], header['Y Resolution'], header['Z Resolution'], 1, 1, 1, 1]
    outpath              = os.path.join(output_path, 'Primary_Eigenvector.nii')
    nib.save(img, outpath)
    
def Save_NRRD_Segmentation(Mask, Header, file_path, file_name):
    from collections import OrderedDict
    import numpy     as     np
    import os
    import nrrd
    
    if os.path.exists(file_path) == False:
        os.makedirs(os.path.join(file_path))
    rows             = Mask.shape[0]
    cols             = Mask.shape[1]
    slcs             = Mask.shape[2]
    affine_matrix    = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    Voxel_Resolution = np.array([Header['X Resolution'], Header['Y Resolution'], Header['Z Resolution']])
    
    header = OrderedDict()
    header['type']             = 'unsigned short'
    header['dimension']        = slcs
    header['space']            = 'left-posterior-superior'
    header['sizes']            = np.array([rows, cols, slcs])
    header['space directions'] = np.abs(affine_matrix[0:3, 0:3] * Voxel_Resolution)
    header['kinds']            = ['domain', 'domain', 'domain']
    header['endian']           = 'little'
    header['encoding']         = 'gzip'
    header['space origin']     = np.array([0., 0., 0.])
    header['DICOM_0008_0060']  = '{"values":[{"t":0, "z":0, "value":"SEG"}]}'
    header['DICOM_0008_103E']  = '{"values":[{"t":0, "z":0, "value":"CarDpy Segmentation"}]}'
    
    file_name = file_name + '.nrrd'
    outpath   = os.path.join(file_path, file_name)
    nrrd.write(outpath, Mask, header)
    
