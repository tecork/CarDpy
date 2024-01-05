def sample_DICOMs():
    """
    ########## Definition Inputs ##################################################################################################################
    # image_space       : Matrix in image space.
    ########## Definition Outputs #################################################################################################################
    # k_space           : Matrix in k-space.
    """
    ########## Definition Information #############################################################################################################
    ### Originally written by John Pauly in MATLAB
    ### Converted to Python by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import os                                                      # Import numpy module
    main_path  = __file__
    main_path  = main_path.split('Data_Paths.py')[0]
    DICOM_path = os.path.join(main_path, 'DICOMS')
    return DICOM_path
def sample_NifTis():
    """
    ########## Definition Inputs ##################################################################################################################
    # image_space       : Matrix in image space.
    ########## Definition Outputs #################################################################################################################
    # k_space           : Matrix in k-space.
    """
    ########## Definition Information #############################################################################################################
    ### Originally written by John Pauly in MATLAB
    ### Converted to Python by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import os                                                      # Import numpy module
    main_path      = __file__
    main_path      = main_path.split('Data_Paths.py')[0]
    main_path      = os.path.join(main_path, 'NifTis')
    NifTi_path     = os.path.join(main_path, 'Sample_Data.nii')
    b_values_path  = os.path.join(main_path, 'Sample_Data.bvals')
    b_vectors_path = os.path.join(main_path, 'Sample_Data.bvecs')
    header_path    = os.path.join(main_path, 'Sample_Data.header')
    return [NifTi_path, b_values_path, b_vectors_path, header_path]
    
def sample_Output_Folder():
    """
    ########## Definition Inputs ##################################################################################################################
    # image_space       : Matrix in image space.
    ########## Definition Outputs #################################################################################################################
    # k_space           : Matrix in k-space.
    """
    ########## Definition Information #############################################################################################################
    ### Originally written by John Pauly in MATLAB
    ### Converted to Python by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    ########## Import Modules ######################################################################################################################
    import os                                                      # Import numpy module
    main_path   = __file__
    main_path   = main_path.split('Data_Paths.py')[0]
    output_path = os.path.join(main_path, 'Output')
    return output_path
