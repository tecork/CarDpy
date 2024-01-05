def IntERCOMS_GUI(original_matrix, mean_diffusivity, primary_eigenvector, Line_Width = None):
    from   sys                               import platform
    import tkinter                           as tk
    import matplotlib.pyplot                 as plt
    import numpy                             as np
    from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from   RangeSlider.RangeSlider           import RangeSliderV
    import sys
    from   cardpy.Colormaps                  import cDTI_Colormaps_Generator
    from   screeninfo import get_monitors

    
    
    global slc, image, Next_Button, Finish_Button, Image_Label
    global dummy_scale1_idx, dummy_scale2_idx
    global y_scale_idx, x_scale_idx
    global y_scale, x_scale
    global min_clim, max_clim, matrix
    global dummy1, dummy_canvas1
    global root, MEDIUMFONT, SMALLFONT, app_txt_col, app_bkg_col
    global Anterior_x, Anterior_y, ARVIP_Coordinates
    global Anterior_x1_Label, Anterior_y1_Label
    global Inferior_x, Inferior_y, IRVIP_Coordinates
    global Inferior_x1_Label, Inferior_y1_Label
    global var, conditions
    global Slice_Location, Anterior_RVIP, Inferior_RVIP
    global min_clim, max_clim, WindowMinEntry, WindowMaxEntry
    global image1, image2, image3, slc
    global Endo_Centers, Epi_Centers, Endo_Axes, Epi_Axes
    global x_center, y_center
    global Endo_Center_x1_Label, Endo_Center_y1_Label
    global Epi_Center_x1_Label, Epi_Center_y1_Label
    global Epi_Major_scale, Epi_Minor_scale
    global Endo_Major_scale, Endo_Minor_scale
    global dummy1, dummy3, dummy5
    global dummy_canvas1, dummy_canvas2, dummy_canvas3
    global matrix, MD_Map, E1_Map
    global Epi_Center_y, Epi_Center_x
    global Endo_Center_y, Endo_Center_x
    global cDTI_cmaps
    global monitor_scale, line_width
    

    matrix     = original_matrix
    MD_Map     = mean_diffusivity
    E1_Map     = primary_eigenvector
    cDTI_cmaps = cDTI_Colormaps_Generator()
    if Line_Width == None:
        line_width = int(np.round(monitor_scale * 10))
    else:
        line_width = Line_Width
    
    if matrix.shape[0] > matrix.shape[1]:
        matrix_ratio = matrix.shape[1] / matrix.shape[0]
        matrix_type = 'SQUARE'
        matrix_type = 'TALL'
        print(matrix_ratio)
        print(matrix_type)
    if matrix.shape[1] > matrix.shape[0]:
        matrix_ratio = matrix.shape[0] / matrix.shape[1]
        matrix_type = 'LONG'
        print(matrix_ratio)
        print(matrix_type)
    if matrix.shape[0] == matrix.shape[1]:
        matrix_ratio = 1
        matrix_type = 'SQUARE'
        print(matrix_ratio)
        print(matrix_type)
    slc = 0
    image1 = matrix[:, :, slc]
    image2 = MD_Map[:, :, slc]
    image3 = np.abs(E1_Map[:, :, slc, :])
    min_clim = np.int(np.round(np.min(image1)))
    max_clim = np.int(np.round(np.max(image1)))
    x_center = image1.shape[0]
    y_center = image1.shape[1]
#     image               = np.max(matrix[:, :, slc, :], axis = 2)
#     image_normalization = image / image.max()
    conditions   = [0, 0, 0, 0]
    Endo_Centers = []
    Epi_Centers  = []
    Endo_Axes    = []
    Epi_Axes     = []
    for ii in range(matrix.shape[2]):
        Endo_Centers.append([])
        Epi_Centers.append([])
        Endo_Axes.append([])
        Epi_Axes.append([])
    # root window
    root = tk.Tk()
    var = tk.IntVar()
    if platform == 'darwin':
        from tkmacosx import Button
    else:
        from tkinter import Button
        
    for m in get_monitors():
        print(str(m))
    monitor_height = m.height
    monitor_width  = m.width
    monitor_scale  = monitor_width / 5120
    monitor_scale  = monitor_width / 2560
    LARGEFONT  = ("Verdana", int(np.round(monitor_scale * 35)))
    MEDIUMFONT = ("Verdana", int(np.round(monitor_scale * 25)))
    SMALLFONT  = ("Verdana", int(np.round(monitor_scale * 15)))
    app_bkg_col    = '#1A2028'    # Dark Blue 1  (Notebook Background)
    app_txt_col    = '#FFEC8E'    # Yellow
    frame_bkg_col1 = '#30394A'    # Dark Blue 2  (Notebook Base 1)
    frame_bkg_col2 = '#363F4E'    # Dark Blue 3  (Notebook Base 2)
    frame_txt_col1 = '#fea47f'    # Orange
    frame_txt_col2 = '#e17e85'    # Light Red

    button_bkg_col  = app_bkg_col
    button_txt_col1 = '#B5C2D9'
    button_txt_col2 = '#B5C2D9'
    button_txt_col3 = '#B5C2D9'

    root.configure(bg = app_bkg_col)
    #Get the current screen width and height
    screen_width  = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    print(screen_width)
    print(screen_height)
    scale_factor = 0.95

    if screen_width > screen_height:
        screen_min = screen_height
    if screen_width > screen_height:
        screen_min = screen_height
    print(screen_min)
    if matrix_type == 'SQUARE':
        window_width  = int(screen_min * scale_factor)
        window_height = int(screen_min * scale_factor * matrix_ratio)
    if matrix_type == 'TALL':
        window_width  = int(screen_min * scale_factor)
        window_height = int(screen_min * scale_factor)
    if matrix_type == 'LONG':
        window_width  = int(screen_min * scale_factor)
        window_height = int(screen_min * scale_factor * matrix_ratio)

    geometry_string = str(window_width) + 'x' + str(window_height) + '+0+0'
    print(geometry_string)
    root.geometry(geometry_string)
    root.resizable(0, 0)
    root.title('INTeractive Elliptical Region Contouring of Myocardium Software')

    Image_Label = tk.Label(root, font = LARGEFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Short Axis (SAX) View of Slice ' + str(slc + 1))
    Image_Label.place(relheight = 0.1,
                      relwidth  = 1.0,
                      relx      = 0.0,
                      rely      = 0.0)

    #####
    Step0_Label = tk.Label(root, font = MEDIUMFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step0_Label.config(text = 'Directions')
    Step0_Label.place(relheight = 0.025,
                      relwidth  = 0.450,
                      relx      = 0.025,
                      rely      = 0.075)
    
    Step1_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step1_Label.config(text = 'Step 1: Set Window Level for Low b-Value Image')
    Step1_Label.place(relheight = 0.025,
                      relwidth  = 0.450,
                      relx      = 0.025,
                      rely      = 0.100)
    Window_Min_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Window_Min_Label.config(text = 'Window Min')
    Window_Min_Label.place(relheight = 0.025,
                           relwidth  = 0.075,
                           relx      = 0.025,
                           rely      = 0.125)
    Window_Max_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Window_Max_Label.config(text = 'Window Max')
    Window_Max_Label.place(relheight = 0.025,
                           relwidth  = 0.075,
                           relx      = 0.185,
                           rely      = 0.125)
    
    WindowMinEntry = tk.Entry(root, font = SMALLFONT,
                              justify = tk.CENTER, fg = '#000000', bg = '#ECECEC')
    WindowMinEntry.insert(0, min_clim)
    WindowMinEntry.place(relheight = 0.025,
                         relwidth  = 0.075,
                         relx      = 0.100,
                         rely      = 0.125)
    
    WindowMaxEntry = tk.Entry(root, font = SMALLFONT,
                              justify = tk.CENTER, fg = '#000000', bg = '#ECECEC')
    WindowMaxEntry.insert(0, max_clim)
    WindowMaxEntry.place(relheight = 0.025,
                         relwidth  = 0.075,
                         relx      = 0.260,
                         rely      = 0.125)

    Window_Button = Button(root, text = "Set Window Level", font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col, command = set_window)
    Window_Button.place(relheight = 0.025,
                        relwidth  = 0.130,
                        relx      = 0.345,
                        rely      = 0.125)

    Step2_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step2_Label.config(text = 'Step 2: Set Center of Endocardial (Endo) Elliptical Contour')
    Step2_Label.place(relheight = 0.025,
                      relwidth  = 0.45,
                      relx      = 0.025,
                      rely      = 0.150)
    Endo_Center_x_Label = tk.Label(root, font = SMALLFONT,
                                   fg = '#000000', bg = '#ECECEC')
    Endo_Center_x_Label.config(text = 'X Coordinate')
    Endo_Center_x_Label.place(relheight = 0.025,
                              relwidth  = 0.075,
                              relx      = 0.025,
                              rely      = 0.175)
    Endo_Center_y_Label = tk.Label(root, font = SMALLFONT,
                                   fg = '#000000', bg = '#ECECEC')
    Endo_Center_y_Label.config(text = 'Y Coordinate')
    Endo_Center_y_Label.place(relheight = 0.025,
                              relwidth  = 0.075,
                              relx      = 0.185,
                              rely      = 0.175)
    
    Endo_Coordinates = ['N/A', 'N/A']
    Endo_Center_x1_Label = tk.Label(root, font = SMALLFONT,
                                    fg = '#000000', bg = '#ECECEC')
    Endo_Center_x1_Label.config(text = Endo_Coordinates[0])
    Endo_Center_x1_Label.place(relheight = 0.025,
                               relwidth  = 0.075,
                               relx      = 0.100,
                               rely      = 0.175)
    Endo_Center_y1_Label = tk.Label(root, font = SMALLFONT,
                                    fg = '#000000', bg = '#ECECEC')
    Endo_Center_y1_Label.config(text = Endo_Coordinates[1])
    Endo_Center_y1_Label.place(relheight = 0.025,
                               relwidth  = 0.075,
                               relx      = 0.260,
                               rely      = 0.175)
    Endo_Button = Button(root, text = "Set Endo Center", font = SMALLFONT,
                         fg = app_txt_col, bg = app_bkg_col, command = select_Endo_Center)
    Endo_Button.place(relheight = 0.025,
                      relwidth  = 0.130,
                      relx      = 0.345,
                      rely      = 0.175)
    ### Set epicenters button
    Step3_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step3_Label.config(text = 'Step 3: Set Radii (X and Y) for Endocardial (Endo) Ellipse')
    Step3_Label.place(relheight = 0.025,
                      relwidth  = 0.45,
                      relx      = 0.025,
                      rely      = 0.200)
    Endo_Major_Label = tk.Label(root, font = SMALLFONT,
                                fg = app_txt_col, bg = app_bkg_col)
    Endo_Major_Label.config(text = 'Endo X Radius')
    Endo_Major_Label.place(relheight = 0.050,
                           relwidth  = 0.100,
                           relx      = 0.025,
                           rely      = 0.225)
    Endo_Major_scale = tk.Scale(root, from_ = 1, to = matrix.shape[0]/2, font = SMALLFONT,
                               bg = frame_bkg_col2, fg = '#ffffff', orient= tk.HORIZONTAL,
                               command = update_plots)
    Endo_Major_scale.set(int(matrix.shape[0]/2) * 0.10)
    Endo_Major_scale_idx   = Endo_Major_scale.get()
    Endo_Major_scale.place(relheight = 0.050,
                           relwidth  = 0.350,
                           relx      = 0.125,
                           rely      = 0.225)
    Endo_Minor_Label = tk.Label(root, font = SMALLFONT,
                                fg = app_txt_col, bg = app_bkg_col)
    Endo_Minor_Label.config(text = 'Endo Y Radius')
    Endo_Minor_Label.place(relheight = 0.050,
                           relwidth  = 0.100,
                           relx      = 0.025,
                           rely      = 0.275)
    Endo_Minor_scale = tk.Scale(root, from_ = 1, to = matrix.shape[0]/2, font = SMALLFONT,
                               bg = frame_bkg_col2, fg = '#ffffff', orient= tk.HORIZONTAL,
                               command = update_plots)
    Endo_Minor_scale.set(int(matrix.shape[1]/2)  * 0.10)
    Endo_Minor_scale_idx   = Endo_Minor_scale.get()
    Endo_Minor_scale.place(relheight = 0.050,
                           relwidth  = 0.350,
                           relx      = 0.125,
                           rely      = 0.275)
    Endo_Contour_Button = Button(root, text = "Set Endocardium Contour", font = SMALLFONT,
                         fg = app_txt_col, bg = app_bkg_col, command = select_Endo_Contour)
    Endo_Contour_Button.place(relheight = 0.025,
                              relwidth  = 0.45,
                              relx      = 0.025,
                              rely      = 0.325)
    Step4_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step4_Label.config(text = 'Step 4: Set Center of Epicardial (Epi) Elliptical Contour')
    Step4_Label.place(relheight = 0.025,
                      relwidth  = 0.45,
                      relx      = 0.025,
                      rely      = 0.350)
    Epi_Center_x_Label = tk.Label(root, font = SMALLFONT,
                                  fg = '#000000', bg = '#ECECEC')
    Epi_Center_x_Label.config(text = 'X Coordinate')
    Epi_Center_x_Label.place(relheight = 0.025,
                             relwidth  = 0.075,
                             relx      = 0.025,
                             rely      = 0.375)
    Epi_Center_y_Label = tk.Label(root, font = SMALLFONT,
                                  fg = '#000000', bg = '#ECECEC')
    Epi_Center_y_Label.config(text = 'Y Coordinate')
    Epi_Center_y_Label.place(relheight = 0.025,
                             relwidth  = 0.075,
                             relx      = 0.185,
                             rely      = 0.375)
    
    Epi_Coordinates = ['N/A', 'N/A']
    Epi_Center_x1_Label = tk.Label(root, font = SMALLFONT,
                                   fg = '#000000', bg = '#ECECEC')
    Epi_Center_x1_Label.config(text = Epi_Coordinates[0])
    Epi_Center_x1_Label.place(relheight = 0.025,
                              relwidth  = 0.075,
                              relx      = 0.100,
                              rely      = 0.375)
    Epi_Center_y1_Label = tk.Label(root, font = SMALLFONT,
                                   fg = '#000000', bg = '#ECECEC')
    Epi_Center_y1_Label.config(text = Epi_Coordinates[1])
    Epi_Center_y1_Label.place(relheight = 0.025,
                              relwidth  = 0.075,
                              relx      = 0.260,
                              rely      = 0.375)
    Epi_Button = Button(root, text = "Set Epi Center", font = SMALLFONT,
                        fg = app_txt_col, bg = app_bkg_col, command = select_Epi_Center)
    Epi_Button.place(relheight = 0.025,
                     relwidth  = 0.130,
                     relx      = 0.345,
                     rely      = 0.375)
    Step5_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step5_Label.config(text = 'Step 5: Set Radii (X and Y) for Epicardial (Epi) Ellipse')
    Step5_Label.place(relheight = 0.025,
                      relwidth  = 0.45,
                      relx      = 0.025,
                      rely      = 0.400)
    Epi_Major_Label = tk.Label(root, font = SMALLFONT,
                               fg = app_txt_col, bg = app_bkg_col)
    Epi_Major_Label.config(text = 'Epi X Radius')
    Epi_Major_Label.place(relheight = 0.050,
                          relwidth  = 0.100,
                          relx      = 0.025,
                          rely      = 0.425)
    Epi_Major_scale = tk.Scale(root, from_ = 1, to = matrix.shape[0]/2, font = SMALLFONT,
                               bg = frame_bkg_col2, fg = '#ffffff', orient= tk.HORIZONTAL,
                               command = update_plots)
    Epi_Major_scale.set(int(matrix.shape[0]/2) * 0.2)
    Epi_Major_scale_idx   = Epi_Major_scale.get()
    Epi_Major_scale.place(relheight = 0.050,
                          relwidth  = 0.350,
                          relx      = 0.125,
                          rely      = 0.425)
    Epi_Minor_Label = tk.Label(root, font = SMALLFONT,
                               fg = app_txt_col, bg = app_bkg_col)
    Epi_Minor_Label.config(text = 'Epi Y Radius')
    Epi_Minor_Label.place(relheight = 0.050,
                          relwidth  = 0.100,
                          relx      = 0.025,
                          rely      = 0.475)
    Epi_Minor_scale = tk.Scale(root, from_ = 1, to = matrix.shape[0]/2, font = SMALLFONT,
                               bg = frame_bkg_col2, fg = '#ffffff', orient= tk.HORIZONTAL,
                               command = update_plots)
    Epi_Minor_scale.set(int(matrix.shape[1]/2) * 0.2)
    Epi_Minor_scale_idx   = Epi_Minor_scale.get()
    Epi_Minor_scale.place(relheight = 0.050,
                          relwidth  = 0.350,
                          relx      = 0.125,
                          rely      = 0.475)
    Epi_Contour_Button = Button(root, text = "Set Epicardium Contour", font = SMALLFONT,
                                fg = app_txt_col, bg = app_bkg_col, command = select_Epi_Contour)
    Epi_Contour_Button.place(relheight = 0.025,
                             relwidth  = 0.45,
                             relx      = 0.025,
                             rely      = 0.525)

    #####
    x_scale = tk.Scale(root, from_ = 0, to = matrix.shape[0] - 1, font = SMALLFONT,
                      bg = frame_bkg_col2, fg = '#ffffff',
                      command = update_plots)
    x_scale.set(int(matrix.shape[0]/2))
    x_scale_idx   = x_scale.get()
    x_scale.place(relheight = 0.40,
                  relwidth  = 0.05,
                  relx      = 0.525,
                  rely      = 0.100)

    y_scale = tk.Scale(root, from_ = 0, to = matrix.shape[1] - 1, font = SMALLFONT,
                      bg = frame_bkg_col2, fg = '#ffffff', orient = tk.HORIZONTAL,
                      command = update_plots)
    y_scale.set(int(matrix.shape[1]/2))
    y_scale_idx   = y_scale.get()
    y_scale.place(relheight = 0.05,
                  relwidth  = 0.40,
                  relx      = 0.575,
                  rely      = 0.500)
    ###
    Lowbval_Label = tk.Label(root, font = MEDIUMFONT,
                             fg = app_txt_col, bg = app_bkg_col)
    Lowbval_Label.config(text = 'Low b-value Image')
    Lowbval_Label.place(relheight = 0.025,
                        relwidth  = 0.4,
                        relx      = 0.575,
                        rely      = 0.075)
    
    Epi_Center_y  = int(matrix.shape[1]/2)
    Epi_Center_x  = int(matrix.shape[0]/2)
    Endo_Center_y = int(matrix.shape[1]/2)
    Endo_Center_x = int(matrix.shape[0]/2)
    #####
    dummy_figure1 = plt.Figure(facecolor = '#ECECEC', tight_layout = True)
    dummy1        = dummy_figure1.add_subplot(111)
    dummy2        = dummy1.imshow(image1, cmap = 'gray')
#     min_clim      = WindowMinEntry.get()
#     max_clim      = WindowMaxEntry.get()
    dummy2.set_clim([min_clim, max_clim])
    dummy1.set_aspect('equal')
    
    dummy_scale1_idx = matrix.shape[0]/2
    dummy_scale2_idx = matrix.shape[1]/2
    dummy1.plot(y_center, x_center,
                marker = 'o', color = 'red', markersize = int(np.round(monitor_scale * 10)))

    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)
    dummy_canvas1 = FigureCanvasTkAgg(dummy_figure1, root)
    dummy_canvas1.get_tk_widget().place(relheight = 0.4,
                                        relwidth  = 0.4,
                                        relx      = 0.575,
                                        rely      = 0.100)

    ###
    Meandiff_Label = tk.Label(root, font = MEDIUMFONT,
                              fg = app_txt_col, bg = app_bkg_col)
    Meandiff_Label.config(text = 'Mean Diffusivity Map')
    Meandiff_Label.place(relheight = 0.025,
                         relwidth  = 0.4,
                         relx      = 0.575,
                         rely      = 0.550)
    #####
    dummy_figure2 = plt.Figure(facecolor = '#ECECEC', tight_layout = True)
    dummy3        = dummy_figure2.add_subplot(111)
    dummy4        = dummy3.imshow(image2, cmap = cDTI_cmaps['MD'])
#     min_clim      = WindowMinEntry.get()
#     max_clim      = WindowMaxEntry.get()
    dummy4.set_clim([0, 3])
    dummy3.set_aspect('equal')
    
    dummy_scale1_idx = matrix.shape[0]/2
    dummy_scale2_idx = matrix.shape[1]/2
    dummy3.plot(y_center, x_center,
                marker = 'o', color = 'k', markersize = int(np.round(monitor_scale * 10)))

    dummy3.axes.xaxis.set_visible(False)
    dummy3.axes.yaxis.set_visible(False)
    dummy3.spines.top.set_visible(False)
    dummy3.spines.left.set_visible(False)
    dummy3.spines.bottom.set_visible(False)
    dummy3.spines.right.set_visible(False)
    dummy_canvas2 = FigureCanvasTkAgg(dummy_figure2, root)
    dummy_canvas2.get_tk_widget().place(relheight = 0.4,
                                        relwidth  = 0.4,
                                        relx      = 0.575,
                                        rely      = 0.575)

    Primary_E1_Label = tk.Label(root, font = MEDIUMFONT,
                                fg = app_txt_col, bg = app_bkg_col)
    Primary_E1_Label.config(text = 'Primary Eigenvector Map')
    Primary_E1_Label.place(relheight = 0.025,
                           relwidth  = 0.4,
                           relx      = 0.025,
                           rely      = 0.550)
    #####
    dummy_figure3 = plt.Figure(facecolor = '#ECECEC', tight_layout = True)
    dummy5        = dummy_figure3.add_subplot(111)
    dummy6        = dummy5.imshow(np.abs(image3))
#     min_clim      = WindowMinEntry.get()
#     max_clim      = WindowMaxEntry.get()
#     dummy4.set_clim([0, 3])
    dummy5.set_aspect('equal')
    
    dummy_scale1_idx = matrix.shape[0]/2
    dummy_scale2_idx = matrix.shape[1]/2
    dummy5.plot(y_center, x_center,
                marker = 'o', color = 'k', markersize = int(np.round(monitor_scale * 10)))

    dummy5.axes.xaxis.set_visible(False)
    dummy5.axes.yaxis.set_visible(False)
    dummy5.spines.top.set_visible(False)
    dummy5.spines.left.set_visible(False)
    dummy5.spines.bottom.set_visible(False)
    dummy5.spines.right.set_visible(False)
    dummy_canvas3 = FigureCanvasTkAgg(dummy_figure3, root)
    dummy_canvas3.get_tk_widget().place(relheight = 0.40,
                                        relwidth  = 0.40,
                                        relx      = 0.025,
                                        rely      = 0.575)
        
    if slc == matrix.shape[2] - 1:
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                               fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.1,
                            relwidth  = 0.1,
                            relx      = 0.45,
                            rely      = 0.875)
        Finish_Button["state"] = "disabled"
    else:
        Next_Button = Button(root, text = "Next", font = SMALLFONT,
                             fg = app_txt_col, bg = app_bkg_col, command = next_slice)
        Next_Button.place(relheight = 0.1,
                          relwidth  = 0.1,
                          relx      = 0.45,
                          rely      = 0.875)
        Next_Button["state"] = "disabled"
    root.mainloop()
    return [Endo_Centers, Endo_Axes, Epi_Centers, Epi_Axes]

def set_window():
    global max_clim, min_clim
    min_clim = int(WindowMinEntry.get())
    max_clim = int(WindowMaxEntry.get())
    update_plots(var)
def select_Endo_Center():
    import numpy as np
    global Endo_Center, Endo_Center_x, Endo_Center_y
    global Endo_Centers, conditions
    global Endo_Center_x1_Label, Endo_Center_y1_Label
    global conditions
    global x_center, y_center

    Endo_Center_x = np.int(np.round(x_center))
    Endo_Center_y = np.int(np.round(y_center))
    Endo_Center   = [Endo_Center_x, Endo_Center_y]
    Endo_Center_x1_Label.config(text = Endo_Center[0])
    Endo_Center_y1_Label.config(text = Endo_Center[1])
    conditions[0] = 1
    Endo_Centers[slc] = Endo_Center
    update_plots(var)
    if conditions == [1, 1, 1, 1]:
        if slc == matrix.shape[2] - 1:
            Finish_Button["state"] = "normal"
        else:
            Next_Button["state"] = "normal"
    return Endo_Center
def select_Endo_Contour():
    import numpy as np
    global Endo_Axes, conditions
    global endo_width, endo_height
    
    Endo_x_axis = endo_height
    Endo_y_axis = endo_width
    Endo_Axis   = [Endo_x_axis, Endo_y_axis]
    Endo_Axes[slc] = [Endo_x_axis, Endo_y_axis]

    conditions[1] = 1
    if conditions == [1, 1, 1, 1]:
        if slc == matrix.shape[2] - 1:
            Finish_Button["state"] = "normal"
        else:
            Next_Button["state"] = "normal"
    return Endo_Axis
def select_Epi_Center():
    import numpy as np
    global Epi_Center, Epi_Center_x, Epi_Center_y
    global Epi_Centers, conditions
    global Epi_Center_x1_Label, Epi_Center_y1_Label
    global conditions
    global x_center, y_center
    
    Epi_Center_x = np.int(np.round(x_center))
    Epi_Center_y = np.int(np.round(y_center))
    Epi_Center   = [Epi_Center_x, Epi_Center_y]
    Epi_Center_x1_Label.config(text = Epi_Center[0])
    Epi_Center_y1_Label.config(text = Epi_Center[1])
    conditions[2] = 1
    Epi_Centers[slc] = Epi_Center
    update_plots(var)
    if conditions == [1, 1, 1, 1]:
        if slc == matrix.shape[2] - 1:
            Finish_Button["state"] = "normal"
        else:
            Next_Button["state"] = "normal"
    return Epi_Centers
def select_Epi_Contour():
    import numpy as np
    global Epi_Axes, conditions
    global epi_width, epi_height

    Epi_x_axis = epi_height
    Epi_y_axis = epi_width
    Epi_Axis   = [Epi_x_axis, Epi_y_axis]
    Epi_Axes[slc] = [Epi_x_axis, Epi_y_axis]
    conditions[3] = 1
    if conditions == [1, 1, 1, 1]:
        if slc == matrix.shape[2] - 1:
            Finish_Button["state"] = "normal"
        else:
            Next_Button["state"] = "normal"

    return Epi_Axis
    
def next_slice():
    import tkinter as tk
    from sys import platform
    import numpy as np
    from   cDTIpy.Colormaps.Diffusion import cDTI_Colormaps_Generator
    if platform == 'darwin':
        from tkmacosx import Button
    else:
        from tkinter import Button
    global slc, image, Next_Button, Finish_Button, Image_Label
    global dummy_scale1_idx, dummy_scale2_idx
    global dummy1
    global ARVIP_Coordinates, Anterior_x, Anterior_y
    global Anterior_x1_Label, Anterior_y1_Label
    global IRVIP_Coordinates, Inferior_x, Inferior_y
    global Inferior_x1_Label, Inferior_y1_Label
    global conditions, Slice_Location
    global var
    global y_scale_idx, x_scale_idx
    global y_scale, x_scale
    global image1, image2, image3
    global x_center, y_center
    global Epi_Center_x, Epi_Center_y
    global Endo_Center_x, Endo_Center_y
    global cDTI_cmaps


    slc = slc + 1
#     image               = np.max(matrix[:, :, slc, :], axis = 2)
#     image_normalization = image / image.max()
    image1 = matrix[:, :, slc]
    image2 = MD_Map[:, :, slc]
    image3 = np.abs(E1_Map[:, :, slc, :])
    
    Image_Label.destroy()
    Image_Label = tk.Label(root, font = MEDIUMFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Short Axis (SAX) View of Slice ' + str(slc + 1))
    Image_Label.place(relheight = 0.1,
                  relwidth  = 1.0,
                  relx      = 0.0,
                  rely      = 0.0)
    conditions = [0, 0, 0, 0]
    Endo_Coordinates = ['N/A', 'N/A']
    Endo_Center_x1_Label.config(text = Endo_Coordinates[0])
    Endo_Center_y1_Label.config(text = Endo_Coordinates[1])
    
    Epi_Coordinates = ['N/A', 'N/A']
    Epi_Center_x1_Label.config(text = Epi_Coordinates[0])
    Epi_Center_y1_Label.config(text = Epi_Coordinates[1])
    
    Next_Button["state"] = "disabled"
    if slc == matrix.shape[2] - 1:
        Next_Button.destroy()
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                       fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.1,
                            relwidth  = 0.1,
                            relx      = 0.45,
                            rely      = 0.875)
        Finish_Button["state"] = "disabled"
    dummy1.cla()
    tmp1 = dummy1.imshow(image1, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])
    dummy_scale1_idx = matrix.shape[0]/2
    dummy_scale2_idx = matrix.shape[1]/2
    dummy1.plot(y_center, x_center,
                marker = 'o', color = 'yellow', markersize = int(np.round(monitor_scale * 10)))
    dummy1.plot(epi_y, epi_x,
                color = 'cyan',    linewidth = line_width)
    dummy1.plot(endo_y, endo_x,
                color = 'magenta', linewidth = line_width)

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()
    
    dummy3.cla()
    tmp2 = dummy3.imshow(image2, cmap = cDTI_cmaps['MD'])
    tmp2.set_clim([0, 3])
    dummy_scale1_idx = matrix.shape[0]/2
    dummy_scale2_idx = matrix.shape[1]/2
    dummy3.plot(y_center, x_center,
                marker = 'o', color = 'k', markersize = int(np.round(monitor_scale * 10)))
    dummy3.plot(epi_y, epi_x,
                color = 'k', linewidth = line_width)
    dummy3.plot(endo_y, endo_x,
                color = 'k', linewidth = line_width)

    dummy3.set_aspect('equal')
    dummy3.axes.xaxis.set_visible(False)
    dummy3.axes.yaxis.set_visible(False)
    dummy3.spines.top.set_visible(False)
    dummy3.spines.left.set_visible(False)
    dummy3.spines.bottom.set_visible(False)
    dummy3.spines.right.set_visible(False)

    dummy_canvas2.draw()
    
    dummy5.cla()
    tmp3 = dummy5.imshow(image3)
    dummy_scale1_idx = matrix.shape[0]/2
    dummy_scale2_idx = matrix.shape[1]/2
    dummy5.plot(y_center, x_center,
                marker = 'o', color = 'k', markersize = int(np.round(monitor_scale * 10)))
    dummy5.plot(epi_y, epi_x,
                color = 'k', linewidth = line_width)
    dummy5.plot(endo_y, endo_x,
                color = 'k', linewidth = line_width)

    dummy5.set_aspect('equal')
    dummy5.axes.xaxis.set_visible(False)
    dummy5.axes.yaxis.set_visible(False)
    dummy5.spines.top.set_visible(False)
    dummy5.spines.left.set_visible(False)
    dummy5.spines.bottom.set_visible(False)
    dummy5.spines.right.set_visible(False)

    dummy_canvas3.draw()

def finish_program():
    root.destroy()

def update_plots(val):
    ### Set Figure Plots
    import numpy as np
    from   cDTIpy.Colormaps.Diffusion import cDTI_Colormaps_Generator

    global slc, dummy1, dummy_canvas1
    global dummy_scale1_idx, dummy_scale2_idx
    global min_clim, max_clim
    global y_scale_idx, x_scale_idx
    
    global Endo_Center, Endo_Center_x, Endo_Center_y
    global Endo_Centers, Epi_Centers, conditions
    global Epi_Center, Epi_Center_x, Epi_Center_y

    global Endo_Center_x1_Label, Endo_Center_y1_Label
    global conditions
    global Epi_Major_scale, Epi_Minor_scale
    global Endo_Major_scale, Endo_Minor_scale
    global y_center, x_center
    global epi_width, epi_height
    global endo_width, endo_height
    global image1, image2, image3
    global epi_x, epi_y, endo_x, endo_y
    global Epi_Center_y, Epi_Center_x
    global Endo_Center_y, Endo_Center_x
    global cDTI_cmaps


    
    y_center = y_scale.get()
    x_center = x_scale.get()
    
    y_center_epi  = Epi_Center_y
    x_center_epi  = Epi_Center_x
    y_center_endo = Endo_Center_y
    x_center_endo = Endo_Center_x
    
    epi_width   = Epi_Minor_scale.get()
    epi_height  = Epi_Major_scale.get()
    endo_width  = Endo_Minor_scale.get()
    endo_height = Endo_Major_scale.get()
    
    dummy_scale1_idx = int(x_center)
    dummy_scale2_idx = int(y_center)
    dummy1.cla()

    
    tmp1 = dummy1.imshow(image1, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])
    
    t_epi  = np.linspace(0, 2 * np.pi, 200)
    t_endo = np.linspace(0, 2 * np.pi, 200)

    epi_x  = x_center_epi  + epi_height * np.cos(t_epi)
    epi_y  = y_center_epi  + epi_width  * np.sin(t_epi)
    endo_x = x_center_endo + endo_height * np.cos(t_endo)
    endo_y = y_center_endo + endo_width  * np.sin(t_endo)
    
    dummy1.plot(y_center, x_center,
                marker = 'o', color = 'yellow', markersize = int(np.round(monitor_scale * 10)))
    dummy1.plot(epi_y, epi_x,
                color = 'cyan',    linewidth = line_width)
    dummy1.plot(endo_y, endo_x,
                color = 'magenta', linewidth = line_width)

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()
    
    dummy3.cla()
    tmp2 = dummy3.imshow(image2, cmap = cDTI_cmaps['MD'])
    tmp2.set_clim([0, 3])


    
    dummy3.plot(y_center, x_center,
                marker = 'o', color = 'k', markersize = int(np.round(monitor_scale * 10)))
    dummy3.plot(epi_y, epi_x,
                color = 'k', linewidth = line_width)
    dummy3.plot(endo_y, endo_x,
                color = 'k', linewidth = line_width)

    dummy3.set_aspect('equal')
    dummy3.axes.xaxis.set_visible(False)
    dummy3.axes.yaxis.set_visible(False)
    dummy3.spines.top.set_visible(False)
    dummy3.spines.left.set_visible(False)
    dummy3.spines.bottom.set_visible(False)
    dummy3.spines.right.set_visible(False)

    dummy_canvas2.draw()

    dummy5.cla()
    tmp3 = dummy5.imshow(image3)

    dummy5.plot(y_center, x_center,
                marker = 'o', color = 'k', markersize = int(np.round(monitor_scale * 10)))
    dummy5.plot(epi_y, epi_x,
                color = 'k', linewidth = line_width)
    dummy5.plot(endo_y, endo_x,
                color = 'k', linewidth = line_width)

    dummy5.set_aspect('equal')
    dummy5.axes.xaxis.set_visible(False)
    dummy5.axes.yaxis.set_visible(False)
    dummy5.spines.top.set_visible(False)
    dummy5.spines.left.set_visible(False)
    dummy5.spines.bottom.set_visible(False)
    dummy5.spines.right.set_visible(False)

    dummy_canvas3.draw()
    
def IntERCOMS_Mask_Making(matrix, Endo_Centers, Endo_Axes, Epi_Centers, Epi_Axes):
    import cv2
    import numpy as np
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    slcs = matrix.shape[2]
    
    myocardium_mask = np.zeros([rows, cols, slcs])
    blood_pool_mask = np.zeros([rows, cols, slcs])
    Myo_BP_mask     = np.zeros([rows, cols, slcs])
    
    for slc in range(slcs):
        image1             = np.zeros([rows, cols])
        center_coordinates = Epi_Centers[slc]
        axesLength         = Epi_Axes[slc]
        angle              = 0
        startAngle         = 0
        endAngle           = 360
        color              = (255, 0, 0)
        thickness          = -1
        epi_image = cv2.ellipse(image1,
                                (center_coordinates[1], center_coordinates[0]),
                                (axesLength[1], axesLength[0]),
                                angle, startAngle, endAngle, 1, thickness)
        image2             = np.zeros([rows, cols])
        center_coordinates = Endo_Centers[slc]
        axesLength         = Endo_Axes[slc]
        angle              = 0
        startAngle         = 0
        endAngle           = 360
        color              = (255, 0, 0)
        thickness          = -1
        endo_image = cv2.ellipse(image2,
                                 (center_coordinates[1], center_coordinates[0]),
                                 (axesLength[1], axesLength[0]),
                                 angle, startAngle, endAngle, 1, thickness)
        
        myocardium_mask[:, :, slc] = epi_image - endo_image
        blood_pool_mask[:, :, slc] = endo_image
        Myo_BP_mask[:, :, slc]     = epi_image
        del endo_image, epi_image
    return [myocardium_mask, Myo_BP_mask, blood_pool_mask]
