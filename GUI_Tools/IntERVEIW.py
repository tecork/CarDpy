def IntERVEIW_GUI(original_matrix):
    from   sys                               import platform
    import tkinter                           as tk
    import matplotlib.pyplot                 as plt
    import numpy                             as np
    from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from   RangeSlider.RangeSlider           import RangeSliderV
    from   screeninfo                        import get_monitors

    
    
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
    global Image_Label
    global monitor_scale


    matrix = original_matrix
    
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
    image = matrix[:, :, slc]
    min_clim = int(np.round(np.min(image)))
    max_clim = int(np.round(np.max(image)))
#     image               = np.max(matrix[:, :, slc, :], axis = 2)
#     image_normalization = image / image.max()
    conditions     = [0, 0, 0]
    Slice_Location = []
    Anterior_RVIP  = []
    Inferior_RVIP  = []

    for ii in range(matrix.shape[2]):
        Slice_Location.append([])
        Anterior_RVIP.append([])
        Inferior_RVIP.append([])
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

#    LARGEFONT  = ("Verdana", 35)
#    MEDIUMFONT = ("Verdana", 25)
#    SMALLFONT  = ("Verdana", 15)
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
    root.title('INTeractive Enhanced Right Ventricular Insertions Estimate Widget')

    Image_Label = tk.Label(root, font = MEDIUMFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Short Axis (SAX) View of Slice ' + str(slc + 1))
    Image_Label.place(relheight = 0.1,
                      relwidth  = 1.0,
                      relx      = 0.0,
                      rely      = 0.0)
    Step0_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step0_Label.config(text = 'DIRECTIONS')
    Step0_Label.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.125)
    #####
    Step1_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step1_Label.config(text = 'Step 1:')
    Step1_Label.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.150)
    Window_Level_Label = tk.Label(root, font = SMALLFONT,
                                  fg = '#000000', bg = '#ECECEC')
    Window_Level_Label.config(text = 'Window Level')
    Window_Level_Label.place(relheight = 0.025,
                     relwidth  = 0.1,
                     relx      = 0.875,
                     rely      = 0.175)
    Window_Min_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Window_Min_Label.config(text = 'Min')
    Window_Min_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.875,
                           rely      = 0.200)
    Window_Max_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Window_Max_Label.config(text = 'Max')
    Window_Max_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.925,
                           rely      = 0.200)
    
    WindowMinEntry = tk.Entry(root, font = SMALLFONT,
                              justify = tk.CENTER, fg = '#000000', bg = '#ECECEC')
    WindowMinEntry.insert(0, min_clim)
    WindowMinEntry.place(relheight = 0.025,
                          relwidth  = 0.05,
                          relx      = 0.875,
                          rely      = 0.225)
    
    WindowMaxEntry = tk.Entry(root, font = SMALLFONT,
                              justify = tk.CENTER, fg = '#000000', bg = '#ECECEC')
    WindowMaxEntry.insert(0, max_clim)
    WindowMaxEntry.place(relheight = 0.025,
                          relwidth  = 0.05,
                          relx      = 0.925,
                          rely      = 0.225)

    Window_Button = Button(root, text = "Set Window Level", font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col, command = set_window)
    Window_Button.place(relheight = 0.025,
                        relwidth  = 0.1,
                        relx      = 0.875,
                        rely      = 0.250)
    #####
    Step2_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step2_Label.config(text = 'Step 2:')
    Step2_Label.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.300)
    SAX1_Label = tk.Label(root, font = SMALLFONT,
                          fg = '#000000', bg = '#ECECEC')
    SAX1_Label.config(text = 'Short Axis Location')
    SAX1_Label.place(relheight = 0.026,
                     relwidth  = 0.1,
                     relx      = 0.875,
                     rely      = 0.325)
#     SAX2_Label = tk.Label(root, font = SMALLFONT,
#                           fg = '#000000', bg = '#ECECEC')
#     SAX2_Label.config(text = 'Slice Location')
#     SAX2_Label.place(relheight = 0.025,
#                      relwidth  = 0.1,
#                      relx      = 0.875,
#                      rely      = 0.350)
    BasalButton = tk.Radiobutton(root, font = SMALLFONT,
                                 variable = var, value = 1)
    BasalButton.config(text = 'Basal')
    BasalButton.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.351)
    MidButton = tk.Radiobutton(root, font = SMALLFONT,
                               variable = var, value = 2)
    MidButton.config(text = 'Mid Cavity')
    MidButton.place(relheight = 0.025,
                    relwidth  = 0.1,
                    relx      = 0.875,
                    rely      = 0.375)
    ApicalButton = tk.Radiobutton(root, font = SMALLFONT,
                                 variable = var, value = 3)
    ApicalButton.config(text = 'Apical')
    ApicalButton.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.400)
    ApexButton = tk.Radiobutton(root, font = SMALLFONT,
                                variable = var, value = 4)
    ApexButton.config(text = 'Apex')
    ApexButton.place(relheight = 0.025,
                     relwidth  = 0.1,
                     relx      = 0.875,
                     rely      = 0.425)
    SkipButton = tk.Radiobutton(root, font = SMALLFONT,
                                 variable = var, value = 5)
    SkipButton.config(text = 'Skip Slice')
    SkipButton.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.450)
    Location_Button = Button(root, text = "Select Location", font = SMALLFONT,
                             fg = app_txt_col, bg = app_bkg_col, command = select_Location)
    Location_Button.place(relheight = 0.025,
                          relwidth  = 0.1,
                          relx      = 0.875,
                          rely      = 0.475)
    
    #####
    Step3_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step3_Label.config(text = 'Step 3:')
    Step3_Label.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.525)
    Anterior1_Label = tk.Label(root, font = SMALLFONT,
                               fg = '#000000', bg = '#ECECEC')
    Anterior1_Label.config(text = 'Anterior RVIP')
    Anterior1_Label.place(relheight = 0.025,
                          relwidth  = 0.1,
                          relx      = 0.875,
                          rely      = 0.550)
    Anterior2_Label = tk.Label(root, font = SMALLFONT,
                               fg = '#000000', bg = '#ECECEC')
    Anterior2_Label.config(text = 'Coordinates')
    Anterior2_Label.place(relheight = 0.025,
                          relwidth  = 0.1,
                          relx      = 0.875,
                          rely      = 0.575)
    Anterior_x_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Anterior_x_Label.config(text = 'x')
    Anterior_x_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.875,
                           rely      = 0.600)
    Anterior_y_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Anterior_y_Label.config(text = 'y')
    Anterior_y_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.925,
                           rely      = 0.600)
    
    ARVIP_Coordinates = ['N/A', 'N/A']
    Anterior_x1_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Anterior_x1_Label.config(text = ARVIP_Coordinates[0])
    Anterior_x1_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.875,
                           rely      = 0.625)
    Anterior_y1_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Anterior_y1_Label.config(text = ARVIP_Coordinates[1])
    Anterior_y1_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.925,
                           rely      = 0.625)
    ARVIP_Button = Button(root, text = "Select Anterior", font = SMALLFONT,
                          fg = app_txt_col, bg = app_bkg_col, command = select_ARVIP)
    ARVIP_Button.place(relheight = 0.025,
                       relwidth  = 0.1,
                       relx      = 0.875,
                       rely      = 0.650)
    #####
    Step4_Label = tk.Label(root, font = SMALLFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Step4_Label.config(text = 'Step 4:')
    Step4_Label.place(relheight = 0.025,
                      relwidth  = 0.1,
                      relx      = 0.875,
                      rely      = 0.700)
    Inferior1_Label = tk.Label(root, font = SMALLFONT,
                               fg = '#000000', bg = '#ECECEC')
    Inferior1_Label.config(text = 'Inferior RVIP')
    Inferior1_Label.place(relheight = 0.025,
                          relwidth  = 0.1,
                          relx      = 0.875,
                          rely      = 0.725)
    Inferior2_Label = tk.Label(root, font = SMALLFONT,
                               fg = '#000000', bg = '#ECECEC')
    Inferior2_Label.config(text = 'Coordinates')
    Inferior2_Label.place(relheight = 0.025,
                          relwidth  = 0.1,
                          relx      = 0.875,
                          rely      = 0.750)
    Inferior_x_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Inferior_x_Label.config(text = 'x')
    Inferior_x_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.875,
                           rely      = 0.775)
    Inferior_y_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Inferior_y_Label.config(text = 'y')
    Inferior_y_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.925,
                           rely      = 0.775)
    IRVIP_Coordinates = ['N/A', 'N/A']
    Inferior_x1_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Inferior_x1_Label.config(text = IRVIP_Coordinates[0])
    Inferior_x1_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.875,
                           rely      = 0.800)
    Inferior_y1_Label = tk.Label(root, font = SMALLFONT,
                                fg = '#000000', bg = '#ECECEC')
    Inferior_y1_Label.config(text = IRVIP_Coordinates[1])
    Inferior_y1_Label.place(relheight = 0.025,
                           relwidth  = 0.05,
                           relx      = 0.925,
                           rely      = 0.800)
    IRVIP_Button = Button(root, text = "Select Inferior", font = SMALLFONT,
                          fg = app_txt_col, bg = app_bkg_col, command = select_IRVIP)
    IRVIP_Button.place(relheight = 0.025,
                       relwidth  = 0.1,
                       relx      = 0.875,
                       rely      = 0.825)
    #####
    x_scale = tk.Scale(root, from_ = 0, to = matrix.shape[0] - 1, font = SMALLFONT,
                      bg = frame_bkg_col2, fg = '#ffffff',
                      command = update_plots)
    x_scale.set(int(matrix.shape[0]/2))
    x_scale_idx   = x_scale.get()
    x_scale.place(relheight = 0.8,
                  relwidth  = 0.05,
                  relx      = 0.025,
                  rely      = 0.125)

    y_scale = tk.Scale(root, from_ = 0, to = matrix.shape[1] - 1, font = SMALLFONT,
                      bg = frame_bkg_col2, fg = '#ffffff', orient = tk.HORIZONTAL,
                      command = update_plots)
    y_scale.set(int(matrix.shape[1]/2))
    y_scale_idx   = y_scale.get()
    y_scale.place(relheight = 0.05,
                  relwidth  = 0.8,
                  relx      = 0.075,
                  rely      = 0.925)
    #####
    dummy_figure1 = plt.Figure(facecolor = '#ECECEC', tight_layout = True)
    dummy1        = dummy_figure1.add_subplot(111)
    dummy2        = dummy1.imshow(image, cmap = 'gray')
#     min_clim      = WindowMinEntry.get()
#     max_clim      = WindowMaxEntry.get()
    dummy2.set_clim([min_clim, max_clim])
    dummy1.set_aspect('equal')
    
    dummy_scale1_idx = matrix.shape[0]/2
    dummy_scale2_idx = matrix.shape[1]/2
    dummy1.plot(dummy_scale2_idx, dummy_scale1_idx,
                marker = 'o', color = 'red', markersize = int(np.round(monitor_scale * 10)))

    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)
    dummy_canvas1 = FigureCanvasTkAgg(dummy_figure1, root)
    dummy_canvas1.get_tk_widget().place(relheight = 0.8,
                                        relwidth  = 0.8,
                                        relx      = 0.075,
                                        rely      = 0.125)
        
    if slc == matrix.shape[2] - 1:
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                               fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.05,
                            relwidth  = 0.1,
                            relx      = 0.875,
                            rely      = 0.925)
        Finish_Button["state"] = "disabled"
    else:
        Next_Button = Button(root, text = "Next", font = SMALLFONT,
                             fg = app_txt_col, bg = app_bkg_col, command = next_slice)
        Next_Button.place(relheight = 0.05,
                          relwidth  = 0.1,
                          relx      = 0.875,
                          rely      = 0.925)
        Next_Button["state"] = "disabled"
    root.mainloop()
    return [Slice_Location, Anterior_RVIP, Inferior_RVIP]
def select_Location():
    import numpy as np
    global ARVIP_Coordinates, Anterior_RVIP
    global IRVIP_Coordinates, Inferior_RVIP
    global Anterior_x1_Label, Anterior_y1_Label
    global conditions, Slice_Location
    global var
    if var.get() == 1:
        slice_string = 'Basal'
#         print('Base')
    if var.get() == 2:
#         print('Mid')
        slice_string = 'Mid-Ventricular'
    if var.get() == 3:
#         print('Apex')
        slice_string = 'Apical'
    if var.get() == 4:
#         print('Apex')
        slice_string = 'Apex'
    if var.get() == 5:
#         print('Apex')
        slice_string = 'N/A'
    Slice_Location[slc] = slice_string
    if var.get() > 0:
        conditions[0] = 1
        if var.get() == 5:
            ARVIP_Coordinates = ['N/A', 'N/A']
            IRVIP_Coordinates = ['N/A', 'N/A']
            Anterior_RVIP[slc] = ARVIP_Coordinates
            Inferior_RVIP[slc] = IRVIP_Coordinates
            conditions = [1, 1, 1]
            print(conditions)
    if conditions == [1, 1, 1]:
        if slc == matrix.shape[2] - 1:
            Finish_Button["state"] = "normal"
        else:
            Next_Button["state"] = "normal"
    return slice_string

def set_window():
    global max_clim, min_clim
    min_clim = int(WindowMinEntry.get())
    max_clim = int(WindowMaxEntry.get())
    update_plots(var)
def select_ARVIP():
    import numpy as np
    global ARVIP_Coordinates, Anterior_x, Anterior_y
    global Anterior_x1_Label, Anterior_y1_Label
    global conditions
    import numpy as np
    global slc, dummy1, dummy_canvas1
    global dummy_scale1_idx, dummy_scale2_idx
    global min_clim, max_clim
    global y_scale_idx, x_scale_idx
    
    Anterior_x = int(np.round(x_scale_idx))
    Anterior_y = int(np.round(y_scale_idx))
    ARVIP_Coordinates = [Anterior_x, Anterior_y]
    Anterior_x1_Label.config(text = ARVIP_Coordinates[0])
    Anterior_y1_Label.config(text = ARVIP_Coordinates[1])
    conditions[1] = 1
    Anterior_RVIP[slc] = ARVIP_Coordinates
    if conditions == [1, 1, 1]:
        if slc == matrix.shape[2] - 1:
            Finish_Button["state"] = "normal"
        else:
            Next_Button["state"] = "normal"
#     print(ARVIP_Coordinates)
    y_scale_idx     = y_scale.get()
    x_scale_idx     = x_scale.get()
    
    dummy_scale1_idx = int(x_scale_idx)
    dummy_scale2_idx = int(y_scale_idx)
    dummy1.cla()
    
    tmp1 = dummy1.imshow(image, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])
    dummy1.plot(y_scale_idx, x_scale_idx,
                marker = '$A$', color = 'red', markersize = int(np.round(monitor_scale * 10)))

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)
    update_plots(var)
    return ARVIP_Coordinates
    
def select_IRVIP():
    import numpy as np
    global IRVIP_Coordinates, Inferior_x, Inferior_y
    global Inferior_x1_Label, Inferior_y1_Label
    global conditions
    import numpy as np
    global slc, dummy1, dummy_canvas1
    global dummy_scale1_idx, dummy_scale2_idx
    global min_clim, max_clim
    global y_scale_idx, x_scale_idx

    
    Inferior_x = int(np.round(x_scale_idx))
    Inferior_y = int(np.round(y_scale_idx))
    IRVIP_Coordinates = [Inferior_x, Inferior_y]
    Inferior_x1_Label.config(text = IRVIP_Coordinates[0])
    Inferior_y1_Label.config(text = IRVIP_Coordinates[1])
    conditions[2] = 1
    Inferior_RVIP[slc] = IRVIP_Coordinates
    if conditions == [1, 1, 1]:
        if slc == matrix.shape[2] - 1:
            Finish_Button["state"] = "normal"
        else:
            Next_Button["state"] = "normal"

#     print(IRVIP_Coordinates)
    y_scale_idx     = y_scale.get()
    x_scale_idx     = x_scale.get()
    
    dummy_scale1_idx = int(x_scale_idx)
    dummy_scale2_idx = int(y_scale_idx)
    dummy1.cla()
    
    tmp1 = dummy1.imshow(image, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])
    dummy1.plot(y_scale_idx, x_scale_idx,
                marker = '$I$', color = 'red', markersize = int(np.round(monitor_scale * 10)))

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)
    update_plots(var)
    return IRVIP_Coordinates
    
def next_slice():
    import tkinter as tk
    from sys import platform
    import numpy as np
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
    global min_clim, max_clim
    global dummy_scale1_idx, dummy_scale2_idx

    slc = slc + 1
#    image               = np.max(matrix[:, :, slc], axis = 2)
    image               = matrix[:, :, slc]
#    image_normalization = image / image.max()
#    min_clim = int(np.round(np.min(image)))
#    max_clim = int(np.round(np.max(image)))
#    Image_Label.destroy()
#    Image_Label = tk.Label(root, font = MEDIUMFONT,
#                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Short Axis (SAX) View of Slice ' + str(slc + 1))
    conditions = [0, 0, 0]
    ARVIP_Coordinates = ['N/A', 'N/A']
    Anterior_x1_Label.config(text = ARVIP_Coordinates[0])
    Anterior_y1_Label.config(text = ARVIP_Coordinates[1])
    
    IRVIP_Coordinates = ['N/A', 'N/A']
    Inferior_x1_Label.config(text = IRVIP_Coordinates[0])
    Inferior_y1_Label.config(text = IRVIP_Coordinates[1])
    
    Next_Button["state"] = "disabled"
    if slc == matrix.shape[2] - 1:
        Next_Button.destroy()
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                       fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.05,
                            relwidth  = 0.1,
                            relx      = 0.875,
                            rely      = 0.925)
        Finish_Button["state"] = "disabled"
    dummy1.cla()
    tmp1 = dummy1.imshow(image, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])
    dummy1.plot(dummy_scale2_idx, dummy_scale1_idx,
                marker = 'o', color = 'red', markersize = int(np.round(monitor_scale * 10)))

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()

def finish_program():
    root.destroy()

def update_plots(val):
    ### Set Figure Plots
    import numpy as np
    global slc, dummy1, dummy_canvas1
    global dummy_scale1_idx, dummy_scale2_idx
    global min_clim, max_clim
    global y_scale_idx, x_scale_idx
    y_scale_idx     = y_scale.get()
    x_scale_idx     = x_scale.get()
    
    dummy_scale1_idx = int(x_scale_idx)
    dummy_scale2_idx = int(y_scale_idx)
    dummy1.cla()
    
    tmp1 = dummy1.imshow(image, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])
    dummy1.plot(y_scale_idx, x_scale_idx,
                marker = 'o', color = 'red', markersize = int(np.round(monitor_scale * 10)))
    if conditions[1] == 1:
            dummy1.plot(Anterior_y, Anterior_x,
                        marker = '$A$', color = 'red',
                        markersize = int(np.round(monitor_scale * 50)))
    if conditions[2] == 1:
            dummy1.plot(Inferior_y, Inferior_x,
                        marker = '$I$', color = 'red',
                        markersize = int(np.round(monitor_scale * 50)))

        
    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()
