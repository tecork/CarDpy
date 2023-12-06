def INTERACT_GUI(original_matrix, organ_of_intrest):
    from   sys                               import platform
    import tkinter                           as tk
    import matplotlib.pyplot                 as plt
    import numpy                             as np
    from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from   RangeSlider.RangeSlider           import RangeSliderH, RangeSliderV
    
    global hVar1, hVar2, hVar3, hVar4, hVar5, hVar6
    global slc, image_normalization, Next_Button, Finish_Button, Image_Label
    global dummy_scale1_idx, dummy_scale2_idx, dummy_scale3_idx, dummy_scale4_idx
    global min_clim, max_clim, matrix, organ
    global dummy1, dummy_canvas1
    global x_start, x_end, y_start, y_end
    global root, MEDIUMFONT, app_txt_col, app_bkg_col
    global SMALLFONT
    
    matrix = original_matrix
    organ  = organ_of_intrest
    
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
    image               = np.max(matrix[:, :, slc, :], axis = 2)
    image_normalization = image / image.max()
    x_start = []
    x_end   = []
    y_start = []
    y_end   = []

    for ii in range(matrix.shape[2]):
        x_start.append([])
        x_end.append([])
        y_start.append([])
        y_end.append([])
    # root window
    root = tk.Tk()

    if platform == 'darwin':
        from tkmacosx import Button
    else:
        from tkinter import Button
    LARGEFONT  = ("Verdana", 35)
    MEDIUMFONT = ("Verdana", 25)
    SMALLFONT  = ("Verdana", 15)
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
    root.title('INTeractive Enhanced Rectangular Area Cropping Tool')

    Image_Label = tk.Label(root, font = MEDIUMFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Normalized Maximum Intensity Projection (MIP) for Slice ' + str(slc + 1) +
                              '\nOrgan to Crop: '+ str(organ))
    Image_Label.place(relheight = 0.1,
                      relwidth  = 1.0,
                      relx      = 0.0,
                      rely      = 0.0)

    dummy_figure1 = plt.Figure(facecolor = '#ECECEC', tight_layout = True)
    dummy1        = dummy_figure1.add_subplot(111)
    dummy2        = dummy1.imshow(image_normalization, cmap = 'gray')
    min_clim      = 0
    max_clim      = 1
    dummy2.set_clim([min_clim, max_clim])
    dummy1.set_aspect('equal')
    
    dummy_scale1_idx = 0
    dummy_scale2_idx = matrix.shape[1] - 1
    dummy_scale3_idx = 0
    dummy_scale4_idx = matrix.shape[0] - 1

    dummy1.axvline(x = dummy_scale1_idx, color = '#61ba86', linewidth = 10)
    dummy1.axvline(x = dummy_scale2_idx, color = '#61ba86', linewidth = 10)
    dummy1.axhline(y = dummy_scale3_idx, color = '#be86e3', linewidth = 10)
    dummy1.axhline(y = dummy_scale4_idx, color = '#be86e3', linewidth = 10)

    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)
    dummy_canvas1 = FigureCanvasTkAgg(dummy_figure1, root)
    dummy_canvas1.get_tk_widget().place(relheight = 0.8,
                                        relwidth  = 0.8,
                                        relx      = 0.1,
                                        rely      = 0.1)

    hVar1 = tk.DoubleVar()  # left handle variable
    hVar2 = tk.DoubleVar()  # right handle variable
    hVar3 = tk.DoubleVar()  # left handle variable
    hVar4 = tk.DoubleVar()  # right handle variable
    hVar5 = tk.DoubleVar()  # left handle variable
    hVar6 = tk.DoubleVar()  # right handle variable
    pad_x  = 15
    if matrix_type == 'SQUARE':
        slider_width  = int(window_width / 10) * 8
        slider_height = int(window_height / 10 )
    if matrix_type == 'TALL':
        slider_width  = int(window_width / 10) * 8 * matrix_ratio
        slider_height = int(window_height / 10 )
    if matrix_type == 'LONG':
        slider_width  = int(window_width / 10) * 8
        slider_height = int(window_height / 10)
    if slider_width < 75:
        slider_width = 75
    if slider_height < 75:
        slider_height = 75

    

    
    H_min_val     = 0
    H_max_val     = matrix.shape[1] - 1
    H_line_width  = 5
    H_bar_radius  = 10
    H_font_size   = 15
    H_font_family = 'Verdana'
    
    H_padX   = max(max(len(str(H_min_val)), len(str(H_max_val))) * H_font_size * 1.33 / 4, H_bar_radius)
    H_width  = 2 * (H_padX + H_bar_radius)
    H_height = 2 * (1.33 * H_font_size + H_bar_radius)
    rs1 = RangeSliderH(root,
                       [hVar1, hVar2],
                       Width           = slider_width,
                       Height          = H_height,
                       min_val         = H_min_val,
                       max_val         = H_max_val,
                       padX            = H_padX,
                       line_width      = H_line_width,
                       bar_radius      = H_bar_radius,
                       font_size       = H_font_size,
                       font_family     = H_font_family,
                       bar_color_inner = '#FFFFFF',
                       bar_color_outer = '#61ba86',
                       line_s_color    = '#000000',
                       line_color      = '#808080',
                       bgColor         = '#ECECEC',
                       show_value      = False,
                       digit_precision = '.0f')
    rs1.place(relheight = 0.1,
              relwidth  = 0.8,
              relx      = 0.1,
              rely      = 0.9)

    if matrix_type == 'SQUARE':
        slider_width  = int(window_width / 10)
        slider_height = int(window_height / 10 ) * 8
    if matrix_type == 'TALL':
        slider_width  = int(window_width / 10)
        slider_height = int(window_height / 10 ) * 8
    if matrix_type == 'LONG':
        slider_width  = int(window_width / 10)
        slider_height = int(window_height / 10) * 8
    # width  = int(window_width / 10) # - pad_x
    if slider_width < 75:
        slider_width = 75
    # height = int(window_height / 10 ) * 8
    if slider_height < 75:
        slider_height = 75
    
    
    V_min_val     = 0
    V_max_val     = matrix.shape[0] - 1
    V_line_width  = 5
    V_bar_radius  = 10
    V_font_size   = 15
    V_font_family = 'Verdana'
    
    V_padY   = max(V_bar_radius, V_font_size * 1.33 / 2)
    V_width  = 2 * (V_bar_radius + max(len(str(V_min_val)), len(str(V_max_val))) * V_font_size / 1.2)
    V_height = 2 * (V_padY + V_bar_radius)
    rs2 = RangeSliderV(root,
                       [hVar3, hVar4],
                       Width           = V_width,
                       Height          = slider_height,
                       min_val         = V_min_val,
                       max_val         = V_max_val,
                       padY            = V_padY,
                       line_width      = V_line_width,
                       bar_radius      = V_bar_radius,
                       font_size       = V_font_size,
                       font_family     = V_font_family,
                       bar_color_inner = '#FFFFFF',
                       bar_color_outer = '#61ba86',
                       line_s_color    = '#000000',
                       line_color      = '#808080',
                       bgColor         = '#ECECEC',
                       show_value      = False,
                       digit_precision = '.0f')
    
    
    
#    rs2 = RangeSliderV(root,
#                       [hVar3, hVar4],
#                       Width           = slider_width,
#                       Height          = slider_height,
#                       min_val         = 0,
#                       max_val         = matrix.shape[0] - 1,
#                       padY            = pad_x,
#                       line_width      = 5,
#                       bar_radius      = 10,
#                       bar_color_inner = '#FFFFFF',
#                       bar_color_outer = '#be86e3',
#                       line_s_color    = '#000000',
#                       line_color      = '#808080',
#                       bgColor         = '#ECECEC',
#                       show_value      = False,
#                       digit_precision = '.0f')
    rs2.place(relheight = 0.8,
              relwidth  = 0.1,
              relx      = 0.0,
              rely      = 0.1)

    Limit_Label = tk.Label(root, font = SMALLFONT,
                           fg = '#000000', bg = '#ECECEC')
    Limit_Label.config(text = 'Window \nLimit')
    Limit_Label.place(relheight = 0.1,
                      relwidth  = 0.1,
                      relx      = 0.9,
                      rely      = 0.1)
    if matrix_type == 'SQUARE':
        slider_width  = int(window_width / 10)
        slider_height = int(window_height / 10 ) * 7
    if matrix_type == 'TALL':
        slider_width  = int(window_width / 10)
        slider_height = int(window_height / 10 ) * 7
    if matrix_type == 'LONG':
        slider_width  = int(window_width / 10)
        slider_height = int(window_height / 10) * 7
        
    V_min_val_2   = 0
    V_max_val_2   = 1
    V_line_width  = 5
    V_bar_radius  = 10
    V_font_size   = 15
    V_font_family = 'Verdana'
    
    V_padY   = max(V_bar_radius, V_font_size * 1.33 / 2)
    V_width  = 2 * (V_bar_radius + max(len(str(V_min_val_2)), len(str(V_max_val_2))) * V_font_size / 1.2)
    V_height = 2 * (V_padY + V_bar_radius)
    
    rs3 = RangeSliderV(root,
                       [hVar5, hVar6],
                       Width           = V_width,
                       Height          = slider_height,
                       min_val         = 0,
                       max_val         = 1,
                       padY            = V_padY,
                       line_width      = V_line_width,
                       bar_radius      = V_bar_radius,
                       font_size       = V_font_size,
                       font_family     = V_font_family,
                       bar_color_inner = '#FFFFFF',
                       bar_color_outer = '#61ba86',
                       line_s_color    = '#000000',
                       line_color      = '#808080',
                       bgColor         = '#ECECEC',
                       show_value      = False,
                       digit_precision = '.0f')
#    rs3 = RangeSliderV(root,
#                       [hVar5, hVar6],
#                       Width           = slider_width,
#                       Height          = slider_height,
#                       min_val         = 0,
#                       max_val         = 1,
#                       padY            = pad_x,
#                       line_width      = 5,
#                       bar_radius      = 10,
#                       bar_color_inner = '#FFFFFF',
#                       bar_color_outer = '#000000',
#                       line_s_color    = '#000000',
#                       line_color      = '#808080',
#                       bgColor         = '#ECECEC',
#                       show_value      = True,
#                       valueSide       = 'RIGHT',
#                       digit_precision = '.2f',
#                       font_family     = "Verdana",
#                       font_size       = 15)
    rs3.place(relheight = 0.7,
              relwidth  = 0.1,
              relx      = 0.9,
              rely      = 0.2)
    hVar1.trace_add('write', update_plots)
    hVar3.trace_add('write', update_plots)
    hVar5.trace_add('write', update_plots)

    Crop_Button = Button(root, text = "Crop", font = SMALLFONT,
                            fg = app_txt_col, bg = app_bkg_col, command = execute_crop)
    Crop_Button.place(relheight = 0.1,
                      relwidth  = 0.1,
                      relx      = 0.0,
                      rely      = 0.9)
    if slc == matrix.shape[2] - 1:
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                               fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.1,
                            relwidth  = 0.1,
                            relx      = 0.9,
                            rely      = 0.9)
        Finish_Button["state"] = "disabled"
    else:
        Next_Button = Button(root, text = "Next", font = SMALLFONT,
                             fg = app_txt_col, bg = app_bkg_col, command = next_slice)
        Next_Button.place(relheight = 0.1,
                          relwidth  = 0.1,
                          relx      = 0.9,
                          rely      = 0.9)
        Next_Button["state"] = "disabled"
    root.mainloop()
    return [x_start, x_end, y_start, y_end]

def execute_crop():
    import numpy as np
    global x_start, x_end, y_start, y_end, Next_Button, Finish_Button
    y_start[slc] = image_normalization.shape[0] - 1 - int(np.round(hVar4.get()))
#    print(hVar4.get())
#    print(x_start)
    y_end[slc]   = image_normalization.shape[0] - 1 - int(np.round(hVar3.get()))
#    print(hVar3.get())
#    print(x_end)
    x_start[slc] = int(np.round(hVar1.get()))
#    print(hVar1.get())
#    print(y_start)
    x_end[slc]   = int(np.round(hVar2.get()))
#    print(hVar2.get())
#    print(y_end)
    #Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]
    if slc == matrix.shape[2] - 1:
        Finish_Button["state"] = "normal"
    else:
        Next_Button["state"] = "normal"
    return x_start, x_end, y_start, y_end

def next_slice():
    import tkinter as tk
    from sys import platform
    import numpy as np
    if platform == 'darwin':
        from tkmacosx import Button
    else:
        from tkinter import Button
    global slc, organ, image_normalization, Next_Button, Finish_Button, Image_Label
    global dummy_scale1_idx, dummy_scale2_idx, dummy_scale3_idx, dummy_scale4_idx
    global dummy1

    slc = slc + 1
    image               = np.max(matrix[:, :, slc, :], axis = 2)
    image_normalization = image / image.max()
    Image_Label.destroy()
    Image_Label = tk.Label(root, font = MEDIUMFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Normalized Maximum Intensity Projection (MIP) for Slice ' + str(slc + 1) +
                              '\nOrgan to Crop: '+ str(organ))
    Image_Label.place(relheight = 0.1,
                      relwidth  = 1.0,
                      relx      = 0.0,
                      rely      = 0.0)
    Next_Button["state"] = "disabled"
    if slc == matrix.shape[2] - 1:
        Next_Button.destroy()
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                       fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.1,
                            relwidth  = 0.1,
                            relx      = 0.9,
                            rely      = 0.9)
        Finish_Button["state"] = "disabled"
    dummy1.cla()
    tmp1 = dummy1.imshow(image_normalization, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])

    dummy1.axvline(x = dummy_scale1_idx, color = '#61ba86', linewidth = 10)
    dummy1.axvline(x = dummy_scale2_idx, color = '#61ba86', linewidth = 10)
    dummy1.axhline(y = dummy_scale3_idx, color = '#be86e3', linewidth = 10)
    dummy1.axhline(y = dummy_scale4_idx, color = '#be86e3', linewidth = 10)

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()
    hVar1.trace_add('write', update_plots)
    hVar3.trace_add('write', update_plots)
    hVar5.trace_add('write', update_plots)

def finish_program():
    root.destroy()

def update_plots(var, index, mode):
    ### Set Figure Plots
    import numpy as np
    global slc, dummy1, dummy_canvas1
    global dummy_scale1_idx, dummy_scale2_idx, dummy_scale3_idx, dummy_scale4_idx
    global min_clim, max_clim
    dummy_scale1_idx = int(np.round(hVar1.get()))
    dummy_scale2_idx = int(np.round(hVar2.get()))
    dummy_scale3_idx = matrix.shape[0] - 1 - int(np.round(hVar3.get()))
    dummy_scale4_idx = matrix.shape[0] - 1 - int(np.round(hVar4.get()))
    min_clim         = hVar5.get()
    max_clim         = hVar6.get()
    dummy1.cla()
    tmp1 = dummy1.imshow(image_normalization, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])

    dummy1.axvline(x = dummy_scale1_idx, color = '#61ba86', linewidth = 10)
    dummy1.axvline(x = dummy_scale2_idx, color = '#61ba86', linewidth = 10)
    dummy1.axhline(y = dummy_scale3_idx, color = '#be86e3', linewidth = 10)
    dummy1.axhline(y = dummy_scale4_idx, color = '#be86e3', linewidth = 10)

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()
