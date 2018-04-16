#------------------------------------------------------------------------------------------
#                                      PLOT_FARGO
#                A PYTHON PLOTTING CLASS TO VISUALIZE THE OUTPUT OF FARGO3D
#                                     A. D. SCHNEIDER
#                                         2018
#
# The class Plot_FARGO can be used for the visualisation of FARGO3D Output
#
# Capabilities:
# - simplified 2D ploting
# - creating of 2D plots using ffmpeg
#
# Design philosophy:
# The complexity and the possibilitys provided by Matplotlib is great. This code tries to shorten unnescessary things
# without loosing the possibilities of Matplotlib. Therefore after apllying the routines one can still access
# fig, ax to continue individual ploting.
#
# Dependancies:
# - Python 3
# - matplotlib, numpy
# - ffmpeg (if plot_2D_video is used)
# - the multiprocessing module of python might be used in future
#
# There is no guarante that these routines work in every case.
#
#
#
# Todo: parallelise plot_2D_video using multiprocessing
# Todo: make 3D Plots (maybe in future)
# Todo: spherical Plots (not nescescary for my work)

    ###############################
    # __init__
    ###############################
    # reads in the setup.par file to initialise the plotting
    #
    # arguments:
    # - (string) path_to_fargo:           absolute directory of fargo (without "/" at end)
    # - (string) setup:                   name of setup that is plotted.
    #                                     Needs to be identical (case-sensitive) to the setupname used in FARGO3D
    # - (string) setup_dir (optional):    if a different location for the setup files is used, take this directory
    #                                     instead. Nevertheless the directory still needs to be a subfolder of the
    #                                     fargo directory

    ###############################
    # set_clim
    ###############################
    # analogue to matplotlib.pyplot.set_clim, mainly needed for plot_2D_video
    #
    # arguments:
    # - (number) cm_min:         minimum value of the colormap
    # - (number) cm_max:         maximum value of the colormap

    ###############################
    # plot_2D
    ###############################
    # main Method that takes the output number and output type (gasdens, etc) and creates a 2D plot
    # plot_2D only works with cubic or cylindrical coordinates
    #
    # returns (if filename = ""):
    # - fig: returned figure
    # - ax:  returnes axes
    #
    # arguments:
    # - (number)        output_number:         number of the output being ploted
    # - (list of chars) direct (optional):     only needed in 3D! gives the direction in which the plot is done
    # - (number)        ax (optional):         only needed in 3D! gives the indicee in perpendicular directionat which
    #                                          the profile is ploted
    # - (string)        tp (optional):         tp specifys the variable that is ploted. by default this is the density
    #                                          important: tp needs to be the exact same as the relating filename
    # - (string)        filename (optional):   specifies the filename for saving the plot.
    # - (Bool)          log10 (optional):      set log10 = False to get a linear Plot
    # - (Bool)          polar (optional):      if true the returned figure is ploted in cylindrical coordinates

    ###############################
    # plot_2D_video (needs ffmpeg)
    ###############################
    # Method that creates a video of the output files using ffmpeg
    # Please note: this routine needs the subfolders "single_frames" and "videos"
    # Warning: needs time, CPU power und storage (for temporary pictures and video)
    #
    # arguments (see also plot_2D):
    # - (string) filename:              the filename of the created video (and its temporary files)
    # - (number) framesteps (optional): can be set >1 if one doesn't want to plot every picture
    # - (number) N (optional):          can be set, if the total number of outputs isn't the same as in setup.par
