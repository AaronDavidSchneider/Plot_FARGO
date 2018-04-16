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

import numpy as np
import os
from multiprocessing import Pool


class Plot_FARGO:
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

    def __init__(self, path_to_fargo, setup, setup_dir = None):
        self.setup = setup

        # read in the setup.par file:
        if setup_dir is None: setup_dir = "setups/"+setup
        vals = np.genfromtxt(path_to_fargo + "/" + setup_dir + "/" + setup + ".par", usecols=(1), dtype=str)
        names = np.genfromtxt(path_to_fargo + "/" + setup_dir + "/" + setup + ".par", usecols=(0), dtype=str)
        self.parameters = dict(zip(names, vals))
        if self.parameters['OutputDir'][0] == "@":
            self.parameters['OutputDir'] = self.parameters['OutputDir'][1:]


        #additional initializations:
        self.cm_min, self.cm_max = None, None
        self.path_to_fargo = path_to_fargo

        # set the output directory used in fargo:
        if os.getenv('FARGO_OUT') is None:
            self.output_dir = path_to_fargo + "/" + self.parameters['OutputDir'] + "/"
        elif os.getenv('FARGO_OUT')[0] == "/":
            self.output_dir = os.getenv('FARGO_OUT') + "/" + self.parameters['OutputDir'] + "/"
        else:
            self.output_dir = path_to_fargo + "/" + os.getenv('FARGO_OUT') + "/" + self.parameters['OutputDir'] + "/"

        # Import the domains (needed for scaling reasons):
        self.domain = np.array([np.genfromtxt(self.output_dir + "domain_x.dat"),
                                np.genfromtxt(self.output_dir + "domain_y.dat"),
                                np.genfromtxt(self.output_dir + "domain_z.dat")])

    ###############################
    # set_clim
    ###############################
    # analogue to matplotlib.pyplot.set_clim, mainly needed for plot_2D_video
    #
    # arguments:
    # - (number) cm_min:         minimum value of the colormap
    # - (number) cm_max:         maximum value of the colormap


    def set_clim(self, cm_min, cm_max):
        self.cm_min, self.cm_max = cm_min, cm_max

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


    def plot_2D(self, output_number, direct = None, ax = 0, tp = "gasdens", filename="", log10 = True, polar = True):
        import matplotlib.pyplot as plt

        n_array = np.array([int(self.parameters.get("Nx",1)),
                            int(self.parameters.get("Ny",1)),
                            int(self.parameters.get("Nz",1))])

        data = np.fromfile(self.output_dir + tp + str(output_number) + ".dat")\
            .reshape(n_array[2], n_array[1], n_array[0])
        data = np.squeeze(data)

        if direct is None:
            ind_to_ax = dict(zip([0, 1, 2], ['x', 'y', 'z']))
            index_not_1 = np.where(n_array != 1)[0]
            direct = [ind_to_ax[index_not_1[0]], ind_to_ax[index_not_1[1]]]
            direct.sort()

        polar_temp = polar

        if data.ndim == 3:
            if direct == ['x','y']:
                data = data[ax,3:-3,:]
                domain = self.domain[1:]
                domain[1] = domain[1][3:-3]
            elif direct == ['y','z']:
                data = data[3:-3,3:-3,ax]
                domain = self.domain[:-1]
                domain[0] = domain[0][3:-3]
                domain[1] = domain[1][3:-3]
                polar_temp = False  # nescessary for cylindrical setup
            elif direct == ['x','z']:
                data = data[3:-3,ax,:]
                domain = self.domain[::2]
                domain[1] = domain[1][3:-3]
        else:
            domain = self.domain[np.where(n_array != 1)]
            if direct == ['x','y']:
                #data = data[3:-3,:]
                domain[1] = domain[1][3:-3]
            elif direct == ['y','z']:
                #data = data[3:-3,3:-3]
                domain[0] = domain[0][3:-3]
                domain[1] = domain[1][3:-3]
                polar_temp = False # nescessary for cylindrical setup
            elif direct == ['x','z']:
                #data = data[3:-3,:]
                domain[1] = domain[1][3:-3]

        if data.ndim == 3 and direct is None: print("3 Dim - spezialisiere deine Dimension!")

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=polar_temp)

        if log10:
            cax = ax.pcolormesh(domain[0], domain[1], np.log10(data))
        else:
            cax = ax.pcolormesh(domain[0], domain[1], data)


        fig.colorbar(cax)
        if self.cm_max or self.cm_min is not None:
            cax.set_clim(self.cm_min, self.cm_max)

        ax.set_title("Plot of " + self.setup + " - " + tp + "\n")

        if polar: # assumes cylindrical setup!
            dir_to_label = dict(zip(['x','y','z'],["Azimuth - $\phi$ / $^\circ$", "radius", "Z"]))
            ax.set_xlabel(dir_to_label[direct[0]] + "\n")
        else:
            dir_to_label = dict(zip(['x', 'y', 'z'], ["X", "Y", "Z"]))
            ax.set_xlabel(dir_to_label[direct[0]] + "\n")
            ax.set_ylabel(dir_to_label[direct[1]] + "\n")

        if filename != "":
            plt.savefig(filename, dpi = 300)
            plt.close()
        else:
            return fig, ax

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

    def plot_2D_video(self, filename, direct = None, tp = "gasdens", log10 = True, polar = True,
                      framesteps = 1, N = None):
        if N is None:
            N = int(int(self.parameters["Ntot"])/int(self.parameters.get("Ninterm", 1)))

        def plot_and_save(i):
            self.plot_2D(i, direct = direct, tp=tp, log10 = log10, polar = polar,
                         filename = "single_frames/"+filename+"{:05d}".format(i)+".jpeg")
            if i > 10 and i % round(N / 10) == 0: print(round(i / N * 100), "%")

        for i in range(0, N+1, framesteps):
            plot_and_save(i)

        cmd_string = "ffmpeg -framerate 24 -i single_frames/"\
                     +filename+"%05d.jpeg -r 24 -vcodec libx264 -crf 18 -pix_fmt yuv420p videos/"+filename+".mp4"
        del_string = "rm -rf single_frames/"+filename+"*.jpeg"

        os.system(cmd_string)
        os.system(del_string)