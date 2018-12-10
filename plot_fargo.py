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

import numpy as np
import os
from multiprocessing import Pool
import natconst as n

au = n.au
ms = n.MS
T0 = 20512

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
    # - (bool) FARGOCA                    Some parameters in this library have been adapted for the specific use with FARGOCA.

    def __init__(self, path_to_fargo, setup, setup_dir = None, parname=None, FARGOCA=False,):
        self.setup = setup

        # read in the setup.par file:
        if setup_dir is None: setup_dir = "setups/"+setup
        if parname is None: parname = setup
        vals = np.genfromtxt(path_to_fargo + "/" + setup_dir + "/" + parname + ".par", usecols=(1), dtype=str)
        names = np.genfromtxt(path_to_fargo + "/" + setup_dir + "/" + parname + ".par", usecols=(0), dtype=str)
        for i in range(len(names)):
            names[i]= names[i].upper()

        self.parameters = dict(zip(names, vals))
        if self.parameters['OUTPUTDIR'][0] == "@":

            self.parameters['OUTPUTDIR'] = self.parameters['OUTPUTDIR'][1:]


        #additional initializations:
        self.cm_min, self.cm_max = None, None
        self.path_to_fargo = path_to_fargo
        self.FARGOCA = FARGOCA

        # set the output directory used in fargo:
        if os.getenv('FARGO_OUT') is None:
            self.output_dir = path_to_fargo + "/" + self.parameters['OUTPUTDIR'] + "/"
        elif os.getenv('FARGO_OUT')[0] == "/":
            self.output_dir = os.getenv('FARGO_OUT') + "/" + self.parameters['OUTPUTDIR'] + "/"
        else:
            self.output_dir = path_to_fargo + "/" + os.getenv('FARGO_OUT') + "/" + self.parameters['OUTPUTDIR'] + "/"

        # Import the domains (needed for scaling reasons):
        if self.FARGOCA: #mod fargo
            print(self.parameters['RMAX'])
            self.domain = np.array([
                np.array([0]),
                np.genfromtxt(self.output_dir + "used_rad.dat"),
                np.genfromtxt(self.output_dir + "used_phi.dat", usecols=0)
            ])
        else: #default fargo
            self.domain = np.array([np.genfromtxt(self.output_dir + "domain_x.dat"),
                                np.genfromtxt(self.output_dir + "domain_y.dat"),
                                np.genfromtxt(self.output_dir + "domain_z.dat")])
        self.units = None
        self.xlim = None
        self.ylim = None

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
    # set_units
    ###############################
    # can be used to convert units
    # right now: only converts cm to AU
    #
    # arguments:
    # - (string) units:         minimum value of the colormap
    #
    # possible units:
    # - "AU":   converts Y from cm to AU
    def set_units(self,units): #converts cm to AU
        self.units = units
        if units == "AU":
            self.domain[1] = self.domain[1]/1.496e+13
            self.domain[2] = self.domain[2]/1.496e+13
        if units == "AUsph":
            self.domain[2] = np.ones(len(self.domain[2]))*np.pi/2-self.domain[2]
            if self.FARGOCA:
                self.domain[1] = self.domain[1]*5.2
            else:
                self.domain[1] = self.domain[1]/1.496e+13


    ###############################
    # set_xlim
    ###############################
    # can be used to fix the x_axis
    # right now: only used in plot_1D and plot_1D_video()
    # works like ax.set_xlim
    #
    # arguments:
    # - (list) xlim:         [x_min, x_max]
    def set_xlim(self, xlim):
        self.xlim = xlim

    ###############################
    # set_ylim
    ###############################
    # can be used to fix the y_axis
    # right now: only used in plot_1D and plot_1D_video()
    # works like ax.set_ylim
    #
    # arguments:
    # - (list) ylim:         [y_min, y_max]
    def set_ylim(self, ylim):
        self.ylim = ylim


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



    def plot_2D(self, output_number, direct = None, axis = 0, tp = "gasdens", filename="", xlog10 = True, ylog10 = False, datalog10=False, rlog10=False, polar = True, sph=False, scale="scalefree", animated=False, fig=None, ax=None, settitle = True, N_start=1):
        import matplotlib.pyplot as plt

        if self.FARGOCA:
            n_array = np.array([
                            1,
                            int(self.parameters.get("NRAD",1)),
                            int(self.parameters.get("NINC",1))])


            data = np.fromfile(self.output_dir + tp + str(output_number) + ".dat").reshape(n_array[1], n_array[2], int(self.parameters.get("NSEC",1)))
            data = data[:,:,0] #drop azimuthal cells
            data = data.T

        else:
            n_array = np.array([int(self.parameters.get("NX",1)),
                            int(self.parameters.get("NY",1)),
                            int(self.parameters.get("NZ",1))])

            data = np.fromfile(self.output_dir + tp + str(output_number) + ".dat").reshape(n_array[2], n_array[1], n_array[0])

        data = np.squeeze(data)
        if self.FARGOCA and tp=="gasdens":
            data = data*ms/(5.2*au)**3
        elif self.FARGOCA and tp=="gastemper":
            data = data*T0

        self.direct = direct
        if self.direct is None:
            ind_to_ax = dict(zip([0, 1, 2], ['x', 'y', 'z']))
            index_not_1 = np.where(n_array != 1)[0]
            self.direct = [ind_to_ax[index_not_1[0]], ind_to_ax[index_not_1[1]]]
            self.direct.sort()

        polar_temp = polar

        if data.ndim == 3:
            if self.direct == ['y','z']:
                data = data[:,:,axis]
                domain = self.domain[1:]

                if !self.FARGOCA:
                    domain[0] = domain[0][3:-3]
                    domain[1] = domain[1][3:-3]
                data = np.squeeze(data)

            elif self.direct == ['x','y']:
                data = data[axis,:,:]
                domain = self.domain[:-1]
                if !self.FARGOCA:
                    domain[1] = domain[1][3:-3] #nescescary for sph
                polar_temp = False  # nescessary for cylindrical setup
                data = np.squeeze(data)

            elif self.direct == ['x','z']:
                data = data[:,axis,:]
                domain = self.domain[::2]
                if !self.FARGOCA:
                    domain[0] = domain[0][3:-3]
                data = np.squeeze(data)

        elif data.ndim == 1:
            print("please use plot_1D() ")
        else:
            domain = self.domain[np.where(n_array != 1)]
            if self.direct == ['x','y']:
                if !self.FARGOCA:
                    domain[1] = domain[1][3:-3]
                if not sph:
                    polar_temp = False # nescessary for cylindrical setup
            elif self.direct == ['x','z']:
                if !self.FARGOCA:
                    domain[1] = domain[1][3:-3]
            elif self.direct == ['y','z']:
                if !self.FARGOCA:
                    domain[0] = domain[0][3:-3]
                    domain[1] = domain[1][3:-3]
                data = np.squeeze(data)

        if data.ndim == 3 and self.direct is None: print("3 Dim - specialise your direction!")

        if not animated:
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=polar_temp)

        fig.subplots_adjust(top = 0.83)
        if polar_temp:
            fig.subplots_adjust(left=-0.2)

        if datalog10:
            cax = ax.pcolormesh(domain[0], domain[1], np.log10(data), animated=animated, linewidth=0,rasterized=True)
        else:
            cax = ax.pcolormesh(domain[0], domain[1], data, animated=animated, linewidth=0,rasterized=True)

        if xlog10:
            ax.set_xscale('log')
        if ylog10:
            ax.set_yscale('log')
        if rlog10 and sph:
            ax.set_rscale('log')

        if self.xlim is not None:
            ax.set_xlim(self.xlim)

        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        if settitle:
            if scale == "scalefree":
                orbit = int(
                    output_number * int(self.parameters["NINTERM"]) * float(self.parameters.get("DT")) / (2 * np.pi))
                title_text= "Plot of " + self.setup + " - " + tp + " at orbit " + str(orbit)
            elif scale == "years":
                year = output_number * float(self.parameters.get("DT")) * int(self.parameters["NINTERM"])
                title_text= "Plot of " + self.setup + " - " + tp + " at time " + str(round(year,2))+"y"

            elif scale == "CodeFARGO":
                year = (output_number-N_start) * float(self.parameters.get("DT")) * int(self.parameters["NINTERM"])/(2*np.pi)*11.862
                title_text= "t = {:.2f} y".format(year)

            else:
                title_text="Plot of " + self.setup + " - " + tp +' Output nr: '+ str(output_number)

            title = ax.text(0.5,1.05,title_text,
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )

        if polar and not sph: # assumes cylindrical setup!
            if self.units == "AUsph":
                dir_to_label = dict(zip(['x', 'y', 'z'], ["Azimuth - $\phi$ / $^\circ$", "r / AU", r"z/r"]))
            else:
                dir_to_label = dict(zip(['x','y','z'],["Azimuth - $\phi$ / $^\circ$", "radius", "Z"]))

            ax.set_xlabel(r''+dir_to_label[self.direct[0]])
            ax.set_ylabel(r''+dir_to_label[self.direct[1]])
        elif polar and sph:
            ax.set_thetamin(float(self.parameters.get("ZMIN",1))*180/np.pi)
            ax.set_thetamax(float(self.parameters.get("ZMAX",1))*180/np.pi)
            if self.units == "AUsph":
                dir_to_label = dict(zip(['x', 'y', 'z'], ["Azimuth - $\phi$ / $^\circ$", "r / AU", r"z/r"]))
            else:
                dir_to_label = dict(zip(['x', 'y', 'z'], ["Azimuth - $\phi$ / $^\circ$", "r", "z"]))
            ax.set_xlabel(dir_to_label[self.direct[0]])
            ax.set_ylabel(dir_to_label[self.direct[1]])
        else:
            dir_to_label = dict(zip(['x', 'y', 'z'], ["X", "Y", "Z"]))
            if self.units == "AUsph":
                dir_to_label = dict(zip(['x', 'y', 'z'], ["Azimuth - $\phi$ / $^\circ$", "r / AU", r"z/r"]))

            ax.set_xlabel(dir_to_label[self.direct[0]] + "\n")
            ax.set_ylabel(dir_to_label[self.direct[1]] + "\n")

        if filename != "":
            plt.savefig(filename, dpi = 300)
        else:
            if animated==False:
                if self.cm_max or self.cm_min is not None:
                    cax.set_clim(self.cm_min, self.cm_max)
                cbar = fig.colorbar(cax)

                if (tp =="gasdens" and datalog10):
                    cbar.set_label(r"$\mathrm{log}_{10}\left(\rho_\mathrm{gas} \left[\frac{\mathrm{g}}{\mathrm{cm}^3}\right]\right)$")
                if (tp =="gastemper" and datalog10):
                    cbar.set_label(r"$\mathrm{log}_{10}\left(T_\mathrm{gas}\left[\mathrm{K}\right]\right)$")

                return fig, ax, cax
            elif settitle:
                return cax, title
            else:
                return cax

    def plot_1D(self, output_number, filename="", tp = "gasdens", xlog10= True, ylog10 = True, scale = "scalefree", div=True, animated=False):
        import matplotlib.pyplot as plt

        n_array = np.array([int(self.parameters.get("NX", 1)),
                            int(self.parameters.get("NY", 1)),
                            int(self.parameters.get("NZ", 1))])

        data = np.fromfile(self.output_dir + tp + str(output_number) + ".dat") \
            .reshape(n_array[2], n_array[1], n_array[0])
        data = np.squeeze(data)

        if self.FARGOCA and tp=="gasdens":
            data = data*ms/(5.2*au)**3
        elif self.FARGOCA and tp=="gastemper":
            data = data*T0

        if div:
            data_0 = np.fromfile(self.output_dir + tp + str(0) + ".dat") \
                .reshape(n_array[2], n_array[1], n_array[0])
            data_0 = np.squeeze(data_0)
            data -= data_0

        if data.ndim != 1:
            if (n_array[1]!=1 and n_array[0]==1):
                print("data is averaged among radius")
                data = np.mean(data, axis=0)

        ind_to_ax = dict(zip([0, 1, 2], ['x', 'y', 'z']))
        index_not_1 = np.where(n_array != 1)[0]
        self.direct = ind_to_ax[index_not_1[0]]

        if self.direct == 'x':
            domain = self.domain[0][:-1]
        elif self.direct == 'y':
            domain = self.domain[1][3:-4]
        elif self.direct == 'z':
            domain = self.domain[3][3:-4]


        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.plot(domain, data)
        if xlog10:
            ax.set_xscale('log')
        if ylog10:
            ax.set_yscale('log')

        if scale == "scalefree":
            orbit = int(
                output_number * int(self.parameters["NINTERM"]) * float(self.parameters.get("DT")) / (2 * np.pi))
            title_text= "Plot of " + self.setup + " - " + tp + " at orbit " + str(orbit)
            title = ax.text(0.5,1.05,title_text,
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
        elif scale == "years":
            year = output_number * float(self.parameters.get("DT")) * int(self.parameters["NINTERM"])
            title_text= "Plot of " + self.setup + " - " + tp + " at time " + str(round(year,2))+"y"
            title = ax.text(0.5,1.05,title_text,
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )

        elif scale == "CodeFARGO":
            year = (output_number-N_start) * float(self.parameters.get("DT")) * int(self.parameters["NINTERM"])/(2*np.pi)*11.862
            title_text= "t = {:.2f} y".format(year)
            title = ax.text(0.5,1.05,title_text,
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )

        else:
            title_text="Plot of " + self.setup + " - " + tp +' Output nr: '+ str(output_number)
            title = ax.text(0.5,1.05,title_text,
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )

        dir_to_label = dict(zip(['x', 'y', 'z'], ["X", "Y", "Z"]))
        ax.set_xlabel(dir_to_label[self.direct])
        if self.units=="AU" and self.direct == "y":
            ax.set_xlabel("radius in AU")
        ax.set_ylabel(tp)

        if self.xlim is not None:
            ax.set_xlim(self.xlim)

        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        if filename != "":
            plt.savefig(filename, dpi=300)
        else:
            if not animated:
                return fig, ax, cax
            elif settitle:
                return cax, title
            else:
                return cax

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

    def plot_2D_video_old(self, filename, direct = None, tp = "gasdens", xlog10 = True, ylog10 = True, polar = True,
                      framesteps = 1, N = None, N_start=0, scale="scalefree",sph=False, datalog10=False):
        import matplotlib.pyplot as plt
        import multiprocessing
        if N is None:
            N = int(int(self.parameters["NTOT"])/int(self.parameters.get("NINTERM", 1)))

        def plot_and_save(i):
            plot_nr = int((i - N_start) / framesteps)
            self.plot_2D(i, direct = direct, tp=tp, xlog10 = xlog10, ylog10=ylog10, polar = polar,
                         filename = "single_frames/"+filename+"{:05d}".format(plot_nr)+".png", scale=scale,sph=sph, datalog10=datalog10)
            if i > 10 and i % round(N / 10) == 0: print(round(i / N * 100), "%")
            plt.close()
            plt.clf()
            #print('.')

        #p = multiprocessing.Pool(1)
        #p.map(plot_and_save,range(N_start, int(N/framesteps) +1, framesteps))
        for i in range(N_start, int(N/framesteps) +1, framesteps):
            plot_and_save(i)


        cmd_string = "ffmpeg -framerate 24 -i single_frames/"+filename+"%05d.png -r 24 videos/"+filename+".mp4"
        del_string = "rm -rf single_frames/"+filename+"*.png"

        os.system(cmd_string)
        os.system(del_string)

    def plot_2D_video(self, filename="", direct = None, tp = "gasdens", xlog10 = True, ylog10 = True, polar = True,
                      framesteps = 1, N = None, N_start=0, scale="scalefree",sph=False, datalog10=False):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, polar=polar)

        cax = self.plot_2D(N_start, direct = direct, tp=tp, xlog10 = xlog10, ylog10=ylog10, polar = polar, scale=scale,sph=sph, datalog10=datalog10, animated=True, fig=fig, ax=ax, settitle=False, N_start=N_start)
        if self.cm_max or self.cm_min is not None:
            cax.set_clim(self.cm_min, self.cm_max)
        cbar = fig.colorbar(cax)

        if (tp =="gasdens" and datalog10):
            cbar.set_label(r"$\mathrm{log}_{10}\left(\rho_\mathrm{gas} \left[\frac{\mathrm{g}}{\mathrm{cm}^3}\right]\right)$")
        if (tp =="gastemper" and datalog10):
            cbar.set_label(r"$\mathrm{log}_{10}\left(T_\mathrm{gas}\left[\mathrm{K}\right]\right)$")

        if N is None:
            N = int(int(self.parameters["NTOT"])/int(self.parameters.get("NINTERM", 1)))

        def plot_and_save(i):
            im, title = self.plot_2D(i, direct = direct, tp=tp, xlog10 = xlog10, ylog10=ylog10, polar = polar, scale=scale,sph=sph, datalog10=datalog10, animated=True, fig=fig, ax=ax, N_start=N_start)
            if self.cm_max or self.cm_min is not None:
                im.set_clim(self.cm_min, self.cm_max)

            if i > 10 and i % round(N / 10) == 0: print(round(i / N * 100), "%")
            return im, title

        print('creating images...')
        ims=[]
        for i in range(N_start, int(N/framesteps) +1, framesteps):
            im, title =plot_and_save(i)
            ims.append([im, title])

        print('animating...')
        ani = animation.ArtistAnimation(fig, ims, interval=50)
        #ani = animation.FuncAnimation(fig, plot_and_save, frames=range(N_start, int(N/framesteps) +1, framesteps), interval=15)
        if filename!="":
            print('saving...')
            writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save("videos/"+filename+".mp4", writer=writer)
        plt.show()
        #ani.save("videos/"+filename+".mp4")
        print('done! video saved as '+filename+'.mp4')
        return ani



    #def plot_1D_video(self, filename, tp = "gasdens", xlog10 = True, ylog10 = True,
    #                  framesteps = 1, N = None, div=True, N_start=0, scale="scalefree"):
        #import matplotlib.pyplot as plt
        #if N is None:
        #    N = int(int(self.parameters["NTOT"])/int(self.parameters.get("NINTERM", 1)))

        #def plot_and_save(i):
        #    plot_nr = int((i-N_start)/framesteps)
        #    self.plot_1D(i,  tp=tp, xlog10 = xlog10, ylog10=ylog10,
        #                 filename = "single_frames/"+filename+"{:05d}".format(plot_nr)+".png", div=div, scale=scale)
        #    if i > 10 and i % round(N / 10) == 0: print(round(i / N * 100), "%")
        #    plt.close()

        #for i in range(N_start, int(N/framesteps) +1, framesteps):
        #    plot_and_save(i)

        #cmd_string = "ffmpeg -framerate 24 -i single_frames/"+filename+"%05d.png -r 24 videos/"+filename+".mp4"
        #del_string = "rm -rf single_frames/"+filename+"*.png"

        #os.system(cmd_string)
        #os.system(del_string)
