import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

def grid_interp(X,Y,Z):
    xi = np.linspace(-10,2,300)
    yi = np.linspace(-3.5,7,300)
    zi = griddata((X, Y), Z , (xi[None,:], yi[:,None]), method='linear')
    
    return xi,yi,zi

def kriging_interpolator(X,Y,Z):
    gridx = np.linspace(-10,2,300)
    gridy = np.linspace(-3.5,7,300)

    # Create the ordinary kriging object. Required inputs are the X-coordinates of
    # the data points, the Y-coordinates of the data points, and the Z-values of the
    # data points. If no variogram model is specified, defaults to a linear variogram
    # model. If no variogram model parameters are specified, then the code automatically # calculates the parameters by fitting the variogram model to the binned
    # experimental semivariogram. The verbose kwarg controls code talk-back, and
    # the enable_plotting kwarg controls the display of the semivariogram.
    OK = OrdinaryKriging(X, Y, Z, variogram_model='linear',verbose=False, enable_plotting=False)
    # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular # grid of points, on a masked rectangular grid of points, or with arbitrary points. # (See OrdinaryKriging.__doc__ for more information.)
    zi, ss = OK.execute('grid', gridx, gridy,mask=True)
    
    return gridx,gridy,zi,ss
    # Writes the kriged grid to an ASCII grid file.
    #kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
    
def grid_plotter(dat, method='linear'):
    if method=='ok':
        kxi,kyi,kzi,ssi=kriging_interpolator(dat.X,dat.Y,dat.Z)
        xi,yi,zi=grid_interp(dat.X,dat.Y,dat.Z)
        
        masked_zi=kzi*(zi/zi)
        masked_ssi=ssi*(zi/zi)
        
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    
    
        ax1.contour(xi, yi, masked_zi, 15, linewidths = 0.5, colors = 'k')
        zplot = ax1.pcolormesh(xi, yi, masked_zi, cmap = plt.get_cmap('rainbow'),vmin=-2,vmax=5)
        ax1.set_xlabel("Easting (m)")
        ax1.set_ylabel("Northing (m)")

        ax1.scatter(dat.X, dat.Y, marker = 'o', c = 'b', s = 5, zorder = 10)
        ax1.set_xlim(-10,0)
        ax1.set_ylim(-3.5,7)
        plt.colorbar(zplot,ax=ax1,label='height (m)')
        
        ax2.contour(xi, yi, masked_ssi, 15, linewidths = 0.5, colors = 'k')
        ssplot = ax2.pcolormesh(xi, yi, masked_ssi, cmap = plt.get_cmap('rainbow'))
        ax2.scatter(dat.X, dat.Y, marker = 'o', c = 'b', s = 5, zorder = 10)
        plt.colorbar(ssplot,ax=ax2,label='variance (m$^2$)')
        ax2.set_ylabel("Northing (m)")
        ax2.set_xlabel("Easting (m)")

        ax2.set_xlim(-10,0)
        ax2.set_ylim(-3.5,7)
        plt.show()
        
    else:
        xi,yi,zi=grid_interp(dat.X,dat.Y,dat.Z)
    
    

