import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_excel(filename):

    header=np.arange(0,12)

    df = pd.read_excel(filename,
        skiprows=header,
        usecols=[1,2,3,4,6,7,8,9],
        index_col = [0,3])

    df = df.sort_index()

    return df

def transectPlotter(df,transects,filename='../fig/isotope_transect.pdf'):
   
    """
    WHAT THIS DOES:
    ---------------
    This is designed to plot the isotope data relative to height.
    
    """
    fig,(ax_H2,ax_O18) = plt.subplots(1,2,figsize=(4,4),sharey=True)

    for i in transects:
        H2 = df.loc[transects[i]]["d2H"].values
        H2_err = df.loc[transects[i]]["s.d. d2H"].values
        O18_err = df.loc[transects[i]]["s.d. d18O"].values
        O18 = df.loc[transects[i]]["d18O"].values
        yi = df.loc[transects[i]]["column height (cm)"].values
        yi_err = df.loc[transects[i]]["s.d. height"].values

        ax_O18.plot(O18,yi,'.',label= i)
        ax_O18.errorbar(O18,yi,yerr =  yi_err,xerr= O18_err,
                marker = '.',
                markersize = 0,
                lw = 0,
                elinewidth = 1,
                ecolor = 'black')

        ax_H2.plot(H2,yi,'.',color='firebrick',label= i)
        ax_H2.errorbar(H2,yi,yerr =  yi_err,xerr= H2_err,
                marker = '.',
                markersize = 0,
                lw = 0,
                elinewidth = 1,
                ecolor = 'black')

    ax_O18.set(xlabel = '$\delta^{18}$O [‰] (VSMOW)')
    ax_H2.set(ylabel = 'height (cm)', xlabel = '$\delta^{2}$H [‰] (VSMOW)')

    plt.tight_layout()
    plt.savefig(filename,dpi=300)
    plt.show()

def crossPlotter(df,transects,filename='../fig/isotope_crossplot_sample.pdf'):

    
    """
    WHAT THIS DOES:
    ---------------
    Designed to plot the dH and d18O of samples with error bars.
    'transects' can be selected to plot certain groups of isotope
    samples which belong together.
    
    """

    fig, ax = plt.subplots(figsize = (8,5))


    for i in transects:

        yi = df.loc[transects[i]]["d2H"].values
        xi_err = df.loc[transects[i]]["s.d. d18O"].values
        yi_err = df.loc[transects[i]]["s.d. d2H"].values
        xi = df.loc[transects[i]]["d18O"].values

        ax.plot(xi,yi,'.',label= i)
        ax.errorbar(xi,yi,yerr =  yi_err,xerr= xi_err,
            marker = '.',
            markersize = 0,
            lw = 0,
            elinewidth = 1,
            ecolor = 'black')


    ax.set(ylabel = '$\delta^{2}$H [‰] (VSMOW)', xlabel = '$\delta^{18}$O [‰] (VSMOW)')
    ax.legend()
    plt.savefig(filename, dpi = 300)

    plt.show()
