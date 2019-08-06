"""A module with functions called HOBO
-------------------------------------

This mainly reads off the HOBO mobile exported csv files, which
contain temperature, humidity and dew point measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(filename,freq):
    try: 
        dataframe = pd.read_csv(filename,
            skiprows=[0], 
            index_col = 'Date Time - GMT +01:00', 
            usecols = ['Date Time - GMT +01:00','Temp, (*C)','RH, (%)'],
            parse_dates =True,
            dayfirst=True)
    except:
        dataframe = pd.read_csv(filename, 
    skiprows=[0],
            index_col = 'Date Time - GMT +02:00', 
            usecols = ['Date Time - GMT +02:00','Temp, (*C)','RH, (%)'],
            parse_dates =True,
            dayfirst=True)
    dataframe.index = pd.to_datetime(dataframe.index)
    for i in dataframe:
        dataframe[i] = pd.to_numeric(dataframe[i])

    label= filename[7:19]
    dataframe = dataframe.resample(freq).mean()
    return (dataframe,label)



def simplePlotter(dataframes,series,start,end,filename = "../fig/temperature_plot.pdf",):

    fig, ax = plt.subplots(figsize = (8,5))

    dat2 = dataframes[-1][0][series][start:end]
    ax.plot(dat2, lw = 0.75, label = dataframes[-1][1],color = "gray")
    
    for i in dataframes[:-1]:
        dat = i[0][series][start:end]
        ax.plot(dat, lw = 0.75,label= i[1]) # this is the curve that is plotted

    ax.legend()
    ax.grid()
    ax.set(xlabel = 'Date Time', ylabel = series,xlim=(start,end))
    ax.axhline(0,lw = 0.75, color = "black")
    ax.tick_params(direction='in',top=True)
    plt.savefig(filename, dpi = 300)

    plt.show()

def doublePlotter(dataframes,series,start,end,filename = "../fig/temperature_plot.pdf",):
    
    """
    WHAT THIS DOES:
    ---------------
    Designed to plot a temperature curve, with another curve for outside
    temperature with a different scale
    
    USAGE:
    ------
    dataframe
    
    """

    fig, ax = plt.subplots(figsize = (8,5))
    
    dat2 = dataframes[-1][0][series][start:end]
    ax2 = ax.twinx()
    ax2.plot(dat2, lw = 0.75, label = dataframes[-1][1],color = "gray")
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Temperature $\degree$ C at {} station'.format(dataframes[-1][1]),color='gray')
    ax2.tick_params(direction= 'in')

    for i in dataframes[:-1]:
        dat = i[0][series][start:end]
        ax.plot(dat, lw = 0.5,label= i[1]) # this is the curve that is plotted

    
    

    ax.legend(loc='upper left')
    ax.grid()
    ax.set(xlabel = 'Date Time', ylabel = 'Temperature $\degree$ C',xlim=(start,end))
    ax.tick_params(direction='in',top=True)
    plt.savefig(filename, dpi = 300)

    plt.show()
