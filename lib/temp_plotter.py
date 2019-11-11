import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tempPlotter(mypd,params):

    
    """
    WHAT THIS DOES:
    ---------------
    Designed to plot a temperature curve, 
    with positive (above 0°C) filled in with red, 
    and negative values filled with blue.
    
    USAGE:
    ------
    mypd: a pandas dataframe
    params: a python dictionary of the following structure:
    
    params = {
    'filename':'place_holder_name',      ## the exported file name
    'Title' : 'place_holder_title' ,     ## the plot title
    'plot_dims' : (width,height),        ## needs to be a tuple with two integers
    'series' : 'place_holder_series',    ## this string is the name of series header in a pandas dataframe.
    'start' :'start_date',               ## again a string for start date in yyyy-mm-dd format
    'end' : 'end_date'}
    
    """

    fig, ax = plt.subplots(figsize = params['plot_dims'])
    
    dat = mypd[params['series']][params['start']:params['end']]

    ax.set_title(params['Title'])
    ax.plot(dat, color='black', lw = 0.5, alpha = 0.5) # this is the curve that is plotted
    #ax.fill_between(dat.index, dat, 0, where=dat>=0, color='#ff8080', lw = 0.5, alpha = 0.5) #red fill for positive values
    #ax.fill_between(dat.index, dat, 0, where=dat<=0, color='#004d99', lw = 0.5, alpha = 0.5) #blueish fill for negative values
    ax.legend(loc = 0)

    ax.set(xlabel = 'Time', ylabel = 'Temperature $\degree$ C')
    ax.grid()
    ax.set(xlabel = 'Date Time', ylabel = params['series'],xlim=(params['start'],params['end']))
    ax.axhline(0,lw = 0.75, color = "black")
    ax.tick_params(direction='in',top=True)
    plt.savefig(params['start']+'_'+params['end']+'_'+params['filename']+'.pdf', dpi = 300)

    plt.show()


def rrReader(filepath):

    filepath= filepath
    header = np.arange(0,11)
    columns = dict(zip(["DATUM","RR"],[1,2]))
    rrpd = pd.read_excel(filepath,skiprows=header,usecols=[columns[i] for i in columns], index_col= 'DATUM')

    rrpd.index= pd.to_datetime(rrpd.index, dayfirst=True)
    rrpd = rrpd.query('RR != "---"')

    rrpd = pd.to_numeric(rrpd['RR'])

    return rrpd

def rrPlotter(ax,mypd):
    dat= mypd.resample('D').sum()
    ax.bar(x=dat.index,height=dat.values,facecolor='teal')

    return ax



def TempRFReader(filepath):

    filepath = filepath
    header = np.arange(0,11)
    columns = dict(zip(["DATUM","TL","TLMAX","TLMIN","RF"],[1,2,3,4,5]))
    mypd=pd.read_excel(filepath,skiprows=header,usecols=[columns[i] for i in columns], index_col ='DATUM')

    mypd.index = pd.to_datetime(mypd.index,dayfirst=True)

    mypd = mypd.query('TL != "---"')
    mypd = mypd.query('RF != "---"')

    for i in mypd:
        pd.to_numeric(mypd[i])

    return mypd
