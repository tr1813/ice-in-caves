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
    ax.fill_between(dat.index, dat, dat.median(), where=dat>=dat.median(), color='#ff8080', lw = 0.5, alpha = 0.5)
    ax.fill_between(dat.index, dat, dat.median(), where=dat<=dat.median(), color='#004d99', lw = 0.5, alpha = 0.5)
    ax.axhline(dat.median(), color='black', lw=0.5)
    ax.legend(loc = 0)

    ax.set(xlabel = 'Time', ylabel = 'Temperature $\degree$ C')
    plt.savefig('fig/'+params['start']+'_'+params['end']+'_'+params['filename'], dpi = 300)

    plt.show()
