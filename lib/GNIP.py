""" A module with functions I called GNIP.
------------------------------------------

It contains functions for reading the WISER csv files, 
computing the Precipitation Weighted Least Squares Regression (PWLSR)
and plotting of all sorts. 

It can be loaded in a Jupyter Notebook using the code:

 %aimport GNIP

 ------------------------------------------"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp

def ANIP_excel_read(filename):

	"""
	For some of the ANIP data, which comes as an excel file?

	"""
	names=['End date','Precipitation','Temperature','O18','s.d. d18O',"H2","s.d. d2H","H3", 's.d. H3']
	columns=[i for i in range(1,13,1)]
	dataframe = pd.read_excel(filename,
		sheet_name="Daten_T",
		skiprows=[0,1,2],
		usecols=columns,
		parse_dates = True,day_first=True, index_col=[0,1])
	dataframe.columns=names
	dataframe.rename_axis(['Site','Date'], inplace=True)

	return dataframe

def csv_read(filename):

	""" 
	Takes the filename and returns a clean pandas dataframe.
	"""

	dataframe = pd.read_csv(filename,
		parse_dates=True,
		usecols=[2,12,13,14,16,18,23], # these columns correspond to Date, Begin of Period, O18, H2, Precipitation
		index_col=["Site","Date"])
	dataframe.dropna(inplace=True)

	return dataframe

def weighted_avg(dataframe,site,series,weights):
	""" 
	Takes a GNIP dataframe and returns a dataframe the amount weighted annual mean of monthly precipitation.
	"""
	
	year_avg = (dataframe.loc[site][series]*dataframe.loc[site][weights]).resample("y").sum()/(dataframe.loc[site][weights].resample("y").sum())

	return year_avg

def period_boxplot(dataframe,site,filename='sample_boxplot.pdf',freq='Q'):
	
	#create a figure with two subplots. Size 8 inches by 5
	fig,(ax_H2,ax_O18,ax_d_ex) = plt.subplots(1,3,figsize=(13,4))

	#define the quarters. 
	quarters = ["Winter","Spring","Summer","Autumn"]
	months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

	#building the two subfigures
	dataframe["d-excess"]=dataframe["H2"]-8*dataframe["O18"]
	for ax,series,labels in zip((ax_H2,ax_O18,ax_d_ex),("H2","O18","d-excess"),("$\delta^{2}$H","$\delta^{18}$O",'d-excess')):
		if freq=='Q':
			bp = dataframe.loc[site].boxplot(column=[series], 
				by=dataframe.loc[site].index.quarter, 
				ax=ax, 
				showfliers=True,
				grid=False,return_type = 'dict')
			ax.set_xticklabels(quarters)
		else:
			if freq=='M' or freq=='m':
				bp = dataframe.loc[site].boxplot(column=[series], 
					by=dataframe.loc[site].index.month, 
					ax=ax, 
					showfliers=True,
					grid=False,return_type = 'dict')
				ax.set_xticklabels(months)
			else:
				print("freq must be either Q (quarterly) or M (monthly)")

		#setting the appropriate labels
		ax.set_ylabel(labels+" [‰] (VSMOW)")
		ax.set_xlabel("")
		
		ax.tick_params(direction='in',top=True,right=True)

		#further line customisations using a workaround
		#(https://stackoverflow.com/questions/35160956/pandas-boxplot-set-color-and-properties-for-box-median-mean)
		[[item.set_color('firebrick') for item in bp[key]['medians']] for key in bp.keys()]
		[[item.set_color('black') for item in bp[key]['boxes']] for key in bp.keys()]
		[[item.set_color('black') for item in bp[key]['whiskers']] for key in bp.keys()]

	#cleaning up the figure of automatically generated subtitles
	plt.suptitle("")
	plt.tight_layout()
	plt.savefig(filename, dpi= 300)
	plt.show()


def PWLSR(dataframe,site):

	"""Computes the Precipitation Weighted Least Squares Regression on a standard monthly GNIP dataset.
	Based on the formulae by Hughes and Crawford.
	"""

	xi = dataframe.loc[site]["O18"]
	yi = dataframe.loc[site]["H2"]
	pi = dataframe.loc[site]["Precipitation"]
	n = len(xi)

	#plug into formula by Hughes and Crawford (2012), equation (9) for the slope
	k1 = (pi*xi*yi).sum()-(((pi*xi).sum())*((pi*yi).sum())/(pi.sum())) 
	k2 = (pi*xi**2).sum()-(((pi*xi).sum())**2/(pi.sum()))
	a = k1/k2

	#intercept (equation (10))
	b = (((pi*yi).sum())-a*((pi*xi).sum()))/(pi.sum())

	#S_yx_w - equation 13.
	k3 = n/(n-2)
	k4 = ((pi*yi**2).sum()-b*(pi*yi).sum()-a*(pi*yi*xi).sum())/(pi.sum())
	S_yx_w = (k3*k4)**(0.5)

	#sigma_aw (equation (11))
	k5 = n/(pi.sum())
	k6 = (pi*xi**2).sum()-(((pi*xi).sum())**2)/(pi.sum())
	sigma_aw = S_yx_w/(k5*k6)**(0.5)

	#sigma_bw (equation (12))
	k7 = ((pi*xi**2).sum())/(n*(((pi*xi**2).sum())-(((pi*xi).sum())**2)/(pi.sum())))
	sigma_bw = S_yx_w*k7**(0.5)

	#r2, equation (14)
	k8 = (((pi*xi*yi).sum())-(((pi*xi).sum())*((pi*yi).sum())/(pi.sum())))**2
	k9 = ((pi*xi**2).sum()-((pi*xi).sum())**2/(pi.sum()))*((pi*yi**2).sum()-((pi*yi).sum())**2/(pi.sum()))
	r2 = k8/k9

	#this returns a dictionary with the computed variables. 
	return dict(zip(['slope','intercept','$\sigma_{a(w)}$','$\sigma_{b(w)}$','std. error','$R^2$','N'],[a,b,sigma_aw,sigma_bw,S_yx_w,r2,n]))

def OLSR_slope_estimate(xi,yi):

	n = len(xi)
	A = (1/(n-2))*np.sum((yi-yi.mean())**2)
	B = np.sum((xi-xi.mean())**2)
	sigma_a = np.sqrt(A/B)


def GMWL(x):
	#global meteoric waterline as defined by Craig (1961)
    return 8*x + 10

def LMWL_PWLSR(x,dataframe,site):
	#local meteoric waterline as defined by precipitation weighted least squares regression
	
	a = PWLSR(dataframe,site)['slope']
	b = PWLSR(dataframe,site)['intercept']

	return a*x + b

def LMWL_OLSR(x,dataframe,site):
	slope, intercept, r_value, p_value, std_err = stats.linregress(dataframe.loc[site]["O18"],dataframe.loc[site]["H2"])

	return slope*x+intercept

def figure_LMWL(dataframe,site,plot_title="Plot title",filename ="../fig/monthly_GNIP_samples.pdf",ax=None,legend='off'):

	#fig,ax = plt.subplots(figsize = (8,5))

	plot_monthly(dataframe,site,ax=ax)
	plot_LMWL(dataframe,site,ax=ax,option='grey')
	
	if legend == 'on':
		ax.legend(prop={'size': 8})
	
	ax.set_xlabel("$\delta^{18}$O [‰] (VSMOW)")
	ax.set_ylabel("$\delta^{2}$H [‰] (VSMOW)")
	ax.title.set_text(plot_title)

	ax.tick_params(direction='in',top=True,right=True)
	
	params = PWLSR(dataframe,site)
	posx=ax.get_xlim()[1]-(ax.get_xlim()[1]-ax.get_xlim()[0])/3
	posy=ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])/4
	ax.text(posx,posy,"y = {:.2f}x {:+.2f}\n$R^2$={:.2f}".format(params['slope'],params['intercept'],params['$R^2$']),color = 'black')

	#plt.title(plot_title)
	#plt.tight_layout()
	#plt.savefig(filename,dpi= 300)
	#plt.show()
	return ax

def plot_monthly(dataframe,site,ax=None):
	O18 = dataframe.loc[site]["O18"]
	H2 = dataframe.loc[site]["H2"]

	ax.plot(O18,H2,'D',markersize=3,label='Samples') #points, monthly GNIP data

	return ax


def plot_LMWL(dataframe,site,ax=None, option='color'):
	#plot first the GMWL.

	O18 = dataframe.loc[site]["O18"]
	H2 = dataframe.loc[site]["H2"]

	#statistics:
	n= O18.size
	m=2
	dof=n-m
	t = stats.t.ppf(0.975,n-m)
	s_err = PWLSR(dataframe,site)['std. error']
	resid=H2-LMWL_PWLSR(O18,dataframe,site)

	xi = np.linspace(min(O18)-1,max(O18)+1,100)
	yi = LMWL_PWLSR(xi,dataframe,site)

	#ax.plot(xi,GMWL(xi),label="Global Meteoric Water Line") #the GMWL defined by Craig (1961)
	if option=='color':
		ax.plot(xi,yi,lw=0.75,color = 'firebrick',label="Local Meteoric Water Line (PWLSR)") #Precipitation Weighted LSR
	#ax.plot(xi,LMWL_OLSR(xi,dataframe,site),label="Local Meteoric Water Line (OLSR)") #Ordinary LSR
	else:
		if option == 'grey':
			ax.plot(xi,yi,lw=0.75,color = 'black',label="Local Meteoric Water Line (PWLSR)")

	
	plot_ci_manual(t,s_err,n,O18,xi,yi,ax=ax)
	plot_pi_manual(t,s_err,n,O18,xi,yi,ax=ax)

	return ax

def plot_pi_manual(t,s_err,n,x,x2,y2,ax=None):
	pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
	ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
	ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
	ax.plot(x2, y2 + pi, "--", color="0.5")

	return ax


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="grey", edgecolor="",alpha= 0.25,label="95% Confidence Interval")

    return ax

def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """ 
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(xs, ys + resamp_resid, 1)                   
        # Plot bootstrap cluster
        ax.plot(xs, sp.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax
   
def get_stats(dataframe,site):
	O18 = dataframe.loc[site]["O18"]
	H2 = dataframe.loc[site]["H2"]
	d_ex = H2-8*O18

	for i in (O18,H2,d_ex):
		
		return i.min(),i.max(),i.mean(),i.median(),i.std()


