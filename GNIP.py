def GNIP_csv_reader(filename):

	""" 
	Takes the filename and returns a clean pandas dataframe.
	"""

	dataframe = pd.read_csv(filename,
		parse_dates=True,
		usecols=[12,13,14,16,18,23], # these columns correspond to Date, Begin of Period, O18, H2, Precipitation
		index_col="Date")
	dataframe.dropna(inplace=True)

	return dataframe

def GNIP_weighted_avg(dataframe,series,weights):
	""" 
	Takes a GNIP dataframe and returns a dataframe the amount weighted annual mean of monthly precipitation.
	"""
	
	year_avg = (dataframe[series]*dataframe[weights]).resample("y").sum()/(dataframe[weights].resample("y").sum())

	return year_avg

def GNIP_quarterly_boxplot(dataframe,filename='sample_boxplot.pdf'):
	
	#create a figure with two subplots. Size 8 inches by 5
	fig,(ax_H2,ax_O18) = plt.subplots(1,2,figsize=(8,5))

	#define the quarters. 
	quarters = ["Winter","Spring","Summer","Autumn"]

	#building the two subfigures
	for ax,series,labels in zip((ax_H2,ax_O18),("H2","O18"),("$\delta^{2}$H","$\delta^{18}$O")):
		bp = dataframe.boxplot(column=[series], 
			by=dataframe.index.quarter, 
			ax=ax, 
			showfliers=True,
			grid=False,return_type = 'dict')

		#setting the appropriate labels
		ax.set_ylabel(labels+" [‰] (VSMOW)")
		ax.set_xlabel("")
		ax.set_xticklabels(quarters)
		ax.tick_params(direction='in',top=True)

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


def GNIP_PWLSR(dataframe):

	"""Computes the Precipitation Weighted Least Squares Regression on a standard monthly GNIP dataset.
	Based on the formulae by Hughes and Crawford.
	"""

	xi = dataframe["O18"]
	yi = dataframe["H2"]
	pi = dataframe["Precipitation"]
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
	return dict(zip(['slope','intercept','$\sigma_{a(w)}$','$\sigma_{b(w)}$','std. error','$R^2$'],[a,b,sigma_aw,sigma_bw,S_yx_w,r2]))
