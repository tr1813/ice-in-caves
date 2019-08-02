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

	for ax,series,labels in zip((ax_H2,ax_O18),("H2","O18"),("$\delta^{2}$H","$\delta^{18}$O")):
		bp = dataframe.boxplot(column=[series], 
			by=dataframe.index.quarter, 
			ax=ax, 
			showfliers=True,
			grid=False,return_type = 'dict')

		ax.set_ylabel(labels+" [‰] (VSMOW)")
		ax.set_xlabel("")
		ax.set_xticklabels(quarters)
		ax.tick_params(direction='in',top=True)
		[[item.set_color('firebrick') for item in bp[key]['medians']] for key in bp.keys()]
		[[item.set_color('black') for item in bp[key]['boxes']] for key in bp.keys()]
		[[item.set_color('black') for item in bp[key]['whiskers']] for key in bp.keys()]


	plt.suptitle("")
	plt.tight_layout()
	plt.savefig(filename)
	plt.show()
