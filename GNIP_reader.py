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
	fig,(ax_O18,ax_H2) = plt.subplots(1,2,figsize=(10,5))

	quarters = ["Winter","Spring","Summer","Autumn"]

	for ax,series,labels in zip((ax_H2,ax_O18),("H2","O18"),("$\delta^{2}$H","$\delta^{18}$O")):
		dataframe.boxplot(column=[series], by=dataframe.index.quarter, ax=ax, showfliers=True,grid=False)

		ax.set_ylabel(labels+" [‰]")
		ax.set_xlabel("")
		ax.set_xticklabels(quarters)
		ax.tick_params(direction='in',top=True)

	plt.suptitle("")
	plt.tight_layout()
	plt.savefig(filename)
	plt.show()
