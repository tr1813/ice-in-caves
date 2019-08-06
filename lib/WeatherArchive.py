"""Another module.
Reads of the csv files of weather station archives like  Feuerkogel (AT) or Vogel (SLO)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename, station_name,freq):
	dataframe = pd.read_csv(filename,
		skiprows = [0,1,2,3,4,5],
		parse_dates = True, dayfirst = True,
		index_col = 'Local time in '+station_name)

	dataframe.index = pd.to_datetime(dataframe.index)

	dataframe["Temp, (*C)"] = pd.to_numeric(dataframe["T"])
	dataframe["RRR"] = pd.to_numeric(dataframe["RRR"])

	dataframe = dataframe.resample(freq).mean()
	return dataframe

def getPrecip(df):
	
	RRR_daily = df["RRR"].resample("D").sum()

	return RRR_daily

def BarPlot(df,start='2018',end='2020',filename= '../fig/precipitation_daily.pdf'):
	
	fig,ax = plt.subplots(figsize = (8,4))

	series=getPrecip(df)[start:end]

	ax.set(ylabel='Precipitation (mm/day)',xlabel = "Date Time")
	ax.tick_params(direction="in",top = True,right = True)

	plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

	ax.bar(series.index,series.values)

	plt.tight_layout()
	plt.savefig(filename,dpi=300)
	plt.show()

