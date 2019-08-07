import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def read_excel(filename):

	columns=dict(zip(["transect","column height (cm)","s.d. height","protocol","d18O","s.d. d18O","d2H","s.d. d2H","layer","layer type"],
		[1,2,3,4,6,7,8,9,13,14]))

	header=np.arange(0,15)

	df = pd.read_excel(filename,
		skiprows=header,
		usecols=[columns[i] for i in columns],
		index_col = [0,3])

	df = df.sort_index()

	return df

def get_layers(df,transect,protocol):
	current_layer=df.loc[(transect,protocol)]['layer'].values[0]
	current_height=df.loc[(transect,protocol)]['column height (cm)'].values[0]
	current_type=df.loc[(transect,protocol)]['layer type'].values[0]
	layer_bounds= []

	for next_layer,next_height,next_type in zip(df.loc[(transect,protocol)]["layer"],df.loc[(transect,protocol)]["column height (cm)"],df.loc[(transect,protocol)]["layer type"]):
		if current_layer != next_layer:

			n= next_layer-current_layer
			for i in range(1,n+1,1):
				layer_height=current_height+i*(next_height-current_height)/(n+1)
				layer_bounds.append((layer_height,current_type))
			current_layer=next_layer
			current_type=next_type
		current_height=next_height
	return layer_bounds


def transectPlotter(df,transects,filename='../fig/isotope_transect.pdf'):
   
	"""
	WHAT THIS DOES:
	---------------
	This is designed to plot the isotope data relative to height.

	INPUT:
	------
	df: a pandas dataframe, read from iso.read_csv(), which has a multi-index
	transects: a list of transects (usually numbered), the second index column.
	Does not work for individual samples of ice or drips which have column height -999.

	RETURNS:
	--------
	plots the O18 and H2 transect according to height in the column. 
	
	"""
	fig,(ax_H2,ax_O18) = plt.subplots(1,2,figsize=(4,5),sharey=True)

	#setting up the custom legend items
	inc_rich= mpatches.Patch(facecolor='paleturquoise',edgecolor='black',label="inclusion rich")
	inc_poor= mpatches.Patch(facecolor='lightblue',edgecolor='black',label="transparent ice")
	line= Line2D([0],[0],color= 'teal',linestyle='dotted',lw=0.75,label='layer boundary')

	ax_H2.legend(loc=(0,1.1),handles=[inc_rich,inc_poor,line])

	for i in transects:
		

		# getting the data series from the pandas dataframe
		H2 = df.loc[transects[i]]["d2H"].values
		H2_err = df.loc[transects[i]]["s.d. d2H"].values
		O18_err = df.loc[transects[i]]["s.d. d18O"].values
		O18 = df.loc[transects[i]]["d18O"].values
		yi = df.loc[transects[i]]["column height (cm)"].values
		yi_err = df.loc[transects[i]]["s.d. height"].values

		#d18O data plotted on right hand side, by default blue points with errorbars.
		ax_O18.plot(O18,yi,'.',label= i)
		ax_O18.errorbar(O18,yi,yerr =  yi_err,xerr= O18_err,
			marker = '.',
			markersize = 0,
			lw = 0,
			elinewidth = 1,
			ecolor = 'black')

		#d2H data plotted on leftt hand side, changed to red points with black error bars.
		ax_H2.plot(H2,yi,'.',color='firebrick',label= i)
		ax_H2.errorbar(H2,yi,yerr =  yi_err,xerr= H2_err,
			marker = '.',
			markersize = 0,
			lw = 0,
			elinewidth = 1,
			ecolor = 'black')

		#printing the layer boundaries
		layer_bounds = get_layers(df,transects[i],'SHALLOW')
		
		#for each plot, add the layers in dotted lines and fill in the layers depending on 
		#presence/absence of inclusions.

		for axes in (ax_H2,ax_O18):

			l0=axes.get_ylim()[0]
			for l in layer_bounds:
			
				axes.axhline(y=l[0], color = "teal",lw=0.75,linestyle = "dotted")
				if l[1]==0:
					axes.add_patch(Rectangle((axes.get_xlim()[0],l[0]),
						width = axes.get_xlim()[1]-axes.get_xlim()[0],
						height =l0-l[0], 
						fill =True,
						facecolor = 'paleturquoise',
						alpha = 0.75))
				else:
					axes.add_patch(Rectangle((axes.get_xlim()[0],l[0]),
						width = axes.get_xlim()[1]-axes.get_xlim()[0],
						height =l0-l[0], 
						fill =True,
						facecolor = 'lightblue',
						alpha = 0.75))
				
				l0=l[0]
			if l[1]==0:
				axes.add_patch(Rectangle((axes.get_xlim()[0],axes.get_ylim()[1]),
					width = axes.get_xlim()[1]-axes.get_xlim()[0],
					height =l0-axes.get_ylim()[1], 
					fill =True,
					facecolor = 'paleturquoise',
					alpha = 0.75))
			else:
				axes.add_patch(Rectangle((axes.get_xlim()[0],axes.get_ylim()[1]),
					width = axes.get_xlim()[1]-axes.get_xlim()[0],
					height =l0-axes.get_ylim()[1], 
					fill =True,
					facecolor = 'lightblue',
					alpha = 0.75))


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

	INPUT:
	------
	df: a pandas dataframe, read from iso.read_csv(), which has a multi-index
	transects: a list of transects (usually numbered), the second index column.
	This does work for individual samples of ice or drips which have column height -999.
	
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