import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import lib.GNIP as gnip
from scipy import stats

def read_excel_old(filename):
	columns = dict(zip(["sample name","d.18O"],[0,1]))

	df= pd.read_excel(filename,usecols=[columns[i] for i in columns])

	df= df.sort_index()

	return df

def read_excel(filename):

	columns=dict(zip(["transect","column height (cm)","s.d. height","protocol","d18O","s.d. d18O","d2H","s.d. d2H","layer","layer type"],
		[1,2,3,4,6,7,8,9,13,14]))

	header=np.arange(0,15)

	df = pd.read_excel(filename,
		skiprows=header,
		usecols=[columns[i] for i in columns],
		index_col = [0,3])
	
	df = df.sort_index(axis=0,level='transect')
	#df = df.sort_values(by=["column height (cm)"])

	return df

def get_layers(df,transect):
	current_layer=df.loc[transect]['layer'].values[0]
	current_height=df.loc[transect]['column height (cm)'].values[0]
	current_type=df.loc[transect]['layer type'].values[0]
	layer_bounds= []

	for next_layer,next_height,next_type in zip(df.loc[transect]["layer"],df.loc[transect]["column height (cm)"],df.loc[transect]["layer type"]):
		if current_layer != next_layer:

			n= int(next_layer)-int(current_layer)
			for i in range(1,n+1,1):
				layer_height=current_height+i*(next_height-current_height)/(n+1)
				layer_bounds.append((layer_height,current_type))
			current_layer=next_layer
			current_type=next_type
		current_height=next_height
	return layer_bounds


def transectPlotter(df,transects,filename='../fig/isotope_transect.pdf',figsize = (6,9),legend = True):
   
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
	fig,(ax_H2,ax_O18,ax_dexcess) = plt.subplots(1,3,figsize=figsize,sharey=True)

	#setting up the custom legend items
	inc_rich= mpatches.Patch(facecolor='white',edgecolor='black',label="firn derived")
	inc_poor= mpatches.Patch(facecolor='lightblue',edgecolor='black',label="congelation")
	line= Line2D([0],[0],color= 'grey',lw=0.45,label='layer boundary')

	if legend == True:
		ax_H2.legend(loc=(0,1.1),handles=[inc_rich,inc_poor,line])

	for i in transects:
		

		# getting the data series from the pandas dataframe
		H2 = df.loc[transects[i]]["d2H"].values
		H2_err = df.loc[transects[i]]["s.d. d2H"].values
		O18_err = df.loc[transects[i]]["s.d. d18O"].values
		O18 = df.loc[transects[i]]["d18O"].values
		dexcess = H2-8*O18
		dexcess_err = np.sqrt(H2_err**2+O18_err**2)
		yi = df.loc[transects[i]]["column height (cm)"].values
		yi_err = df.loc[transects[i]]["s.d. height"].values

		#d18O data plotted on right hand side, by default blue points with errorbars.
		ax_O18.plot(O18,yi,'o',label= i)
		ax_O18.errorbar(O18,yi,yerr =  yi_err,xerr= O18_err,
			marker = '.',
			markersize = 0,
			lw = 0,
			elinewidth = 1,
			ecolor = 'black')
		ax_O18.set_xlim(-14,-7)

		#d2H data plotted on leftt hand side, changed to red points with black error bars.
		ax_H2.plot(H2,yi,'o',color='firebrick',label= i)
		ax_H2.errorbar(H2,yi,yerr =  yi_err,xerr= H2_err,
			marker = '.',
			markersize = 0,
			lw = 0,
			elinewidth = 1,
			ecolor = 'black')

		ax_H2.set_xlim(-100,-50)

		ax_dexcess.plot(dexcess,yi,'o',color='black',label= i)
		ax_dexcess.errorbar(dexcess,yi,yerr =  yi_err,xerr= dexcess_err,
			marker = '.',
			markersize = 0,
			lw = 0,
			elinewidth = 1,
			ecolor = 'black')

		ax_dexcess.set_xlim(8,16)

		#printing the layer boundaries
		layer_bounds = get_layers(df,transects[i])
		
		#for each plot, add the layers in dotted lines and fill in the layers depending on 
		#presence/absence of inclusions.

		for axes in (ax_H2,ax_O18,ax_dexcess):

			l0=axes.get_ylim()[0]
			for l in layer_bounds:
			
				axes.axhline(y=l[0], color = "grey",lw=0.45)
				if l[1]==0:
					axes.add_patch(Rectangle((axes.get_xlim()[0],l[0]),
						width = axes.get_xlim()[1]-axes.get_xlim()[0],
						height =l0-l[0], 
						fill =True,
						facecolor = 'white',
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
					facecolor = 'white',
					alpha = 0.75))
			else:
				axes.add_patch(Rectangle((axes.get_xlim()[0],axes.get_ylim()[1]),
					width = axes.get_xlim()[1]-axes.get_xlim()[0],
					height =l0-axes.get_ylim()[1], 
					fill =True,
					facecolor = 'lightblue',
					alpha = 0.75))


	ax_dexcess.set(xlabel = 'd-excess [‰] (VSMOW)')
	ax_O18.set(xlabel = '$\delta^{18}$O [‰] (VSMOW)')
	ax_H2.set(ylabel = 'height (cm)', xlabel = '$\delta^{2}$H [‰] (VSMOW)')

	
	plt.tight_layout()
	plt.savefig(filename,dpi=300)
	plt.show()

def figureIsotopes(df,transects,filename='../fig/isotope_crossplot_sample.pdf'):
	
	fig, ax = plt.subplots(figsize = (8,5))

	plotIsotopes(df,transects,ax= ax)

	ax.set(ylabel = '$\delta^{2}$H [‰] (VSMOW)', xlabel = '$\delta^{18}$O [‰] (VSMOW)')
	ax.legend()
	plt.savefig(filename, dpi = 300)

	plt.show()

def plotIsotopes(df,transects,ax=None,colors=None):
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
	if colors is not None:
		for i,colour in zip(transects,colors):

			yi = df.loc[transects[i]]["d2H"].values
			xi_err = df.loc[transects[i]]["s.d. d18O"].values
			yi_err = df.loc[transects[i]]["s.d. d2H"].values
			xi = df.loc[transects[i]]["d18O"].values

			ax.plot(xi,yi,'.',label= i,color=colour)
			ax.errorbar(xi,yi,yerr =  yi_err,xerr= xi_err,
			marker = '.',
			markersize = 0,
			lw = 0,
			elinewidth = 1,
			ecolor = 'black')

		return ax
	else:
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

		return ax

def full_plotter(df_ISO,transects,df_GNIP,site,filename="../fig/isotopes/combined_ice_GNIP_figure.pdf",df_old=None,loc='right'):
	#plotting the data relative to a 
	fig,ax = plt.subplots(figsize=(8,5))

	ax1 = plotIsotopes(df_ISO,transects,ax = ax)
	if df_old is not None:
		ax3= ax.twinx()
		ax3 = boxplot_old(df_old)
		ax3.set_ylim(0,10)
		ax3.tick_params(direction=None)
		ax3.get_yaxis().set_visible(False)
		ax3.text(-10.25,1.5,'Old Hundsalm $\delta^{18}O$ transect')




	ax.set_xlim(ax1.get_xlim())
	ax.set_ylim(ax1.get_ylim())
	ax2 = gnip.plot_LMWL(df_GNIP,site,ax = ax,option='grey')

	ax.set(ylabel = '$\delta^{2}$H [‰] (VSMOW)', xlabel = '$\delta^{18}$O [‰] (VSMOW)')
	ax.tick_params(direction='in', top=True,right=True)
	ax.legend(loc=loc)
	plt.tight_layout()
	plt.savefig(filename,dpi= 300)
	plt.show()


def IsoLSRPlotter(df_ISO,transects,df_GNIP,site,ax=None,color=None):

	if color is None:
		ax1 = plotIsotopes(df_ISO,transects,ax = ax,colors=["teal","firebrick","grey"])
	else:
		ax1 = plotIsotopes(df_ISO,transects,ax = ax,colors=[color])

	ax.set_xlim(-15,-5)
	ax.set_ylim(-100,-30)

	ylim=ax1.get_ylim()
	xlim=ax1.get_xlim()

	posy = np.arange(ylim[0],ylim[0]+(ylim[1]-ylim[0])/2,(ylim[1]-ylim[0])/6)+4
	posx = np.ones(3)*xlim[1]-(xlim[1]-xlim[0])/(2.5)
	if color is None:
		for i,c,posx,posy in zip(transects,['teal','firebrick','grey'],posx,posy):

			myODR(ax,df_ISO,transects[i],c,(posx,posy))
	else:
		for i,posx,posy in zip(transects,posx,posy):
			myODR(ax,df_ISO,transects[i],color,(posx,posy))

	ax2 = gnip.plot_LMWL(df_GNIP,site,ax = ax,option='grey')

	ax.set(ylabel = '$\delta^{2}$H [‰] (VSMOW)', xlabel = '$\delta^{18}$O [‰] (VSMOW)')
	ax.tick_params(direction='in')

	return ax

from collections import OrderedDict

def OLSR(ax,pd,transect,colour,posxy):
    xi=np.arange(-50,20)
    slope, intercept, r_value, p_value, std_err = stats.linregress(pd.loc[transect]["d18O"],pd.loc[transect]["d2H"])
    ax.plot(xi,slope*xi+intercept,color=colour,linewidth=0.5)
    
    ax.text(posxy[0],posxy[1],"y={:.2f}x {:+.2f}\n $R^2={:.2f}$".format(slope,intercept,r_value),color=colour)
    return ax

def get_stats(df_ISO,transect):
	values={}
	for t in transect:
		O18 = df_ISO.loc[transect[t]]["d18O"]
		H2 = df_ISO.loc[transect[t]]["d2H"]
		d_ex = H2-8*O18
	
		for i,j in zip((O18,H2,d_ex),('$\delta^{18}$O','$\delta^{2}$H','d-excess')):
			values[j+' '+t]=OrderedDict({'mini':i.min(),'median':i.median(),'maxi':i.max(),'mean':i.mean(),'std.':i.std()})
	df= pd.DataFrame(values)
	return df
	
def get_stats_latex(df_ISO,transect):

	df= get_stats(df_ISO,transect)

	return df.to_latex(index=False)

def boxplot_old(df):
	bp = df.boxplot(column=["d18O"],vert= False,return_type='both',labels=['Old Data'])

	[item.set_color('firebrick') for item in bp[1]['medians']]
	[item.set_color('black') for item in bp[1]['boxes']]
	[item.set_color('black') for item in bp[1]['whiskers']]

	ax = bp[0]

	return ax

def transect_boxplot(df,transects,ax_column):  # where ax_column is a dictionary of column name and axes
	
	axes = tuple([ax_column[i] for i in ax_column])
	keys = [i for i in ax_column]
	fig, axes = plt.subplots(figsize=(4,6))

	temp_axes = []
	for AX,col in zip([ax_column[i] for i in ax_column],keys):
		ax = df.loc[transects].boxplot(by='transect',column= [col],ax = AX,return_type='dict')
		temp_axes.append(ax)
		if col == 'd18O':
			ax.set_ylim(-14,-7)
			ax.set_xlabel("")
			ax.set_title("")
			ax.set_ylabel('$\delta^{18}$O [‰] (VSMOW)')

		else:
			if col == 'd2H':
				ax.set_ylim(-100,-50)
				ax.set_xlabel("")
				ax.set_title("")
				ax.set_ylabel('$\delta^{2}$H [‰] (VSMOW)')

	for bp in temp_axes:
		[[item.set_color('firebrick') for item in bp[key]['medians']] for key in bp.keys()]
		[[item.set_color('black') for item in bp[key]['boxes']] for key in bp.keys()]
		[[item.set_color('black') for item in bp[key]['whiskers']] for key in bp.keys()]

	plt.suptitle("")
	plt.title("")
	plt.show()

import scipy.odr as odr

def myODR(ax,pd,transect,colour,posxy):
	xi=pd.loc[transect]["d18O"]
	yi=pd.loc[transect]["d2H"]
	x_low=np.arange(-30,min(xi)-0.1,0.1)
	x_mid=np.arange(min(xi)-1,max(xi)+0.1,0.1)
	x_high = np.arange(max(xi)+0.1,0,0.1)

	def f(B,x):
		return B[0]*x + B[1]

	linear = odr.Model(f)
	mydata = odr.Data(xi, yi, wd=1./xi.std()**2, we=1./yi.std()**2)
	myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
	myoutput = myodr.run()

	slope = myoutput.beta[0]
	sd_slope = myoutput.sd_beta[0]
	intercept = myoutput.beta[1]
	sd_intercept = myoutput.sd_beta[1]
	def f(x):
		y=x*slope+intercept
		return y
	print(posxy)
	ax.plot(x_low,f(x_low),color=colour,linewidth=0.6)
	ax.plot(x_high,f(x_high),color=colour,linewidth=0.6)
	ax.plot(x_mid,f(x_mid),'--',color=colour,linewidth=0.3)

	ax.text(posxy[0],posxy[1],"y={:.2f} ($\pm${:.2f}) x {:+.2f} ($\pm${:.2f})".format(slope,sd_slope,intercept,sd_intercept),color=colour)

	return ax

