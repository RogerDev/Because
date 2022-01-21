# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
 
def plot(distDict):
    # Make a data frame
    yvals = []
    xmax = 0
    vars = list(distDict.keys())
    vars.sort()
    numVars = len(vars) - 1 # Don't count the _x_ variable
    for key in distDict.keys():
        if key == '_x_':
            dat = distDict[key]
            xmax = max(dat)
            xmin = min(dat)
            continue
        vals = distDict[key]
        #print('vals = ', vals)
        yvals += list(vals)

    ymax = max(yvals)
    ymin = 0

    cols = 3
    rows = math.ceil(numVars / cols)
 
    df = pd.DataFrame(distDict)
    
    # Initialize the figure
    plt.style.use('seaborn-darkgrid')
   
    # create a color palette
    palette = plt.get_cmap('Set1')
    fig, axs = plt.subplots(rows,cols)
    for ax in axs.flat:
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
     # multiple line plot
    num=0
    for var in df.drop('_x_', axis=1):

        #num+=1
    
        # Find the right spot on the plot
        ax = axs.flat[num]
        # Plot the lineplot
        ax.plot(df['_x_'], df[var], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=var)
        
        # Not ticks everywhere
        #if num in range(7) :
        #    plt.tick_params(labelbottom='off')
        #if num not in [1,4,7] :
        #    plt.tick_params(labelleft='off')
    
        # Add title
        ax.set_title(var, loc='left', fontsize=12, fontweight=0, color=palette(num))
        num += 1
    # general title
    plt.suptitle("PDFs of all variables", fontsize=13, fontweight=0, color='black', style='italic', verticalalignment='top', horizontalalignment='center', y=.98)
    vcenter = math.floor(rows / 2)
    hcenter = math.floor(cols / 2)
    if rows > 1:
        axs[rows-1, hcenter].set_xlabel('x', labelpad = 15, ha='center', va='bottom', fontsize=13, fontweight='bold')
        axs[vcenter, 0].set_ylabel('P(X=x)', labelpad = 15, ha='left', va='center', rotation='vertical', fontsize = 13, fontweight='bold')
    else:
        axs[hcenter].set_xlabel('x', labelpad = 15, ha='center', va='bottom', fontsize=13, fontweight='bold')
        axs[0].set_ylabel('P(X=x)', labelpad = 15, ha='left', va='center', rotation='vertical', fontsize = 13, fontweight='bold')
    for axis in axs.flat:
        # Same limits for everybody!
        axis.label_outer()
    # Axis title
    plt.subplots_adjust(hspace=.5)
    plt.show()