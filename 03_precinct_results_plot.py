#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# President: 2016 (Trump), 2020 (Biden), 2024 (Trump)
# U.S. Senate: 2014 (Peters), 2018 (Stabenow), 2020 (Peters), 2024 (Slotkin)
# U.S. House: every cycle
# State Senate: 2014, 2018, 2022
# State House: every cycle

ELECTIONS = {}

ELECTIONS['U.S. House'] =   ['2014', '2016', '2018', '2020', '2022', '2024']
ELECTIONS['State House'] =  ['2014', '2016', '2018', '2020', '2022', '2024']
ELECTIONS['U.S. Senate'] =  ['2014', '2018', '2020', '2024']
ELECTIONS['State Senate'] = ['2014', '2018', '2022']
ELECTIONS['President'] =    ['2016', '2020', '2024']


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# ### Plots unseen data
# Uses vote shares from current election outcome, which is looking into the future.

# In[ ]:


# https://medium.com/%40sinhaaa003/transforming-images-rgb-to-grayscale-and-gamma-correction-with-python-fe5a0afa12b9
# https://stackoverflow.com/questions/73888380/how-to-perform-weighted-sum-of-colors-in-python
# https://bioimagebook.github.io/chapters/1-concepts/4-colors/python.html

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import to_rgba

def plotPrecinctBounds(df, year, office):
    color_map = {
        'D': np.array([0, 0, 255]),   # Blue
        'R': np.array([255, 0, 0]),   # Red
        'I': np.array([255, 255, 0])  # Yellow
    }

    # Weighted sum of RGB components
    def computeMixedColor(row):       
        mixed_rgb = (
            row['dem_share'] * color_map['D'] +
            row['rep_share'] * color_map['R'] +
            row['oth_share'] * color_map['I']
        )
        return tuple(mixed_rgb.astype(int) / 255)

    df['color'] = df.apply(computeMixedColor, axis=1)

    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)

    df.boundary.plot(ax=ax, color="black", linewidth=0.1)
    df.plot(ax=ax, color=df['color'], edgecolor="black", linewidth=0.01)

    ax.margins(0)
    ax.set_title('Precinct Outcomes', fontsize=64)
    ax.set_axis_off()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('output/maps/precincts/' + str(year) + "_" + office.replace('.', '').replace(' ', '_') + "_Map.png")
    
    plt.close(fig)


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    for year in YEARS:
        print(f'Processing year {year}...')
        
        for office in OFFICES:
            print(f'Processing office {year}...')

            df_precinct_outcomes = gpd.read_file('data/generated_data/df_02_vote_changes_calc_' + str(year) + '_' + key.replace('.', '').replace(' ', '_') + '.geojson', driver='GeoJSON')
            plotPrecinctBounds(df_precinct_outcomes, year, office)


# In[ ]:




