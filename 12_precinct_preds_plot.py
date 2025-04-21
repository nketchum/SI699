#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# President: 2016 (Trump), 2020 (Biden), 2024 (Trump)
# U.S. Senate: 2014 (Peters), 2018 (Stabenow), 2020 (Peters), 2024 (Slotkin)
# U.S. House: every cycle
# State Senate: 2014, 2018, 2022
# State House: every cycle

ELECTIONS = {}

ELECTIONS['U.S. House'] =   ['2024']
ELECTIONS['State House'] =  ['2024']
ELECTIONS['President'] =    ['2024']

TARGETS = ['partisan_temp']

TOP_N_FEATURES = 20
FEATURES_ALREADY_RANKED = True


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# In[ ]:


# https://stackoverflow.com/questions/66171467/how-to-get-all-color-codes-from-linearsegmentedcolormap
# https://discourse.matplotlib.org/t/register-colormap-collection/19501

def makeSmoothPartisanColormap(n_colors=99):
    assert n_colors % 2 == 1, "Use an odd number for a center gray"
    half = n_colors // 2

    # Blue side (-1)
    reds_neg = np.linspace(0, 127, half, dtype=int)
    greens_neg = np.linspace(0, 127, half, dtype=int)
    blues_neg = np.linspace(255, 127, half, dtype=int)
    blue_half = [(r, g, b) for r, g, b in zip(reds_neg, greens_neg, blues_neg)]

    # Center gray
    middle = [(128, 128, 128)]

    # Red side (+1)
    reds_pos = np.linspace(129, 255, half, dtype=int)
    greens_pos = np.linspace(129, 0, half, dtype=int)
    blues_pos = np.linspace(129, 0, half, dtype=int)
    red_half = [(r, g, b) for r, g, b in zip(reds_pos, greens_pos, blues_pos)]

    full_rgb = blue_half + middle + red_half
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in full_rgb]
    return LinearSegmentedColormap.from_list("smooth_partisan", hex_colors, N=256)

def getCenteredNorm(df, column, soft_clip=0.3):
    return TwoSlopeNorm(vmin=-soft_clip, vcenter=0, vmax=soft_clip)


# In[ ]:


def plotPrecinctBounds(df, target, year, office, is_pred=True):

    if is_pred == True:
        plot_column = 'predicted_label'
    else:
        plot_column = 'true_label'

    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)

    df.boundary.plot(ax=ax, color="black", linewidth=0.1)

    cmap = makeSmoothPartisanColormap()
    norm = getCenteredNorm(df, plot_column, soft_clip=0.5)
    df.plot(ax=ax, column=plot_column, cmap=cmap, norm=norm, edgecolor="black", linewidth=0.01)

    ax.margins(0)
    ax.set_title('Precinct Preds', fontsize=64)
    ax.set_axis_off()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if is_pred:
        filename = f'output/maps/precincts/{target}_{year}_{office.replace(" ", "_").replace(".", "")}_Prediction_Map.png'
    else:
        filename = f'output/maps/precincts/{target}_{year}_{office.replace(" ", "_").replace(".", "")}_True_Map.png'
    
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for target in TARGETS:
    print(f'Processing target {target}...')

    for key, value in ELECTIONS.items():
        print(f'Num. of years to process: {len(value)}')

        OFFICES = [key]
        YEARS = value
    
        for year in YEARS:
            for office in OFFICES:

                df_precinct_outcomes = gpd.read_file(f'data/generated_data/df_02_vote_changes_calc_{year}_{office.replace(' ', '_').replace('.', '')}.geojson', driver='GeoJSON')
                df_precinct_outcomes['standardized_id_num'] = df_precinct_outcomes['standardized_id_num'].astype(str).str.zfill(13)

                df_preds = pd.read_csv(f'data/generated_data/predicted_{target}_holdout.csv')
                df_preds['standardized_id_num'] = df_preds['standardized_id_num'].astype(int).astype(str).str.zfill(13)
        
                df_precinct_pred = df_precinct_outcomes.merge(df_preds, on="standardized_id_num", how="inner")

                merged_cols = df_precinct_pred.columns
                if f"{target}_x" in merged_cols and f"{target}_y" in merged_cols:
                    df_precinct_pred.drop(columns=[f"{target}_y"], inplace=True)
                    df_precinct_pred.rename(columns={f"{target}_x": target}, inplace=True)
                
                # Sanity check
                if target not in df_precinct_pred.columns:
                    raise ValueError(f"Target '{target}' not found in df_precinct_pred after merge for {year} {office}.")
                
                plotPrecinctBounds(df_precinct_pred, target, year, office, True) # predictions
                plotPrecinctBounds(df_precinct_pred, target, year, office, False) # truth
        
        print('DONE')


# In[ ]:




