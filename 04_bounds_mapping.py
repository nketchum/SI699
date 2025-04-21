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


BOUND_TYPES = ['census_tract', 'school_district', 'zipcode', 'urban_area']


# In[ ]:


from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import RandomForestRegressor
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


pd.set_option("display.max_columns", None)


# In[ ]:


# Valid bound_type values:
# - school_district
# - census_tract
# - zipcode
# - urban_area
#
# sjoin_nearest_column:
# - the column on which to join the bound data and the precinct outcomes data.

def loadBounds(bound_type, year):
    if bound_type == 'school_district':
        df_bounds = gpd.read_file('data/school_districts/School_District.geojson')
        sjoin_nearest_column = 'OBJECTID'
    elif bound_type == 'census_tract':
        if year == '2024': # No 2024 tract data, use 2023.
            year = '2023'
        df_bounds = gpd.read_file('data/census/tracts/cb_' + year + '_26_tract_500k/cb_' + year + '_26_tract_500k.shp')
        sjoin_nearest_column = 'GEOID'
    elif bound_type == 'zipcode':
        df_bounds = gpd.read_file('data/OpenDataDE/mi_michigan_zip_codes_geo.min.json')
        sjoin_nearest_column = 'ZCTA5CE10'
    elif bound_type == 'urban_area':
        if int(year) >= 2020:
            directory = 'HPA_V5'
            filename = 'HPA_V5.shp'
        else:
            directory = 'HPA_v4'
            filename = 'HPA_2010_V4.shp'
        df_bounds = gpd.read_file(f'data/USDOT/{directory}/{filename}')
        df_bounds = df_bounds[df_bounds['HPA_NAME'].str.contains(', MI')]
        df_bounds['id'] = df_bounds.index
        sjoin_nearest_column = 'id'
    else:
        raise Exception('Invalid bound type.')
    
    return df_bounds, sjoin_nearest_column


# In[ ]:


def loadPrecinctOutcomes(year, office):
    df_precinct_outcomes = gpd.read_file("data/generated_data/df_02_vote_changes_calc_" + str(year) + "_" + office.replace('.', '').replace(' ', '_') + ".geojson")
    return df_precinct_outcomes


# In[ ]:


# https://stackoverflow.com/questions/75699024/finding-the-centroid-of-a-polygon-in-python
# https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.centroid.html
# https://medium.com/towards-data-science/geospatial-data-analysis-with-geopandas-876cb72721cb

def plotCentroids(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office):
    # Centroids for each GeoDataFrame
    df_bounds["bound_centroid"] = df_bounds.geometry.representative_point()
    df_precinct_outcomes["precinct_centroid"] = df_precinct_outcomes.geometry.representative_point()
    
    # Temp GeoDataFrames with centroids as the active geometry
    bound_centroids = df_bounds.set_geometry("bound_centroid")
    precinct_centroids = df_precinct_outcomes.set_geometry("precinct_centroid")

    # Nearest spatial join
    # https://docs.coiled.io/blog/spatial-join-dask-geopandas-sjoin.html
    # https://automating-gis-processes.github.io/CSC18/lessons/L4/spatial-join.html
    # https://geopandas.org/en/stable/gallery/spatial_joins.html
    joined = gpd.sjoin_nearest(
        precinct_centroids,
        bound_centroids[[sjoin_nearest_column, "bound_centroid"]],
        how="left",
        distance_col="dist"  # Optional: includes the distance in the output
    )

    nearest_bound_col_name = "nearest_bound_" + bound_type
    
    # Add school district id to precinct dataframe.
    df_precinct_outcomes[nearest_bound_col_name] = joined[sjoin_nearest_column]
    
    # Reset the geometry if needed.
    df_precinct_outcomes = df_precinct_outcomes.set_geometry("geometry")
    
    # Group precincts by the nearest school district identifier
    counts = df_precinct_outcomes.groupby(nearest_bound_col_name).size().reset_index(name="count")
    
    # Merge counts into school districts via sjoin_nearest_column
    df_bounds = df_bounds.merge(
        counts, 
        left_on=sjoin_nearest_column, 
        right_on=nearest_bound_col_name, 
        how="left"
    )
    # Zero-fill missing counts for districts with no associated precincts
    df_bounds["count"].fillna(0, inplace=True)
    
    # Compute school district centroids
    bound_centroids = df_bounds.geometry.representative_point()
    
    # Compute precinct centroids
    precinct_centroids = df_precinct_outcomes.geometry.representative_point()
    
    # Set a scaling factor for the marker sizes (adjust as needed)
    scaling = 50
    sizes = df_bounds["count"] * scaling
    
    # Plot everything
    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)
    
    # Plot precinct boundaries
    df_precinct_outcomes.boundary.plot(ax=ax, color="green", linewidth=0.1, zorder=1)
    df_precinct_outcomes.plot(ax=ax, color="white", edgecolor="green", linewidth=0.01, zorder=2)
    
    # Plot school district boundaries
    df_bounds.boundary.plot(ax=ax, color="orange", linewidth=0.1, zorder=1)
    df_bounds.plot(ax=ax, color="white", edgecolor="orange", linewidth=0.5, zorder=3)
    
    # Plot school district centroids with sized proportional to the precinct count
    ax.scatter(bound_centroids.x, bound_centroids.y, 
               marker='o', color='orange', s=sizes, zorder=4)
    
    # Plot school district precise centroids
    ax.scatter(bound_centroids.x, bound_centroids.y, 
               marker='o', color='red', zorder=4, s=25)
    
    # Plot precinct centroids
    ax.scatter(precinct_centroids.x, precinct_centroids.y, marker='o', color='green', s=10, zorder=5)
    
    ax.margins(0)
    ax.set_title("Bounds with Precincts", fontsize=64)
    ax.set_axis_off()
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("output/maps/" + bound_type + "/centroid/" + str(year) + "_Bound_Centroids_" + office.replace('.', '').replace(' ', '_') + "_Map.png")
    
    plt.close(fig)

    return df_bounds, df_precinct_outcomes


# ### Plots unseen data
# Uses vote shares from current election outcome, which is looking into the future.

# In[ ]:


def plotResults(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office):
    nearest_bound_col_name = "nearest_bound_" + bound_type
    
    # Aggregate precinct results by school district.
    aggregated = df_precinct_outcomes.groupby(nearest_bound_col_name).agg(
        dem_votes=('dem_votes', 'sum'),
        rep_votes=('rep_votes', 'sum'),
        oth_votes=('oth_votes', 'sum'),
        dem_share=('dem_share', 'mean'),
        rep_share=('rep_share', 'mean'),
        oth_share=('oth_share', 'mean')
    ).reset_index()
    
    # Merge the aggregated outcomes into school districts datafra,e.
    df_bounds_with_outcomes = df_bounds.merge(
        aggregated,
        left_on=sjoin_nearest_column,
        right_on=nearest_bound_col_name,
        how="left"
    )
    
    df_plot_bounds_outcome_agg = df_bounds_with_outcomes.copy()

    # https://medium.com/%40sinhaaa003/transforming-images-rgb-to-grayscale-and-gamma-correction-with-python-fe5a0afa12b9
    # https://stackoverflow.com/questions/73888380/how-to-perform-weighted-sum-of-colors-in-python
    # https://bioimagebook.github.io/chapters/1-concepts/4-colors/python.html
    
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
    
    df_plot_bounds_outcome_agg['color'] = df_plot_bounds_outcome_agg.apply(computeMixedColor, axis=1)
    
    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)
    
    df_plot_bounds_outcome_agg.boundary.plot(ax=ax, color="black", linewidth=0.1)
    df_plot_bounds_outcome_agg.plot(ax=ax, color=df_plot_bounds_outcome_agg['color'], edgecolor="black", linewidth=0.01)
    
    ax.margins(0)
    ax.set_title("Precinct Results", fontsize=64)
    ax.set_axis_off()
    
    ax.margins(0)
    ax.set_title("Precinct Results", fontsize=64)
    ax.set_axis_off()
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("output/maps/" + bound_type + "/" + str(year) + "_Results_" + office.replace('.', '').replace(' ', '_') + "_Map.png")
    
    plt.close(fig)

    return df_bounds, df_precinct_outcomes


# ### Plots unseen data
# Uses vote shares from current election outcome, which is looking into the future.

# In[ ]:


def plotResultsSynthetic(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office):
    if "bound_centroid" not in df_bounds.columns:
        df_bounds["bound_centroid"] = df_bounds.geometry.centroid
    
    if "precinct_centroid" not in df_precinct_outcomes.columns:
        df_precinct_outcomes["precinct_centroid"] = df_precinct_outcomes.geometry.centroid
    
    bound_centroids = df_bounds.set_geometry("bound_centroid")
    precinct_centroids = df_precinct_outcomes.set_geometry("precinct_centroid")
    
    joined = gpd.sjoin_nearest(
        precinct_centroids,
        bound_centroids[[sjoin_nearest_column, 'bound_centroid']],
        how="left",
        distance_col="dist"
    )

    nearest_bound_col_name = "nearest_bound_" + bound_type
    df_precinct_outcomes[nearest_bound_col_name] = joined[sjoin_nearest_column]
    
    df_precinct_outcomes = df_precinct_outcomes.set_geometry("geometry")
    
    aggregated = df_precinct_outcomes.groupby(nearest_bound_col_name).agg(
        precinct_count=(nearest_bound_col_name, 'size'),
        dem_votes=('dem_votes', 'sum'),
        rep_votes=('rep_votes', 'sum'),
        oth_votes=('oth_votes', 'sum'),
        dem_share=('dem_share', 'mean'),
        rep_share=('rep_share', 'mean'),
        oth_share=('oth_share', 'mean')
    ).reset_index()
    
    df_bounds_merged = df_bounds.merge(
        aggregated,
        left_on=sjoin_nearest_column,
        right_on=nearest_bound_col_name,
        how="left"
    )
    
    df_bounds_merged["precinct_count"] = df_bounds_merged["precinct_count"].fillna(0)
    
    # For bounds with missing aggregated results, we’ll predict synthetic values.
    df_bounds_merged["centroid_lat"] = df_bounds_merged["bound_centroid"].y
    df_bounds_merged["centroid_lon"] = df_bounds_merged["bound_centroid"].x
    
    # Define features to use for the imputation model.
    features = ["centroid_lat", "centroid_lon", sjoin_nearest_column]  # Adjust "ALAND10" as needed
    
    # List of target variables to synthesize.
    target_vars = ["dem_votes", "rep_votes", "oth_votes", "dem_share", "rep_share", "oth_share"]
    for target in target_vars:
        train_df = df_bounds_merged.dropna(subset=[target])
        missing_df = df_bounds_merged[df_bounds_merged[target].isna()]
        if not missing_df.empty and not train_df.empty:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(train_df[features], train_df[target])
            predicted = model.predict(missing_df[features])
            df_bounds_merged.loc[missing_df.index, target] = predicted

    # https://medium.com/%40sinhaaa003/transforming-images-rgb-to-grayscale-and-gamma-correction-with-python-fe5a0afa12b9
    # https://stackoverflow.com/questions/73888380/how-to-perform-weighted-sum-of-colors-in-python
    # https://bioimagebook.github.io/chapters/1-concepts/4-colors/python.html
    color_map = {
        'D': np.array([0, 0, 255]),
        'R': np.array([255, 0, 0]),
        'I': np.array([255, 255, 0])
    }
    
    def computeMixedColor(row):
        # If any vote share is missing, return white.
        if pd.isna(row['dem_share']) or pd.isna(row['rep_share']) or pd.isna(row['oth_share']):
            return (1.0, 1.0, 1.0)
        mixed_rgb = (row['dem_share'] * color_map['D'] +
                     row['rep_share'] * color_map['R'] +
                     row['oth_share'] * color_map['I'])
        return tuple((mixed_rgb.astype(int) / 255))
    
    df_bounds_merged["color"] = df_bounds_merged.apply(computeMixedColor, axis=1)
    
    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)
    
    df_bounds_merged.plot(ax=ax, color=df_bounds_merged["color"], edgecolor="black", linewidth=0.01)
    
    ax.margins(0.05)
    ax.set_title("Precinct Results (Synthetic Filled)", fontsize=64)
    ax.set_axis_off()
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("output/maps/" + bound_type + "/" + str(year) + "_Results_" + office.replace('.', '').replace(' ', '_') + "_Sythentic_Map.png")
    
    plt.close(fig)
    
    return df_bounds, df_precinct_outcomes


# In[ ]:


def saveResults(df_precinct_outcomes, bound_type, year, office):
    nearest_bound_col_name = "nearest_bound_" + bound_type
    
    df_map_precinct_nearest_bound = df_precinct_outcomes[['standardized_id_num', nearest_bound_col_name]]
    df_map_precinct_nearest_bound = df_map_precinct_nearest_bound.drop_duplicates(subset=['standardized_id_num'], keep='first')
    
    df_map_precinct_nearest_bound.to_csv('data/generated_data/df_04_bound_nearest_' + bound_type + '_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.csv', index=False)


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')
    
    for year in YEARS:
        print(f'Processing year {year}')
        
        for office in OFFICES:
            print(f'Processing office {office}')
            
            for bound_type in BOUND_TYPES:
                print(f'Processing bound type {bound_type}')
                
                # Join bounds and precinct outcomes
                df_bounds, sjoin_nearest_column = loadBounds(bound_type, year)
                df_precinct_outcomes = loadPrecinctOutcomes(year, office)
    
                # Plot both centroids and choropleths
                df_bounds, df_precinct_outcomes = plotCentroids(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office)
                df_bounds, df_precinct_outcomes = plotResults(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office)
    
                # Some bound types leave holes to be imputed.
                if (bound_type == 'census_tract') | (bound_type == 'zipcode'):
                    print(f'Imputing missing data')
                    df_bounds, df_precinct_outcomes = plotResultsSynthetic(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office)
    
                print('Saving results')
                saveResults(df_precinct_outcomes, bound_type, year, office)
                
                print('............................')
            print('----------------------------')
        print('============================')
    print('############################')

print('DONE')


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    for year in YEARS:
        for office in OFFICES:
            print(f'Processing office {office}')
            
            for bound_type in BOUND_TYPES:
                print(f'Processing bound type {bound_type}')
                
                # Join bounds and precinct outcomes
                df_bounds, sjoin_nearest_column = loadBounds(bound_type, year)
                df_precinct_outcomes = loadPrecinctOutcomes(year, office)
    
                # Plot both centroids and choropleths
                df_bounds, df_precinct_outcomes = plotCentroids(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office)
                df_bounds, df_precinct_outcomes = plotResults(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office)
    
                # Some bound types leave holes to be imputed.
                if (bound_type == 'census_tract') | (bound_type == 'zipcode'):
                    print(f'Imputing missing data')
                    df_bounds, df_precinct_outcomes = plotResultsSynthetic(df_bounds, df_precinct_outcomes, bound_type, sjoin_nearest_column, year, office)
    
                print('Saving results')
                saveResults(df_precinct_outcomes, bound_type, year, office)

                print('............................')
            print('----------------------------')
        print('============================')
    print('############################')

print('DONE')


# In[ ]:




