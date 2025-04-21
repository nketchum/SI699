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
ELECTIONS['U.S. Senate'] =  ['2024']
ELECTIONS['President'] =    ['2024']

TARGET = 'partisan_temp'
# TARGET = 'partisanship_lean_curr'

LLM_ACTIVE = True


# In[ ]:


from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist, squareform
from shapely import wkt
from shapely.geometry import box, Point
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import glob
import math
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numbers
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


# ### Analysis & Reprot
# Create an in-depth analysis, imagery, and a full written report containing regional voting personas characterizing geographic clusters across the state that are geographically proximate, show similar socioeconomic census data, and share some degree of similarity in their electoral history.

# #### OpenAI API Key

# In[ ]:


from openai import OpenAI

file = open("OpenAI.key", "r")
key = file.read()

OPENAI_API_KEY = key
client = OpenAI(api_key=OPENAI_API_KEY)

pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)


# #### Config

# In[ ]:


census_datasets = [
    'b02001_race', 'b04007_ancestry', 'b05012_nativity_us', 'b08303_travel_time_work', 
    'b25003_housing_rentership', 'dp04_housing_characteristics', 'dp05_age_race', 's0101_age_sex', 
    's1101_households_families', 's1201_marital_status', 's1501_educational_attainment', 's1701_income_poverty', 
    's1903_median_income', 's2101_veteran_status', 's2201_food_stamps', 's2301_employment_status', 
    's2401_occupation_sex', 's2403_industry_sex', 's2501_occupancy_characteristics', 
    's2503_financial_characteristics', 's2701_health_insurance',
]


# ### Primary Data Loading

# In[ ]:


def loadPrecinctData(year, office):
    office_filename = office.replace(".", "").replace(" ", "_")
    print("Loading precinct data...")
    precinct_data = pd.read_csv(f'data/generated_data/df_06_tract_{year}_{office_filename}.csv')
    precinct_data['standardized_id_num'] = precinct_data['standardized_id_num'].astype(str).str.zfill(13)

    predictions = pd.read_csv(f'data/generated_data/predicted_{TARGET}_{year}_holdout.csv')
    predictions['standardized_id_num'] = predictions['standardized_id_num'].astype(str).str.zfill(13)
    precinct_data = precinct_data.merge(predictions, on='standardized_id_num', how='left')
    return predictions, precinct_data


def loadCensusData(census_datasets, year, office):
    print("Loading census data...")
    census_dataset_dfs = {}
    office_filename = office.replace(".", "").replace(" ", "_")
    
    for census_dataset in census_datasets:
        census_dataset = census_dataset.lower()
        if census_dataset.startswith('s'):
            census_dataset_code = census_dataset[:5].upper()
            census_dataset_label = census_dataset[6:]
            data_type = 'ACSST5Y'
        elif census_dataset.startswith('b'):
            census_dataset_code = census_dataset[:6].upper()
            census_dataset_label = census_dataset[7:]
            data_type = 'ACSDT5Y'
        elif census_dataset.startswith('d'):
            census_dataset_code = census_dataset[:4].upper()
            census_dataset_label = census_dataset[5:]
            data_type = 'ACSDP5Y'
        
        df = pd.read_csv(f'data/generated_data/df_06_{census_dataset_label}_{year}_{office_filename}.csv')
        df.rename(columns={f'geoid_{census_dataset_label}': 'geoidfq_tract'}, inplace=True)
        census_dataset_dfs[census_dataset_label] = df

    return census_dataset_dfs


def mergeDatasets(precinct_data, census_dataset_dfs):
    print("Merging datasets...")
    merged_df = precinct_data.copy()
    
    for key, df_census in census_dataset_dfs.items():
        df_census_copy = df_census.copy()
        df_census_copy.rename(columns={f'geoid_{key}': 'geoidfq_tract'}, inplace=True)
        merged_df = merged_df.merge(df_census_copy, on='geoidfq_tract', how='left')
    
    print(f"Merged dataset has {merged_df.shape[1]} columns")
    return merged_df


# ### Helper Functions

# In[ ]:


def analyzeVotingPatterns(merged_df):
    print("Analyzing voting patterns by cluster...")
    
    if 'partisanship' not in merged_df.columns:
        try:
            merged_df['partisanship'] = merged_df.apply(categorizePartisanship, axis=1)
        except Exception as e:
            print(f"Could not compute partisanship: {e}")
    
    if 'partisanship_change_prev' not in merged_df.columns:
        try:
            merged_df['partisanship_change_prev'] = merged_df.apply(categorizePartisanChange, axis=1)
        except Exception as e:
            print(f"Could not compute partisanship_change_prev: {e}")
    
    def safe_mode(x):
        try:
            return x.mode().iloc[0] if not x.empty else 'unknown'
        except:
            return 'unknown'
    
    agg_dict = {'standardized_id_num': 'count'}

    for col in ['dem_share_prev', 'rep_share_prev', 'dem_share_change_prev', 'rep_share_change_prev', 
               'partisanship', 'partisanship_change_prev']:
        if col in merged_df.columns:
            if col in ['partisanship', 'partisanship_change_prev']:
                agg_dict[col] = safe_mode
            else:
                agg_dict[col] = 'mean'

    for col in ['dem_share_prev', 'rep_share_prev', 'dem_share_change_prev', 'rep_share_change_prev', 
               'partisanship', 'true_label', 'predicted_label']:
        if col in merged_df.columns:
            if col in ['partisanship', 'true_label', 'predicted_label']:
                agg_dict[col] = safe_mode
            else:
                agg_dict[col] = 'mean'
    
    cluster_analysis = merged_df.groupby('cluster').agg(agg_dict).reset_index()

    # Extract ids per cluster
    standardized_id_nums = (
        merged_df.groupby('cluster')['standardized_id_num']
        .apply(list)
        .reset_index()
        .rename(columns={'standardized_id_num': 'standardized_id_nums'})
    )
    cluster_analysis = cluster_analysis.merge(standardized_id_nums, on='cluster')
    
    return cluster_analysis


def calculateClusterStats(cluster_id, cluster_analysis):
    cluster_size = cluster_analysis.loc[cluster_analysis['cluster'] == cluster_id, 'standardized_id_num'].values[0]
    if isinstance(cluster_size, numbers.Number) and pd.notnull(cluster_size):
        cluster_size = int(cluster_size)
        total = cluster_analysis['standardized_id_num'].sum()
        cluster_size_percent = (cluster_size / total) * 100 if total else 0
    else:
        cluster_size = "Unknown"
        cluster_size_percent = 0

    return cluster_size, cluster_size_percent


def categorizePartisanship(row):
    try:
        if row["dem_share_prev"] >= 0.667:
            return "strong democrat"
        elif row["dem_share_prev"] >= 0.501:
            return "leans democrat"
        elif row["rep_share_prev"] >= 0.667:
            return "strong republican"
        elif row["rep_share_prev"] >= 0.501:
            return "leans republican"
        elif row["oth_share_prev"] >= 0.667:
            return "strong independent"
        elif row["oth_share_prev"] >= 0.501:
            return "leans independent"
        else:
            return "neutral"
    except:
        return "unknown"


def categorizePartisanChange(row):
    try:
        change = row["rep_share_change_prev"] - row["dem_share_change_prev"]
        
        if np.abs(change) >= 0.01:
            if change > 0.5:
                return "gargantuanly more republican"
            if change > 0.35:
                return "massively more republican"
            if change > 0.25:
                return "much much more republican"
            if change > 0.15:
                return "much more republican"
            if change > 0.1:
                return "more republican"
            if change > 0.05:
                return "slightly more republican"
            elif change > 0.01:
                return "very slightly more republican"
            elif change > 0.005:
                return "infinitesimally more republican"
            elif change < -0.5:
                return "gargantuanly more democrat"
            elif change < -0.35:
                return "massively more democrat"
            elif change < -0.25:
                return "much much democrat"
            elif change < -0.15:
                return "much more democrat"
            elif change < -0.1:
                return "more democrat"
            elif change < -0.05:
                return "slightly more democrat"
            elif change < -0.01:
                return "very slightly more democrat"
            elif change < -0.005:
                return "infinitesimally more democrat"
        else:
            return "no change"
    except:
        return "unknown"


def createClusters(merged_df, valid_features, n_clusters=5, geo_sensitive=True, geo_weight=1):
    print("Creating clusters...")
    
    X = handleMissingValues(merged_df, valid_features)
    if X.empty or len(valid_features) == 0:
        print("Error: No valid features to cluster.")
        return merged_df, None
    
    # Must preceed scaler
    if geo_sensitive == True:
        merged_df['geometry'] = merged_df['geometry'].apply(lambda g: wkt.loads(g) if isinstance(g, str) else g)
        merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
        merged_df.set_crs(epsg=4326, inplace=True)  # Critical
        gdf_projected = merged_df.to_crs(epsg=6493)
        centroids = gdf_projected.geometry.centroid
        centroids_ll = gpd.GeoSeries(centroids, crs=gdf_projected.crs).to_crs(epsg=4326)
        merged_df['longitude'] = centroids_ll.x
        merged_df['latitude'] = centroids_ll.y
        geo_features = ['longitude', 'latitude']
        X[geo_features] = merged_df[geo_features]
        valid_features += geo_features

    X = X.dropna()
    merged_df = merged_df.loc[X.index]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Must follow scaler
    feature_names = valid_features.copy()
    if geo_sensitive == True:
        feature_names = list(dict.fromkeys(valid_features + geo_features))
        geo_indices = [feature_names.index(gf) for gf in geo_features]
        for idx in geo_indices:
            X_scaled[:, idx] *= geo_weight
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    merged_df['cluster'] = cluster_labels
    
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_names)

    return merged_df, centers


def determinePartisanBaseTrend(dem_share, rep_share, partisan_shift):
    if isinstance(dem_share, (int, float)) and isinstance(rep_share, (int, float)):
        partisan_base = "Democratic-leaning" if dem_share > rep_share else "Republican-leaning"
    else:
        partisan_base = "Unknown"

    if isinstance(partisan_shift, (int, float)):
        if partisan_shift > 0.01:
            trend = "Shifting Right"
        elif partisan_shift < -0.01:
            trend = "Shifting Left"
        else:
            trend = "Stable"
    else:
        trend = "Unknown"

    return partisan_base, trend


def formatSharesAndChanges(dem_share, rep_share, dem_change_prev, rep_change_prev):
    dem_share_fmt = f"{dem_share:.1%}" if isinstance(dem_share, (int, float)) else "Unknown"
    rep_share_fmt = f"{rep_share:.1%}" if isinstance(rep_share, (int, float)) else "Unknown"
    dem_change_prev_fmt = f"{dem_change_prev:+.1%}" if isinstance(dem_change_prev, (int, float)) else "Unknown"
    rep_change_prev_fmt = f"{rep_change_prev:+.1%}" if isinstance(rep_change_prev, (int, float)) else "Unknown"

    return dem_share_fmt, rep_share_fmt, dem_change_prev_fmt, rep_change_prev_fmt


def extractVotingPatterns(cluster_id, cluster_analysis):
    dem_share = cluster_analysis.loc[cluster_analysis['cluster'] == cluster_id, 'dem_share_prev'].values[0] if 'dem_share_prev' in cluster_analysis.columns else 0
    rep_share = cluster_analysis.loc[cluster_analysis['cluster'] == cluster_id, 'rep_share_prev'].values[0] if 'rep_share_prev' in cluster_analysis.columns else 0
    dem_change_prev = cluster_analysis.loc[cluster_analysis['cluster'] == cluster_id, 'dem_share_change_prev'].values[0] if 'dem_share_change_prev' in cluster_analysis.columns else 0
    rep_change_prev = cluster_analysis.loc[cluster_analysis['cluster'] == cluster_id, 'rep_share_change_prev'].values[0] if 'rep_share_change_prev' in cluster_analysis.columns else 0
    partisan_shift = rep_change_prev - dem_change_prev

    # Truth
    true_label = cluster_analysis.loc[cluster_analysis['cluster'] == cluster_id, 'true_label'].values[0] if 'true_label' in cluster_analysis.columns else "unknown"
    
    return dem_share, rep_share, dem_change_prev, rep_change_prev, true_label, partisan_shift


def featuresUsed(features):
    print("Using these features across all charts:")
    for feat in features:
        print(f"  - {feat}")


def getDefiningFeatures(centers, cluster_id):
    try:
        other_centers = centers.drop(cluster_id)
        differences = centers.iloc[cluster_id] - other_centers.mean()
        top_differences = differences.abs().sort_values(ascending=False).head(3)
        defining_features = top_differences.index.tolist()
    except:
        defining_features = []

    return defining_features


def handleMissingValues(df, features):
    print("Handling missing values...")
    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Filling {len(all_nan_cols)} entirely-NaN columns with 0s:")
        for col in all_nan_cols:
            print(f"  - {col}")
        X[all_nan_cols] = 0.0

    X.fillna(X.median(numeric_only=True), inplace=True)
    
    remaining_na = X.isna().sum().sum()
    if remaining_na > 0:
        print(f"Warning: {remaining_na} NaNs remain in the feature matrix after cleaning!")
    
    return X


def makeCensusFeatureLabels(feature_name, year):
    # We need to find the directory by only knowing the first part
    # of the name, which is the census id/code.
    if year == 2024: # No 2024 data yet.
        year = 2023
    feature_name_code = ''
    if feature_name[:1] == 'S':
        feature_name_code = feature_name[:5].upper()
        feature_name_label = feature_name[6:]
        data_type = 'ACSST5Y'
    elif feature_name[:1] == 'B':
        feature_name_code = feature_name[:6].upper()
        feature_name_label = feature_name[7:]
        data_type = 'ACSDT5Y'
    elif feature_name[:1] == 'D':
        feature_name_code = feature_name[:4].upper()
        feature_name_label = feature_name[5:]
        data_type = 'ACSDP5Y'
    else:
        # Not a census feature
        return feature_name

    partial_dir = feature_name_code.lower()
    base_path = 'data/census/'
    
    matching_dir = glob.glob(os.path.join(base_path, partial_dir + '*'))
    
    if matching_dir:
        target_dir = matching_dir[0]
        
        dataset_name = after_underscore = target_dir.split("_", 1)[1] # characters following the code.
        dataset_name = dataset_name.replace('_', ' ').title()
        file_path = os.path.join(target_dir, f'{data_type}{year}.{feature_name_code}-Column-Metadata.csv')

        df_columns = pd.read_csv(file_path)
        label = df_columns[df_columns['Column Name'] == feature_name].values[0][1]
        parts = label.split('!!')
        short_label = ' | '.join(parts)
        feature_label = f'{feature_name} | {dataset_name} | {short_label}'

        return feature_label


def plotPca(X_pca, pca, top_features, merged_df, pc_x, pc_y, filename, year):
    output_dir = 'output/personas'
    
    pc_x_label = f"PC{pc_x} ({pca.explained_variance_ratio_[pc_x-1]:.1%})\nTop: {', '.join(top_features[f'PC{pc_x}'])}"
    pc_y_label = f"PC{pc_y} ({pca.explained_variance_ratio_[pc_y-1]:.1%})\nTop: {', '.join(top_features[f'PC{pc_y}'])}"

    label_text = []
    top_feature_list = ['X Axis'] + ['------------'] + top_features[f'PC{pc_x}'] + ['======'] + ['Y Axis'] + ['------------'] + top_features[f'PC{pc_y}']
    for top_feature in top_feature_list:
        if 'Axis' in top_feature or top_feature in ['------------', '======']:
            label_text.append(top_feature)
        else:
            label_text.append(makeCensusFeatureLabels(top_feature, int(year)))
    label_text_str = '\n'.join(label_text)

    plt.figure(figsize=(16, 8))
    scatter = plt.scatter(X_pca[:, pc_x-1], X_pca[:, pc_y-1], c=merged_df['cluster'], cmap='plasma', alpha=0.5)
    unique_clusters = np.unique(merged_df['cluster'])
    handles = [
        plt.Line2D([], [], marker='o', color='w',
                   markerfacecolor=plt.cm.plasma(c / max(unique_clusters)),
                   label=f"Cluster {c}", markersize=10)
        for c in unique_clusters
    ]
    plt.legend(handles=handles, title='Voter Personas')
    plt.title(f'Voter Personas - PCA PC{pc_x} vs PC{pc_y}')
    plt.xlabel(pc_x_label)
    plt.ylabel(pc_y_label)
    plt.gcf().text(0.5, 0.5, label_text_str, fontsize=6, va='center', ha='left',
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.tight_layout(rect=[0, 0, 0.5, 1])
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()


def visualizePersonasCommon(merged_df, valid_features, year):
    if merged_df.empty or 'cluster' not in merged_df.columns or len(valid_features) < 2:
        print("Not enough data for visualizations")
        return None, None, None

    X = handleMissingValues(merged_df, valid_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components)], index=valid_features)
    top_features = {f'PC{i+1}': loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(3).index.tolist() for i in range(3)}

    return X_pca, pca, top_features, X


# ### Persona Creation

# In[ ]:


def createPersona(cluster_id, name, cluster_size, cluster_size_percent, dem_share_fmt, rep_share_fmt, dem_change_prev_fmt, rep_change_prev_fmt, true_label, predicted_label, defining_features, center_values, standardized_id_nums):
    persona = {
        'cluster_id': cluster_id,
        'name': name,
        'size': f"{cluster_size} precincts ({cluster_size_percent:.1f}%)" if isinstance(cluster_size, (int, float)) else "Unknown",
        'partisan_base': f"D: {dem_share_fmt}, R: {rep_share_fmt}",
        'partisan_trend': f"Change: D {dem_change_prev_fmt}, R {rep_change_prev_fmt}",
        'true_label': true_label,
        'predicted_label': predicted_label,
        'defining_features': defining_features,
        'center_values': center_values,
        'standardized_id_nums': standardized_id_nums,
    }

    return persona


from collections import Counter
def createPersonaDescriptions(merged_df, centers, cluster_analysis, valid_features):
    print("Creating persona descriptions...")

    personas = []

    for i in range(len(centers)):
        standardized_id_nums_list = []
        predicted_labels = []
        
        for standardized_id_nums in cluster_analysis['standardized_id_nums']:
            standardized_id_nums_list.append(standardized_id_nums)

            for standardized_id_num in standardized_id_nums:
                predicted_label = merged_df[merged_df['standardized_id_num'] == standardized_id_num]['predicted_label'].values[0]
                predicted_labels.append(predicted_label)
        
        cluster_id = i
        cluster_size, cluster_size_percent = calculateClusterStats(cluster_id, cluster_analysis)
        
        dem_share, rep_share, dem_change_prev, rep_change_prev, true_label, partisan_shift = extractVotingPatterns(cluster_id, cluster_analysis)
        partisan_base, trend = determinePartisanBaseTrend(dem_share, rep_share, partisan_shift)
        defining_features = getDefiningFeatures(centers, cluster_id)
        dem_share_fmt, rep_share_fmt, dem_change_prev_fmt, rep_change_prev_fmt = formatSharesAndChanges(dem_share, rep_share, dem_change_prev, rep_change_prev)
        center_values = dict(zip(valid_features, centers.iloc[cluster_id]))
        lat_lon = merged_df[merged_df['cluster'] == cluster_id][['latitude', 'longitude']].mean()
        center_values['latitude'] = lat_lon['latitude']
        center_values['longitude'] = lat_lon['longitude']

        # Ketchum
        predicted_label = Counter(predicted_labels).most_common(1)[0][0]
        
        persona = createPersona(
            cluster_id,
            f"Persona {cluster_id + 1}: {partisan_base} {trend}",
            cluster_size,
            cluster_size_percent,
            dem_share_fmt,
            rep_share_fmt,
            dem_change_prev_fmt,
            rep_change_prev_fmt,
            true_label,
            predicted_label,
            defining_features,
            center_values,
            standardized_id_nums_list,
        )

        personas.append(persona)

    return personas


def createPartisanSummary(personas, year, office):
    dem_shares = []
    rep_shares = []
    cluster_ids = []
    persona_names = []
    
    for persona in personas:
        dem_share = float(persona['partisan_base'].split('D: ')[1].split('%')[0])
        rep_share = float(persona['partisan_base'].split('R: ')[1].split('%')[0])
        
        dem_shares.append(dem_share)
        rep_shares.append(rep_share)
        cluster_ids.append(persona['cluster_id'])
        
        name_parts = persona['name'].split(':')
        if len(name_parts) > 1:
            persona_names.append(name_parts[0])
        else:
            persona_names.append(f"Persona {persona['cluster_id']}")
    
    if cluster_ids:
        plt.figure(figsize=(12, 8))
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        sorted_indices = np.argsort(cluster_ids)
        dem_shares_sorted = [dem_shares[i] for i in sorted_indices]
        rep_shares_sorted = [rep_shares[i] for i in sorted_indices]
        persona_names_sorted = [persona_names[i] for i in sorted_indices]
        
        plt.bar(x - width/2, dem_shares_sorted, width, label='Democrat %', color='blue', alpha=0.7)
        plt.bar(x + width/2, rep_shares_sorted, width, label='Republican %', color='red', alpha=0.7)
        
        for i, v in enumerate(dem_shares_sorted):
            plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9) 
        for i, v in enumerate(rep_shares_sorted):
            plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('Voter Persona')
        plt.ylabel('Vote Share (%)')
        plt.title('Partisan Base by Voter Persona')
        plt.xticks(x, persona_names_sorted, rotation=45, ha='right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        plt.text(len(cluster_ids)-1, 50.5, '50%', ha='right', va='bottom', color='black')
        plt.ylim(0, max(max(dem_shares), max(rep_shares)) * 1.15)
        plt.tight_layout()
        plt.savefig(f'output/personas/personas_summary_{year}_{office.replace(' ', '_').replace('.', '')}.png')
        plt.close()


# First point of execution
def createVoterPersonas(year, office, n_clusters=10, use_prediction_features_only=True, geo_sensitive=True, geo_weight=4):
    print(f"Creating voter personas for {year} {office}...")

    office_file = office.replace(".", "").replace(" ", "_")
    predictions, precinct_data = loadPrecinctData(year, office)
    census_dataset_dfs = loadCensusData(census_datasets, year, office)
    merged_df = mergeDatasets(precinct_data, census_dataset_dfs)

    # Many rows do not have predictions.
    merged_df['true_label'] = merged_df['true_label'].fillna(np.nan)
    
    # Classifications version
    feature_importance_file = f'data/generated_data/df_importances_{TARGET}.csv'
    feature_rankings = pd.read_csv(feature_importance_file)

    exclude_values = [
        'dem_share_change_prev', 'rep_share_change_prev', 'oth_share_change_prev',
        'rep_votes', 'rep_share', 'partisan_temp_prev', 'partisan_temp', 'dem_share_prev',
        'rep_share_prev', 'dem_share', 'dem_votes', 'standardized_id_num', 'nearest_bound_census_tract',
        'geoid_tract',
    ]
    feature_rankings = feature_rankings[~feature_rankings['Feature name'].isin(exclude_values)]

    top_n = 31
    importance_col = 'Average'
    top_features = feature_rankings.sort_values(importance_col, ascending=False).head(top_n)['Feature name'].tolist()
    valid_features = [feat for feat in top_features if feat in merged_df.columns]

    print(f"Selected {len(valid_features)} features for clustering")

    merged_df, centers = createClusters(merged_df, valid_features, n_clusters=n_clusters, geo_sensitive=True, geo_weight=4)
    cluster_analysis = analyzeVotingPatterns(merged_df)
    voter_regions = createVoterRegions(cluster_analysis, year, office)
    personas = createPersonaDescriptions(merged_df, centers, cluster_analysis, valid_features)
    createPartisanSummary(personas, year, office) # Bar chart
    visualizePersonas(merged_df, valid_features, personas, year, office) # Radar charts
    report = generatePersonasReport(personas, valid_features, merged_df, year, office)

    print(f"Created {n_clusters} voter personas. Reports and visualizations saved to output/personas/")

    return personas, merged_df, voter_regions


def createVoterRegions(cluster_analysis, year, office):
    dfs = {}
    
    for cluster_id in cluster_analysis['cluster']:
        standardized_id_nums = cluster_analysis['standardized_id_nums'][cluster_id]
        df_standardized_id_nums = pd.DataFrame(standardized_id_nums, columns=['standardized_id_num'])
        df_standardized_id_nums['standardized_id_num'] = df_standardized_id_nums['standardized_id_num'].apply(lambda x: str(x).zfill(13))
        
        df_precincts = gpd.read_file(f'data/generated_data/df_05_precinct_mapped_merged_{year}_{office.replace('.', '').replace(' ', '_')}.geojson')
        df_precincts = pd.merge(df_standardized_id_nums, df_precincts, on="standardized_id_num", how="left")
        df_precincts = df_precincts[['standardized_id_num', 'geometry']]
        df_precincts = gpd.GeoDataFrame(df_precincts, geometry='geometry')

        dfs[cluster_id] = df_precincts

    return dfs


def generatePersonasReport(personas, valid_features, merged_df, year, office):
    print("Generating comprehensive personas report...")
    
    report = []
    report.append("# Voter Persona Analysis\n")
    report.append("## Overview\n")
    report.append("This analysis identifies distinct voter groups based on demographic characteristics and voting patterns ")
    report.append("from precinct-level electoral data and census demographics. ")
    report.append(f"Using {len(valid_features)} key demographic features, we identified {len(personas)} distinct voter personas.\n")
    
    report.append("## Features Used for Analysis\n")
    report.append("The following demographic features were most predictive of voting pattern changes:\n")
    
    for i, feat in enumerate(valid_features, 1):
        feature_label = makeCensusFeatureLabels(feat, int(year))
        report.append(f"{i}. `{feature_label}`\n")

    report.append("\n")
    report.append("## Methodology\n")
    report.append("Voter personas were created using K-means clustering on standardized demographic features. ")
    report.append("Each persona represents a group of precincts with similar characteristics ")
    report.append("and voting patterns.\n")
    
    report.append("\n")
    report.append("## Voter Persona Summary\n")
    # report.append("| Voter Persona | Size | Partisan Base | Trend | True Label | Predicted label |\n")
    # report.append("|---------|------|---------------|-------|----------------|----------------|\n")
    report.append("| Voter Persona | Size | Partisan Base | Trend |\n")
    report.append("|---------|------|---------------|-------|\n")
    
    for p in personas:
        # report.append(f"| {p['name']} | {p['size']} | {p['partisan_base']} | {p['partisan_trend']} | {p['true_label']} | {p['predicted_label']} |\n")
        report.append(f"| {p['name']} | {p['size']} | {p['partisan_base']} | {p['partisan_trend']}|\n")
    
    report.append("\n")
    report.append("## Detailed Voter Persona Profiles\n")
    
    for p in personas:
        if 'center_values' in p and p['center_values']:
            center_values = p['center_values']
            latitude = center_values.pop('latitude')
            longitude = center_values.pop('longitude')

        report.append(f"### {p['name']}\n")
        report.append(f"**Num. Precincts**: {p['size']}\n")
        report.append(f"<br>**Partisan Base**: {p['partisan_base']}\n")
        report.append(f"<br>**Partisan Trend**: {p['partisan_trend']}\n")
        
        # report.append(f"<br>**True label**: {p['true_label']}\n")
        # report.append(f"<br>**Predicted label**: {p['predicted_label']}\n")
        
        report.append(f"<br>**Lat/Lon**: {latitude:.6f}, {longitude:.6f} | [View Map](https://www.google.com/maps/place/{latitude},{longitude}){{:target='_blank'}}\n")
        report.append("\n")
        
        report.append("#### Key Demographics\n")
        
        if 'center_values' in p and p['center_values']:
            center_values = p['center_values']

            feature_medians = {
                feat: pd.to_numeric(merged_df[feat], errors='coerce').median()
                for feat in valid_features
                if feat in merged_df.columns
            }
            
            feature_diffs = {
                feat: (center_values[feat] - feature_medians[feat]) / feature_medians[feat] * 100
                for feat in valid_features if feat in center_values and feature_medians[feat] != 0
            }
            
            sorted_features = sorted(feature_diffs.keys(), key=lambda x: abs(feature_diffs[x]), reverse=True)
            
            llm_feature_labels = []
            i = 1
           
            for feat in sorted_features[:31]:
                value = center_values[feat]
                diff = feature_diffs[feat]
                direction = "higher" if diff > 0 else "lower"
                feature_label = makeCensusFeatureLabels(feat, int(year))
                llm_feature_labels.append(f'{i}. DATASET: "{feature_label}" with an observed value of: "{value}" that is {direction} than median.' + "\n")
                report.append(f"- **{feature_label}**:<br>{value:.2f} ({abs(diff):.1f}% {direction} than median)\n")
                i = i + 1        
            
            report.append("\n")

        if LLM_ACTIVE == True:
            print('Generated LLM persona summary...')
            
            # ChatGPT interpretation of demo features
            prompt_intro = '''You are a political demographer and researcher. You write voter personas much like a dossier. 
                             Voter personas are created by carefully synthesizing census data. Take this numbered list of data points. 
                             Note each category and its value, as well as deviation from the median. Make sure to write at 
                             least three paragraphs, with headings, that characterize this voter persona. Also, make sure 
                             to write a summarized bulleted list. Use the tone and writing style of Robert McNamara, the former 
                             Secretary of Defense for John F Kennedy and Lyndon Johnson. Be sharp, concise, and objective.
                             Use adjectives only when justified by the data. Your audience are top political decision makers.\n'''
            prompt_intro += '''Formatting: use only h4 headings above paragraphs. Do not number lists. Above the bulleted list, 
                               add an h4 heading that says "Key Insights".'''
            
            prompt = prompt_intro + ' '.join(llm_feature_labels)
            
            llm_summary = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response = llm_summary.choices[0].message.content
            report.append("### LLM-Generated Voter Profile\n")
            report.append(response)
        
        report.append("\n---\n\n")
    
    report_path = os.path.join('output/personas', f'personas_report_{year}_{office.replace(' ', '_').replace('.', '')}.md')
    with open(report_path, 'w') as file:
        file.write(''.join(report))
    
    print(f"Comprehensive report saved to {report_path}")
    return ''.join(report)


def visualizePersonas(merged_df, valid_features, personas, year, office):
    print("Visualizing personas...")

    X_pca, pca, top_features, X = visualizePersonasCommon(merged_df, valid_features, year)
    if X_pca is None:
        return
    
    plotPca(X_pca, pca, top_features, merged_df, 1, 2, f'personas_pca_pc1_vs_pc2_{year}_{office.replace(' ', '_').replace('.', '')}.png', year)
    plotPca(X_pca, pca, top_features, merged_df, 2, 3, f'personas_pca_pc2_vs_pc3_{year}_{office.replace(' ', '_').replace('.', '')}.png', year)
    
    if not personas or not all('center_values' in p for p in personas):
        print("No valid persona data for visualization")
        return
    
    # Make plots
    all_feature_values = {feat: [p['center_values'][feat] for p in personas if feat in p['center_values']] for feat in valid_features}
    feature_variances = {feat: np.std(values) for feat, values in all_feature_values.items() if values}
    radar_features = sorted(feature_variances, key=feature_variances.get, reverse=True)[:min(31, len(feature_variances))]
    
    if 'latitude' in radar_features:
        radar_features.remove('latitude')
    if 'longitude' in radar_features:
        radar_features.remove('longitude')
        
    featuresUsed(radar_features)
    visualizePersonasRadar(personas, valid_features, radar_features, year, office, 'individual', X)


def visualizePersonasRadar(personas, valid_features, radar_features, year, office, suffix, X):
    feature_min = {feat: X[feat].min() for feat in radar_features}
    feature_max = {feat: X[feat].max() for feat in radar_features}

    for persona in personas:
        center_values = persona.get('center_values', {})
        if not center_values:
            continue
        
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111, polar=True)
        N = len(radar_features)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

        values_norm = [(center_values.get(feat, 0) - feature_min[feat]) / (feature_max[feat] - feature_min[feat]) if feature_max[feat] > feature_min[feat] else 0.5 for feat in radar_features]
        values_norm.append(values_norm[0])
        
        radar_features_named = [makeCensusFeatureLabels(f, int(year)) for f in radar_features]
        title_text = f'{persona['name']}'
        
        ax.plot(angles, values_norm, linewidth=2, linestyle='solid')
        ax.fill(angles, values_norm, alpha=0.25)
        ax.set_rlim(0, 1)
        plt.xticks(angles[:-1], radar_features_named, size=6)
        plt.title(title_text, size=15, y=1.1)
        plt.figtext(0.5, 0.01,
                    f"Size: {persona['size']}\nBase: {persona['partisan_base']}\nTrend: {persona['partisan_trend']}\nTrue Label: {persona['true_label']}",
                    ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        plt.tight_layout()
        safe_name = persona['name'].lower().replace(" ", "_").replace(":", "").replace("-", "-")
        plt.savefig(f'output/personas/persona_{safe_name}_{suffix}_{year}_{office.replace(' ', '_').replace('.', '')}.png')
        plt.close()


# ### Generate Report

# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    print(f'Num. of years to process: {len(value)}')
    
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    personas_dict = {}
    voter_regions_dict = {}
    
    for year in YEARS:
        print(f'Processing year {year}')
    
        personas_dict[year] = {}
        voter_regions_dict[year] = {}
        
        for office in OFFICES:
            print(f'Processing office {office}')
            office = office.replace(' ', '_').replace('.', '')
            
            personas, merged_df, voter_regions = createVoterPersonas(year, office, n_clusters=30, geo_sensitive=True, geo_weight=4)
    
            personas_dict[year][office] = personas
            voter_regions_dict[year][office] = voter_regions
    
            print('----------------------------')
        
        print('============================')

print('DONE')


# # Plots

# In[ ]:


# https://forum.inductiveautomation.com/t/mapping-a-css-gradient-based-on-numerical-or-percentage-scale/90063/2?utm_source=chatgpt.com
def interpolateColor(c1, c2, t):
    """
    Linearly interpolate between two RGB colors.
    """
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def getColorFromShare(dem_share):
    """
    Maps a dem_share (0-100) to a color on a red → orange → yellow → green → blue gradient.
    """
    t = dem_share / 100  # normalize to 0–1

    stops = [
        (0.00, (255, 0, 0)),     # Red
        (0.25, (255, 165, 0)),   # Orange
        (0.50, (255, 255, 0)),   # Yellow
        (0.75, (0, 255, 0)),     # Green
        (1.00, (0, 0, 255)),     # Blue
    ]
    
    # Find two stops for interpolation
    for i in range(len(stops) - 1):
        left_t, left_color = stops[i]
        right_t, right_color = stops[i + 1]
        
        if left_t <= t <= right_t:
            # Normalize btwn two stops
            local_t = (t - left_t) / (right_t - left_t)
            return interpolateColor(left_color, right_color, local_t)


def getSwingColor(dem_shift, intensity=10.0):
    """
    Maps dem_shift (-100 to 100) to RGB from red to blue using a tanh curve.
    - intensity: how sharply the color ramps up from center (higher = more contrast in small changes)
    """
    x = dem_shift / 100.0  # Normalize to [-1, 1]
    x_scaled = math.tanh(intensity * x)  # Nonlinear stretching within [-1, 1]
    t = (x_scaled + 1) / 2 # Rescale to [0, 1] for color mapping
    red = int(255 * (1 - t))
    blue = int(255 * t)
    green = 0
    return (red, green, blue)


def rgbToHex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def getDistinctColorPalette(n=40):
    base_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe",
        "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080",
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
        "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
        "#ffff99", "#b15928", "#bc80bd", "#ccebc5", "#ffed6f",
        "#bcbd22", "#17becf", "#8dd3c7", "#bebada", "#fb8072"
    ]
    if n > len(base_colors):
        raise ValueError(f"Requesting {n} colors, but only {len(base_colors)} are available.")
    return base_colors[:n]


def polygonToPatches(geom, color, hatch):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
    import numpy as np

    def createPathFromCoords(coords):
        verts = np.array(coords)
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
        return MplPath(verts, codes)

    patches = []

    if geom.geom_type == 'Polygon':
        path = createPathFromCoords(geom.exterior.coords)
        patch = PathPatch(path, facecolor=color, edgecolor='black', linewidth=0.01, hatch=hatch)
        patches.append(patch)

        for interior in geom.interiors:
            hole_path = createPathFromCoords(interior.coords)
            hole_patch = PathPatch(hole_path, facecolor='white', edgecolor='black', linewidth=0.01)
            patches.append(hole_patch)

    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:
            patches.extend(polygonToPatches(part, color, hatch))

    return patches


# ### Voter regions

# In[ ]:


def plotVoterRegion(personas, year, office, subregion=None, labels=True):
    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)
    
    # Cluster colors
    colors = getDistinctColorPalette(len(personas))

    # Hatch patterns for similar colors
    hatch_patterns = ['//', '\\\\', 'xx', '++', '--', '..', '**', 'oo']
    rgb_colors = [to_rgb(c) for c in colors]
    dist_matrix = squareform(pdist(rgb_colors))
    similarity_threshold = 0.25
    cluster_hatches = [None] * len(colors)
    hatch_index = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if dist_matrix[i][j] < similarity_threshold:
                if cluster_hatches[i] is None:
                    cluster_hatches[i] = hatch_patterns[hatch_index % len(hatch_patterns)]
                    hatch_index += 1
                if cluster_hatches[j] is None:
                    cluster_hatches[j] = hatch_patterns[hatch_index % len(hatch_patterns)]
                    hatch_index += 1
    
    # SE Michigan bounds
    se_mi_lon_min, se_mi_lon_max = -84.5, -82.5
    se_mi_lat_min, se_mi_lat_max = 41.8, 43.5
    
    metro_label_drawn = False

    if subregion == 'Southeast Michigan':
        lon_min, lon_max = se_mi_lon_min, se_mi_lon_max
        lat_min, lat_max = se_mi_lat_min, se_mi_lat_max
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        se_mi_bbox = box(lon_min, lat_min, lon_max, lat_max)
    else:
        ax.autoscale()
        se_mi_bbox = box(se_mi_lon_min, se_mi_lat_min, se_mi_lon_max, se_mi_lat_max)
    
    legend_patches = []
    cluster_id = 0
    
    for persona in personas:
        name = persona['name']
        name = name.replace('Persona', 'Region') # treat as regions
        
        color = colors[cluster_id]
        patch = mpatches.Patch(facecolor=color, edgecolor='black', label=name, hatch=cluster_hatches[cluster_id])
        legend_patches.append(patch)
        
        voter_region = voter_regions[cluster_id]
        voter_region.boundary.plot(ax=ax, color="black", linewidth=0.1)

        # Plot colors/hatching
        for geom in voter_region.geometry:
            for patch in polygon_to_patches(geom, color=color, hatch=cluster_hatches[cluster_id]):
                ax.add_patch(patch)
        
        # Combine geometries within cluster
        dissolved = voter_region.dissolve()
        disconnected_parts = dissolved.explode(index_parts=False)

        # Labeling logic
        if labels == True:
            for geom in disconnected_parts.geometry:
                part_gs = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs(epsg=3857)
            
                # Min distance between contiguous regions for labeling
                area_km2 = part_gs.area.iloc[0] / 1e6
                if subregion != None:
                    if area_km2 < 225:
                        continue
                else:
                    if area_km2 < 25:
                        continue
            
                # Centroid for label (reprojected back to WGS84)
                centroid = part_gs.centroid.to_crs(epsg=4326).iloc[0]
    
                # Don't plot anything outside subregion bounding box
                if subregion != None:
                    if not (lon_min <= centroid.x <= lon_max and lat_min <= centroid.y <= lat_max):
                        continue
    
                # Suppress local labels inside Metro Detroit when plotting full state
                if not subregion and se_mi_bbox.contains(centroid):
                    if not metro_label_drawn:
                        ax.text(
                            -83.3, 42.5,
                            "Metro Detroit",
                            fontsize=48,
                            ha='center',
                            va='center',
                            color='black',
                            weight='bold',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
                        )
                        metro_label_drawn = True
                    continue
                
                # Symbol position offset
                x_offset = 0.005
                symbol_x = centroid.x - x_offset
                text_x = centroid.x + x_offset
                y = centroid.y
    
                # Subregions hve larger text
                if subregion:
                    fontsize = 21
                else:
                    fontsize = 14
                
                # Colored symbols
                ax.text(
                    symbol_x,
                    y,
                    '■',
                    fontsize=fontsize,
                    ha='right',
                    va='center',
                    color=color,
                    weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                )
            
                # Black text labels
                ax.text(
                    text_x,
                    y,
                    name,
                    fontsize=fontsize,
                    ha='left',
                    va='center',
                    color='black',
                    weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                )
    
        cluster_id += 1

    # Legend
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=32,
        title='Voter Regions',
        title_fontsize=36,
        handler_map={mpatches.Patch: HandlerPatch()},
    )
        
    ax.margins(0)
    ax.set_title('Voter Regions', fontsize=64)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Append region info
    filename_subregion = "Statewide" if subregion is None else subregion.replace(' ', '_').replace('.', '')
    
    plt.savefig('output/maps/regions/Voter_Region_' + str(year) + "_" + office.replace('.', '').replace(' ', '_') + "_Map_" + filename_subregion + ".png", bbox_inches='tight')
    plt.close()


# In[ ]:


for year in YEARS:
    print(f'Processing year {year}')
    
    for office in OFFICES:
        print(f'Processing office {office}')

        office = office.replace(' ', '_').replace('.', '')

        plotVoterRegion(personas_dict[year][office], year, office, subregion=None, labels=False)
        plotVoterRegion(personas_dict[year][office], year, office, subregion='Southeast Michigan', labels=False)

print('DONE')


# ### Voter Region Leanings

# In[ ]:


def plotVoterRegionLeanings(personas, voter_regions, year, office):
    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)
    
    # Define gradient (normalized to 0–1 range)
    cmap = LinearSegmentedColormap.from_list(
        'dem_share_cmap',
        [
            (0.00, (1.0, 0.0, 0.0)),     # Red
            (0.25, (1.0, 0.65, 0.0)),    # Orange
            (0.50, (1.0, 1.0, 0.0)),     # Yellow
            (0.75, (0.0, 1.0, 0.0)),     # Green
            (1.00, (0.0, 0.0, 1.0))      # Blue
        ]
    )
    
    cluster_id = 0
    for persona in personas:
        dem_share = float(re.search(r'(\d+(?:\.\d+)?)%', persona['partisan_base'])[0][:-1])
        # rep_share = re.findall(r'(\d+(?:\.\d+)?)%', persona['partisan_base'])[1]
        
        color = rgbToHex(getColorFromShare(dem_share))
        
        voter_region = voter_regions[cluster_id]
        voter_region.boundary.plot(ax=ax, color="black", linewidth=0.1)
        voter_region.plot(ax=ax, color=color, edgecolor="black", linewidth=0.01)
    
        cluster_id += 1
    
    ax.margins(0)
    ax.set_title('Voter Regions', fontsize=64)
    ax.set_axis_off()
    
    cax = divider.append_axes("right", size="2%", pad=0.5)
    
    # Colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Democratic Share (%)', fontsize=32)
    cb.ax.tick_params(labelsize=24)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig('output/maps/regions/Voter_Region_' + str(year) + "_" + office.replace('.', '').replace(' ', '_') + "_Leaning_Map.png", bbox_inches='tight')
    plt.close(fig)


# In[ ]:


for year in YEARS:
    print(f'Processing year {year}')
    
    for office in OFFICES:
        print(f'Processing office {office}')

        office = office.replace(' ', '_').replace('.', '')

        plotVoterRegionLeanings(personas_dict[year][office], voter_regions_dict[year][office], year, office)

print('DONE')


# ### Voter Region Shifts

# In[ ]:


# Plot shifts in partisan loyalties across the state by investigating
# "partisan_trends" that was computed in the report/analysis.
def plotVoterRegionShifts(personas, voter_regions, year, office):
    fig, ax = plt.subplots(figsize=(80, 80))
    divider = make_axes_locatable(ax)

    # Define gradient (normalized to 0–1 range)
    cmap = LinearSegmentedColormap.from_list(
        'swing_cmap',
        [
            (0.00, (1.0, 0.0, 0.0)),  # Red
            (0.50, (0.5, 0.0, 0.5)),  # Purple midpoint
            (1.00, (0.0, 0.0, 1.0))   # Blue
        ]
    )
    
    cluster_id = 0
    for persona in personas:
        dem_share = float(re.findall(r'(-?\d+(?:\.\d+)?)%', persona['partisan_trend'])[0])
        rep_share = float(re.findall(r'(-?\d+(?:\.\d+)?)%', persona['partisan_trend'])[1])
        color = rgb_to_hex(getSwingColor(dem_share))
        voter_region = voter_regions[cluster_id]
        voter_region.boundary.plot(ax=ax, color="black", linewidth=0.1)
        voter_region.plot(ax=ax, color=color, edgecolor="black", linewidth=0.01)
        cluster_id += 1
    
    ax.margins(0)
    ax.set_title('Voter Regions', fontsize=64)
    ax.set_axis_off()
    
    cax = divider.append_axes("right", size="2%", pad=0.5)
    norm = mpl.colors.Normalize(vmin=-100, vmax=100)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Partisan Shift (% → Democratic)', fontsize=32)
    cb.ax.tick_params(labelsize=24)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig('output/maps/regions/Voter_Region_' + str(year) + "_" + office.replace('.', '').replace(' ', '_') + "_Shift_Map.png", bbox_inches='tight')
    plt.close(fig)


# In[ ]:


for year in YEARS:
    print(f'Processing year {year}')
    
    for office in OFFICES:
        print(f'Processing office {office}')

        office = office.replace(' ', '_').replace('.', '')

        plotVoterRegionShifts(personas_dict[year][office], voter_regions_dict[year][office], year, office)

print('DONE')


# In[ ]:




