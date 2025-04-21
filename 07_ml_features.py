#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# President: 2016 (Trump), 2020 (Biden), 2024 (Trump)
# U.S. Senate: 2014 (Peters), 2018 (Stabenow), 2020 (Peters), 2024 (Slotkin)
# U.S. House: every cycle
# State Senate: 2014, 2018, 2022
# State House: every cycle

ELECTIONS = {}

# SKIP FIRST TWO ELECTIONS FOR EVERY OFFICE:
ELECTIONS['U.S. House'] =   ['2018', '2020', '2022', '2024']
ELECTIONS['State House'] =  ['2018', '2020', '2022', '2024']
ELECTIONS['U.S. Senate'] =  ['2020', '2024']
ELECTIONS['State Senate'] = ['2022']
ELECTIONS['President'] =    ['2024']


# In[ ]:


from functools import reduce
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import csv
import gc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


pd.set_option("display.max_columns", None)


# In[ ]:


def formatOfficeName(office):
    return office.replace(' ', '_').replace('.', '')


def calcPartisanChangeAmountPrev(row):
    # negative => left => more dem, positive => right => more rep
    change_amount = row["rep_share_change_prev"] - row["dem_share_change_prev"]
    return change_amount


def calcPartisanChangeAmountCurr(row):
    # negative => left => more dem, positive => right => more rep
    change_amount = row["rep_share_change_curr"] - row["dem_share_change_curr"]
    return change_amount


def categorizePartnership(row, col_name):
    #col_names:
    # dem_share_prev or dem_share
    # rep_share_prev or rep_share
    # oth_share_prev or rep_share
    if row[col_name] >= 0.667:
        return "strong democrat"
    elif row[col_name] >= 0.501:
        return "leans democrat"
    elif row[col_name] >= 0.667:
        return "strong republican"
    elif row[col_name] >= 0.501:
        return "leans republican"
    elif row[col_name] >= 0.667:
        return "strong independent"
    elif row[col_name] >= 0.501:
        return "leans independent"
    else:
        return "neutral"


def categorizePartisanChange(row):
    # Model with a number line, ignore "other" parties
    # negative = left = more dem, positive = right = more rep
    change = row["rep_share_change_prev"] - row["dem_share_change_prev"]
    
    if np.abs(change) >= 0.01:
        if change > 0.5:
            return "more republican ++++++++"
        if change > 0.35:
            return "more republican +++++++"
        if change > 0.25:
            return "more republican ++++++"
        if change > 0.15:
            return "more republican +++++"
        if change > 0.1:
            return "more republican ++++"
        if change > 0.05:
            return "more republican +++"
        elif change > 0.01:
            return "more republican ++"
        elif change > 0.005:
            return "more republican +"
        elif change < -0.5:
            return "more democrat ++++++++"
        elif change < -0.35:
            return "more democrat +++++++"
        elif change < -0.25:
            return "more democrat ++++++"
        elif change < -0.15:
            return "more democrat +++++"
        elif change < -0.1:
            return "more democrat ++++"
        elif change < -0.05:
            return "more democrat +++"
        elif change < -0.01:
            return "more democrat ++"
        elif change < -0.005:
            return "more democrat +"
    else:
        return "no change"


def categorizePartisanTemp(temp):
    if temp <= -0.5:
        return "scorching democrat"
    elif temp <= -0.25:
        return "blazing democrat"
    elif temp <= -0.125:
        return "hot democrat"
    elif temp <= -0.0625:
        return "warm democrat"
    elif temp >= 0.5:
        return "scorching republican"
    elif temp >= 0.25:
        return "blazing republican"
    elif temp >= 0.125:
        return "hot republican"
    elif temp >= 0.0625:
        return "warm republican"
    else:
        return "neutral"


def cleanColumnNames(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.replace('.', '', regex=False)
        .str.replace(' ', '_')
        .str.replace('/', '_')
    )
    return df


def formatColumnTypes(df, non_numeric_columns):
    for col in df.columns:
        if col not in non_numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # ignore or coerce
        
        if col == 'standardized_id_num':
            df[col] = df[col].astype(str).str.zfill(13)
            
    return df


# In[ ]:


census_datasets = [
    'b02001_race', 'b04007_ancestry', 'b05012_nativity_us', 'b08303_travel_time_work', 'b25003_housing_rentership', 
    'dp02_selected_social_characteristics', 'dp03_selected_economic_characteristics', 'dp04_housing_characteristics', 'dp05_age_race', 
    's0101_age_sex', 's1101_households_families', 's1201_marital_status', 's1501_educational_attainment', 's1701_income_poverty', 
    's1903_median_income', 's2101_veteran_status', 's2201_food_stamps', 's2301_employment_status', 's2401_occupation_sex', 
    's2403_industry_sex', 's2501_occupancy_characteristics', 's2701_health_insurance', 's2503_financial_characteristics',
]


# In[ ]:


def makeData(years, offices, historic=True):
    dfs = {}

    non_numeric_columns = [
        'Precinct Label', 'Office Description', 'County Name', 'Election Type', 'City/Township Description',
        'standardized_id', 'standardized_id_num', 'geometry', 'geometry_tract', 'geoidfq_tract',
    ]

    drop_columns = [
        'Census County Code', 'City/Township Code', 'District Code', 'Election Type', 'Election Year',
        'Precinct Label', 'County Name', 'City/Township Description', 'Office Description',
        'Michigan County Code', 'Precinct Label', 'Precinct Number', 'Status Code', 'Ward Number',
        # 'registered_voters', 'standardized_id', #'geometry',
        # 'aland_tract', 'awater_tract', 'tractce_tract', 'geoid_tract', 'name_tract', 'geometry_tract',
        # 'nearest_bound_school_district', 'nearest_bound_census_tract', 'nearest_bound_zipcode',
    ]

    for year in years:
        print(f'Processing year {year}...')
        dfs[year] = {}
        
        for office in offices:
            office = formatOfficeName(office)
            print(f'Processing office {office}...')
            
            df = pd.read_csv(f'data/generated_data/df_06_tract_{year}_{office}.csv')

            drop_columns_clean = [col for col in drop_columns if col in df.columns] # Don't drop non-existing columns

            # These are targets
            if historic:
                df["partisan_temp_category"] = df['partisan_temp'].apply(categorizePartisanTemp)
                df["partisanship_lean_curr"] = df.apply(lambda row: categorizePartnership(row, 'dem_share'), axis=1)

            df["partisanship_lean_prev"] = df.apply(lambda row: categorizePartnership(row, 'dem_share_prev'), axis=1)
            df["partisanship_lean_change_prev"] = df.apply(lambda row: categorizePartisanChange(row), axis=1)
            df["partisanship_lean_change_amount_prev"] = df.apply(lambda row: calcPartisanChangeAmountPrev(row), axis=1)
            df["partisanship_lean_change_amount_curr"] = df.apply(lambda row: calcPartisanChangeAmountCurr(row), axis=1)
            non_numeric_columns.extend(['partisanship_lean_curr', 'partisanship_lean_prev', 'partisanship_lean_change_prev', 'partisan_temp_category'])

            df = df.drop(columns=drop_columns_clean, errors='coerce')
            
            df = formatColumnTypes(df, non_numeric_columns)
            df = cleanColumnNames(df)
            
            df['standardized_id_num'] = df['standardized_id_num'].astype(str).str.zfill(13)
    
            print(f'Loading census data...')
            census_dataset_dfs = []
            for census_dataset in census_datasets:
                census_dataset = census_dataset.lower()
                if census_dataset[:1] == 's':
                    census_dataset_code = census_dataset[:5].upper()
                    census_dataset_label = census_dataset[6:]
                elif census_dataset[:1] == 'b':
                    census_dataset_code = census_dataset[:6].upper()
                    census_dataset_label = census_dataset[7:]
                
                df_census_dataset = pd.read_csv(f'data/generated_data/df_06_{census_dataset_label}_' + year + '_' + office + '.csv')
                df_census_dataset.rename(columns={f'geoid_{census_dataset_label}': 'geoidfq_tract'}, inplace=True)
                
                census_dataset_dfs.append(df_census_dataset)

            dfs_extended = [df]
            dfs_extended.extend(census_dataset_dfs)
            
            # Get rid of mysterious column dups, when dataset is third-to-last in the list. But this is brittle.
            df = reduce(lambda left, right: pd.merge(left, right, on='geoidfq_tract', how='left', suffixes=('', '_dup')), dfs_extended)
            df = df.loc[:,~df.columns.str.endswith('_dup')]

            # Remove empty columns and rows
            df = df.replace('', np.nan).dropna(axis=1, how='all')
            
            print(f'Dataframe has {len(df)} rows *before* dropping missing voting history...')
            df = df.dropna(subset=['dem_share_change_prev']) # Drop anything without history
            print(f'Dataframe has {len(df)} rows *after* dropping missing voting history...')

            # For feature ranking
            df.to_csv(f'data/generated_data/07_ml_features_{year}_{office}_with_geometry.csv', index=False)

            df = df.drop(columns=['geometry'])
            df.to_csv(f'data/generated_data/07_ml_features_{year}_{office}.csv', index=False)
            
            dfs[year][office] = df

    return dfs


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    print(f'Num. of years to process: {len(value)}')
    
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    dfs = makeData(YEARS, OFFICES, historic=True)


# In[ ]:




