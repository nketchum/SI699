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


pd.set_option("display.max_columns", None)


# ### Join tract data to precinct results

# If the year is 2024, use 2023 data since 2024 data is not yet available. That is why reading files with years uses this <code>str(YEAR if YEAR != 2024 else 2023)</code> so that 2024 year reads filenames with "2023".

# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    for year in YEARS:
        print(f'Processing year {year}...')
        
        for office in OFFICES:
            print(f'Processing office {office}...')
            
            df_precinct_results_mapped = gpd.read_file('data/generated_data/df_05_precinct_mapped_merged_' + year + '_' + office.replace('.', '').replace(' ', '_') + '.geojson', driver='GeoJSON')
            df_precinct_results_mapped['nearest_bound_census_tract'] = df_precinct_results_mapped['nearest_bound_census_tract'].apply(lambda x: str(x)[:-2]) # Cast as string and remove the decimal
            
            # If year is 2024, use 2023 data since 2024 has no data yet.
            if year == '2024':
                census_year = '2023'
            else:
                census_year = year
            
            df_census_tracts = gpd.read_file('data/census/tracts/cb_' + census_year + '_26_tract_500k/cb_' + census_year + '_26_tract_500k.shp')
            
            df_census_tracts.columns = df_census_tracts.columns.str.lower()
            df_census_tracts['geoid'] = df_census_tracts['geoid'].astype(str)
            
            if int(year) >= 2023:
                # IF YEAR >= 2023 geoidfq
                df_census_tracts.rename(columns={'tractce': 'tractce_tract', 'geoid': 'geoid_tract', 'geoidfq': 'geoidfq_tract', 'name': 'name_tract', 'aland': 'aland_tract', 'awater': 'awater_tract', 'geometry': 'geometry_tract'}, inplace=True)
                df_census_tracts = df_census_tracts[['tractce_tract', 'geoid_tract', 'geoidfq_tract', 'name_tract', 'aland_tract', 'awater_tract', 'geometry_tract']]
            else:
                # IF YEAR < 2023 affgeoid
                df_census_tracts.rename(columns={'tractce': 'tractce_tract', 'geoid': 'geoid_tract', 'affgeoid': 'geoidfq_tract', 'name': 'name_tract', 'aland': 'aland_tract', 'awater': 'awater_tract', 'geometry': 'geometry_tract'}, inplace=True)
                df_census_tracts = df_census_tracts[['tractce_tract', 'geoid_tract', 'geoidfq_tract', 'name_tract', 'aland_tract', 'awater_tract', 'geometry_tract']]
            
            df_precinct_results_mapped['nearest_bound_census_tract'] = df_precinct_results_mapped['nearest_bound_census_tract'].str.ljust(11, '0') # add trailing zeros if less than 11 characters.
            df_precinct_results_tracts = pd.merge(df_precinct_results_mapped, df_census_tracts, left_on='nearest_bound_census_tract', right_on='geoid_tract', how='left')
            
            # geoid_tracts aren't consistently right-padded, so partial match
            i = 0
            for idx, row in df_precinct_results_tracts[df_precinct_results_tracts['geoid_tract'].isna()].iterrows():
                precinct_geoid = str(row['nearest_bound_census_tract']) # Here's one without a match in df_census_tracts
                if len(df_census_tracts[df_census_tracts['geoid_tract'].str.contains(precinct_geoid[:-1])]) > 0: # Remove one character
                    i += 1
                    match_df = df_census_tracts[df_census_tracts['geoid_tract'].str.contains(precinct_geoid[:-1])]
                    match_row = match_df.iloc[0]
                elif len(df_census_tracts[df_census_tracts['geoid_tract'].str.contains(precinct_geoid[:-2])]) > 0: # Remove two characters
                    i += 1
                    match_df = df_census_tracts[df_census_tracts['geoid_tract'].str.contains(precinct_geoid[:-2])]
                    match_row = match_df.iloc[0]
    
                cols_to_copy = ['tractce_tract', 'geoid_tract', 'geoidfq_tract', 'name_tract', 'aland_tract', 'awater_tract', 'geometry_tract']
                for col in cols_to_copy:
                    df_precinct_results_tracts.at[idx, col] = match_row[col]
            print(f'Partial matched {i} geoid_tracts...')
            
            df_precinct_results_tracts['standardized_id_num'] = df_precinct_results_tracts['standardized_id_num'].astype(str).str.zfill(13)
    
            # Store geometry as text to preserve rows
            df_precinct_results_tracts['geometry'] = df_precinct_results_tracts['geometry_tract'].to_wkt()
            df_precinct_results_tracts['geometry_tract'] = df_precinct_results_tracts['geometry_tract'].to_wkt()
    
            df_precinct_results_tracts.to_csv('data/generated_data/df_06_tract_' + year + '_' + office.replace('.', '').replace(' ', '_') + '.csv', index=False)

print('Done.')


# ### Join census demographic data

# In[ ]:


def makeCensusData(datasets, YEARS, OFFICES):
    dfs = {}

    i = 1
    print(f'Num datasets: {len(datasets)}')
    
    for dataset in datasets:
        print(f'Processing census dataset num. {i}...')
        
        for year in YEARS:
            print(f'Processing year {year}...')
            
            dfs[year] = {}
            
            for office in OFFICES:
                print(f'Processing office {office}...')
                
                # Census datasets have varied codes
                dataset = dataset.lower()
                if dataset[:1] == 's':
                    dataset_code = dataset[:5].upper()
                    dataset_label = dataset[6:]
                    data_type = 'ACSST5Y'
                elif dataset[:1] == 'b':
                    dataset_code = dataset[:6].upper()
                    dataset_label = dataset[7:]
                    data_type = 'ACSDT5Y'
                elif dataset[:1] == 'd':
                    dataset_code = dataset[:4].upper()
                    dataset_label = dataset[5:]
                    data_type = 'ACSDP5Y'

                # If year is 2024, use 2023 data since 2024 has no data yet.
                if year == '2024':
                    census_year = '2023'
                else:
                    census_year = year

                df = pd.read_csv(f'data/census/{dataset}/{data_type}' + census_year + f'.{dataset_code}-Data.csv', header=0, skiprows=[1])
        
                df_columns = pd.read_csv(f'data/census/{dataset}/{data_type}' + census_year + f'.{dataset_code}-Column-Metadata.csv', header=0, skiprows=[2])
                df_columns = df_columns[~df_columns["Label"].str.contains("Margin of Error!!", na=False, case=False)]
                columns = list(df_columns['Column Name'])

                for col in df_columns.columns:
                    if col.startswith(('S', 'B', 'D')):
                        invalid_values = ['(X)', '-', 'N/A', 'null', '']
                        df_columns[col] = df_columns[col].replace(invalid_values, np.nan)
                
                df = df[columns]
                df.rename(columns={'GEO_ID': f'geoid_{dataset_label}'}, inplace=True)
                df.to_csv(f'data/generated_data/df_06_{dataset_label}_' + year + '_' + office.replace('.', '').replace(' ', '_') + '.csv', index=False)

                dfs[year][office.replace(' ', '_').replace('.', '')] = df
        
        i += 1
    
    return dfs


# In[ ]:


census_datasets = [
    'b02001_race', 'b04007_ancestry', 'b05012_nativity_us', 'b08303_travel_time_work', 'b25003_housing_rentership', 
    'dp02_selected_social_characteristics', 'dp03_selected_economic_characteristics', 'dp04_housing_characteristics', 'dp05_age_race', 
    's0101_age_sex', 's1101_households_families', 's1201_marital_status', 's1501_educational_attainment', 's1701_income_poverty', 
    's1903_median_income', 's2101_veteran_status', 's2201_food_stamps', 's2301_employment_status', 's2401_occupation_sex', 
    's2403_industry_sex', 's2501_occupancy_characteristics', 's2701_health_insurance', 's2503_financial_characteristics',
]

print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    dfs = makeCensusData(census_datasets, YEARS, OFFICES)


# In[ ]:




