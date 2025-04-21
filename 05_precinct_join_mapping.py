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


import geopandas as gpd
import numpy as np
import pandas as pd


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


pd.set_option("display.max_columns", None)


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
            
            df_precinct_nearest_school = pd.read_csv('data/generated_data/df_04_bound_nearest_school_district_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.csv')
            df_precinct_nearest_tract = pd.read_csv('data/generated_data/df_04_bound_nearest_census_tract_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.csv')
            df_precinct_nearest_zipcode = pd.read_csv('data/generated_data/df_04_bound_nearest_zipcode_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.csv')
            
            ''' 
            Open Elections data.
            Precinct-level voting outcomes per election cycle.
            This dataset and corresponding precinct shape files form the core of data in this project.
            '''
            df_precinct_mapped = gpd.read_file('data/generated_data/df_02_vote_changes_calc_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.geojson', driver='GeoJSON')
            
            from functools import reduce
            dfs_to_merge = [df_precinct_mapped, df_precinct_nearest_school, df_precinct_nearest_tract, df_precinct_nearest_zipcode]
            
            for df in dfs_to_merge:
                df['standardized_id_num'] = df['standardized_id_num'].astype(str).str.zfill(13)
            
            df_precinct_mapped_merged = reduce(lambda left, right: pd.merge(left, right, on='standardized_id_num', how='left'), dfs_to_merge)
            df_precinct_mapped_merged.to_file('data/generated_data/df_05_precinct_mapped_merged_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.geojson', driver='GeoJSON')


# In[ ]:




