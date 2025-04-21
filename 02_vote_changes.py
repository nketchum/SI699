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


import gc
import geopandas as gpd
import numpy as np
import os
import pandas as pd


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


pd.set_option("display.max_columns", None)


# In[ ]:


def remapStandardId(df):
    # Use 2018 and 2020 data that straddle the census
    # that caused the remapping.
    df_id_map = pd.read_csv("data/custom_data/standardized_id_change_map.csv").dropna()
    df_id_map['standardized_id_num_2018'] = df_id_map['standardized_id_num_2018'].astype(float).astype(int).astype(str).str.zfill(13)
    df_id_map['standardized_id_num_2020'] = df_id_map['standardized_id_num_2020'].astype(float).astype(int).astype(str).str.zfill(13)
    reverse_map = dict(zip(df_id_map['standardized_id_num_2020'], df_id_map['standardized_id_num_2018']))
    
    # Clean and format ID column first
    df['standardized_id_num'] = (
        pd.to_numeric(df['standardized_id_num'], errors='coerce')
        .astype('Int64')  # Nullable int
        .astype(str)
        .str.zfill(13)
    )

    # For counting replacements
    original_ids = df['standardized_id_num'].copy()

    # Apply reverse mapping
    print(f'Looking at {len(df['standardized_id_num'].map(reverse_map).fillna(df['standardized_id_num']))} rows...')
    df['standardized_id_num'] = df['standardized_id_num'].map(reverse_map).fillna(df['standardized_id_num'])

    # Count replacements
    num_replaced = (df['standardized_id_num'] != original_ids).sum()
    print(f"{num_replaced} IDs were replaced...")
    
    return df


# In[ ]:


def pedersenIndexRow(row):
    parties = ['dem', 'rep', 'oth']
    total_change = 0.0
    
    for party in parties:
        prev_col = f"{party}_share_prev"
        curr_col = f"{party}_share"
        
        if pd.notna(row[prev_col]) and pd.notna(row[curr_col]):
            total_change += abs(row[curr_col] - row[prev_col])
        else:
            return np.nan
    
    return 0.5 * total_change


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')
    
    dfs = {}
    
    for year in YEARS:
        dfs[year] = {}
        
        for office in OFFICES:
            office = office.replace('.', '').replace(' ', '_')
            df_precinct_vote = gpd.read_file('data/generated_data/df_01_election_' + str(year) + '_' + office + '.geojson', driver='GeoJSON')        
            dfs[year][office] = df_precinct_vote
            del(df_precinct_vote)
    
    for year in dfs:
        for office in OFFICES:
            office = office.replace('.', '').replace(' ', '_')
            df = dfs[year][office]
            
            df = df[['standardized_id_num', 
                'dem_votes', 'rep_votes', 'oth_votes', 
                'dem_share', 'rep_share', 'oth_share', 
                'registered_voters', 'turnout_pct',
                'partisan_temp']]
            
            df = df[df["standardized_id_num"].notna() & (df["standardized_id_num"] != "")]
    
            # Map 2020+ standard ids to older standard ids
            if int(year) >= 2020:
                df = remapStandardId(df)
    
            dfs[year][office] = df
            
            df.to_csv('data/generated_data/df_02_vote_changes_' + str(year) + '_' + office + '.csv', index=False)
    
            del(df)


# ### Add vote shifts

# #### Calculate *past* cycle outcomes
# Use previous two elections from the current election to guage historical changes.

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
            
            # No previous changes to compute until the third dataset.
            # Third dataset and beyond will include the change in partisanship
            # from the previous two cycles, so that we are only looking backwards
            # and not looking at "unseen" information for new data.            
            if (year <= YEARS[1]):
                continue # Third dataset not yet reached.        
    
            # Current cycle
            df_curr = pd.read_csv(f"data/generated_data/df_02_vote_changes_{year}_{office.replace('.', '').replace(' ', '_')}.csv")
    
            # Different offices have different cycles, and the State Senate
            # cycles do not have consistent gaps.
            prev_cycle = 2  # 2 years
            prev_prev_cycle = 4  # 4 years
    
            # U.S. Senate cycles vary
            if office == 'U.S. Senate':
                if year == '2020':
                    prev_cycle = 2
                    prev_prev_cycle = 6
                elif year == '2024':
                    prev_cycle = 4
                    prev_prev_cycle = 6
            # State Senate and Prez are 4-year cycles
            elif (office == 'State Senate') | (office == 'President'):
                    prev_cycle = 4
                    prev_prev_cycle = 8
            
            # Previous-previous cycle – 4 years ago
            prev_prev_year = str(int(year) - prev_prev_cycle)
            df_prev_prev = pd.read_csv(f"data/generated_data/df_02_vote_changes_{prev_prev_year}_{office.replace('.', '').replace(' ', '_')}.csv")
            df_prev_prev = df_prev_prev.rename(columns={
                'dem_votes': 'dem_votes_prev_prev', 'rep_votes': 'rep_votes_prev_prev', 'oth_votes': 'oth_votes_prev_prev', 
                'dem_share': 'dem_share_prev_prev', 'rep_share': 'rep_share_prev_prev', 'oth_share': 'oth_share_prev_prev',
                'registered_voters': 'registered_voters_prev_prev', 'turnout_pct': 'turnout_pct_prev_prev',
                'partisan_temp': 'partisan_temp_prev_prev', 'pedersen_index_prev': 'pedersen_index_prev_prev',
                'pedersen_index_percent_prev': 'pedersen_index_percent_prev_prev',
            })
            
            # Previous cycle – 2 years ago
            prev_year = str(int(year) - prev_cycle)
            df_prev = pd.read_csv(f"data/generated_data/df_02_vote_changes_{prev_year}_{office.replace('.', '').replace(' ', '_')}.csv")
            df_prev = df_prev.rename(columns={
                'dem_votes': 'dem_votes_prev', 'rep_votes': 'rep_votes_prev', 'oth_votes': 'oth_votes_prev', 
                'dem_share': 'dem_share_prev', 'rep_share': 'rep_share_prev', 'oth_share': 'oth_share_prev',
                'registered_voters': 'registered_voters_prev', 'turnout_pct': 'turnout_pct_prev',
                'partisan_temp': 'partisan_temp_prev', 'pedersen_index_percent': 'pedersen_index_percent_prev',
                'pedersen_index': 'pedersen_index_prev',
            })
    
            # Make sure these are 13-char left-padded strings
            df_curr['standardized_id_num'] = df_curr['standardized_id_num'].astype(str).str.zfill(13)
            df_prev['standardized_id_num'] = df_prev['standardized_id_num'].astype(str).str.zfill(13)
            df_prev_prev['standardized_id_num'] = df_prev_prev['standardized_id_num'].astype(str).str.zfill(13)
    
            # Merge previous and current
            df_prev_merged = pd.merge(df_prev_prev, df_prev, on="standardized_id_num", how="left") # inner?
            df_all_merged = pd.merge(df_curr, df_prev_merged, on="standardized_id_num", how="left") # inner?
    
            # Compute share changes between previous two elections
            # This represents "seen" data that can be added to the features of upcoming cycles.
            df_all_merged["dem_share_change_prev"] = df_all_merged["dem_share_prev"] - df_all_merged["dem_share_prev_prev"]
            df_all_merged["rep_share_change_prev"] = df_all_merged["rep_share_prev"] - df_all_merged["rep_share_prev_prev"]
            df_all_merged["oth_share_change_prev"] = df_all_merged["oth_share_prev"] - df_all_merged["oth_share_prev_prev"]
            # Compute share changes between this election and the previous election
            # This represents "unseen" data that is NOT seen in data for future predictions.
            df_all_merged["dem_share_change_curr"] = df_all_merged["dem_share"] - df_all_merged["dem_share_prev"]
            df_all_merged["rep_share_change_curr"] = df_all_merged["rep_share"] - df_all_merged["rep_share_prev"]
            df_all_merged["oth_share_change_curr"] = df_all_merged["oth_share"] - df_all_merged["oth_share_prev"]
    
            # Compute total changes between previous two elections
            # This represents "seen" data that can be added to the features of upcoming cycles.
            df_all_merged["dem_votes_change_prev"] = df_all_merged["dem_votes_prev"] - df_all_merged["dem_votes_prev_prev"]
            df_all_merged["rep_votes_change_prev"] = df_all_merged["rep_votes_prev"] - df_all_merged["rep_votes_prev_prev"]
            df_all_merged["oth_votes_change_prev"] = df_all_merged["oth_votes_prev"] - df_all_merged["oth_votes_prev_prev"]
            # Compute total changes between this election and the previous election
            # This represents "unseen" data that is NOT seen in data for future predictions.
            df_all_merged["dem_votes_change_curr"] = df_all_merged["dem_votes"] - df_all_merged["dem_votes_prev"]
            df_all_merged["rep_votes_change_curr"] = df_all_merged["rep_votes"] - df_all_merged["rep_votes_prev"]
            df_all_merged["oth_votes_change_curr"] = df_all_merged["oth_votes"] - df_all_merged["oth_votes_prev"]
    
            # Compute partisan temperature, current and previous.
            df_all_merged["partisan_temp_change_prev"] = df_all_merged["partisan_temp_prev"] - df_all_merged["partisan_temp_prev_prev"] 
            df_all_merged["partisan_temp_change_curr"] = df_all_merged["partisan_temp"] - df_all_merged["partisan_temp_prev"] # "unseen" future data
            
            # Compute turnout between previous two elections
            # This represents "seen" data that can be added to the features of upcoming cycles.
            df_all_merged["registered_voters_change_prev"] = df_all_merged["registered_voters_prev"] - df_all_merged["registered_voters_prev_prev"]
            df_all_merged["turnout_pct_change_prev"] = df_all_merged["turnout_pct_prev"] - df_all_merged["turnout_pct_prev_prev"]
            # Compute turnout between this election and the previous election
            # This represents "unseen" data that is NOT seen in data for future predictions.
            df_all_merged["registered_voters_change_curr"] = df_all_merged["registered_voters"] - df_all_merged["registered_voters_prev"]
            df_all_merged["turnout_pct_change_curr"] = df_all_merged["turnout_pct"] - df_all_merged["turnout_pct_prev"]
    
            # Pederson
            df_all_merged['pedersen_index'] = df_all_merged.apply(pedersenIndexRow, axis=1)
            df_all_merged['pedersen_index_percent'] = df_all_merged['pedersen_index'] * 100
    
            # Drop rows without matches, no way to compute changes.
            df_all_merged = df_all_merged.dropna(subset=['dem_share_prev', 'rep_share_prev', 'oth_share_prev'], how='all')
    
            # Save as parquet for efficiency
            df_all_merged.to_parquet(f"data/generated_data/df_02_vote_changes_calc_{year}_{office.replace('.', '').replace(' ', '_')}.parquet", index=False)
            
            # Free memory
            # del df_prev_prev, df_prev, df_curr, df_all_merged
            gc.collect()
    
            print('---------------')
        print('=============')
    print('#############')


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
            
            df_precinct_original = gpd.read_file('data/generated_data/df_01_election_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.geojson', driver='GeoJSON')
    
            # Map 2020+ standard ids to older standard ids
            if int(year) >= 2020:
                df_precinct_original = remapStandardId(df_precinct_original)
            
            # No change for first and second cycle.
            if year <= YEARS[1]:
                df_precinct_original['dem_share_change_prev'] = 0.
                df_precinct_original['rep_share_change_prev'] = 0.
                df_precinct_original['oth_share_change_prev'] = 0.
                df_precinct_original['dem_votes_change_prev'] = 0.
                df_precinct_original['rep_votes_change_prev'] = 0.
                df_precinct_original['oth_votes_change_prev'] = 0.
                df_precinct_original['registered_voters_change_prev'] = 0.
                df_precinct_original['turnout_pct_change_prev'] = 0.
                df_precinct_original['partisan_temp_change_prev'] = 0.
                df_precinct_original['pedersen_index_prev'] = 0.
                df_precinct_original['pedersen_index_percent_prev'] = 0.
                df_precinct_original['dem_share_change_curr'] = 0.
                df_precinct_original['rep_share_change_curr'] = 0.
                df_precinct_original['oth_share_change_curr'] = 0.
                df_precinct_original['dem_votes_change_curr'] = 0.
                df_precinct_original['rep_votes_change_curr'] = 0.
                df_precinct_original['oth_votes_change_curr'] = 0.
                df_precinct_original['registered_voters_change_curr'] = 0.
                df_precinct_original['turnout_pct_change_curr'] = 0.
                df_precinct_original['partisan_temp_change_curr'] = 0.
                df_precinct_original.to_file('data/generated_data/df_02_vote_changes_calc_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') +'.geojson')
                continue
        
            df_precinct_change_prev = pd.read_parquet('data/generated_data/df_02_vote_changes_calc_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') +'.parquet')
    
            cols_to_extract = [
                'standardized_id_num', 'dem_share_prev', 'rep_share_prev', 'oth_share_prev', 
                'dem_share_change_prev', 'rep_share_change_prev', 'oth_share_change_prev', 
                'dem_share_change_curr', 'rep_share_change_curr', 'oth_share_change_curr', 
                'dem_votes_change_prev', 'rep_votes_change_prev', 'oth_votes_change_prev', 
                'dem_votes_change_curr', 'rep_votes_change_curr', 'oth_votes_change_curr', 
                'registered_voters_change_prev', 'turnout_pct_change_prev',
                'registered_voters_change_curr', 'turnout_pct_change_curr',
                'partisan_temp', 'partisan_temp_prev',
                'partisan_temp_change_prev', 'partisan_temp_change_curr',
                'pedersen_index', 'pedersen_index_percent',
                'pedersen_index_prev', 'pedersen_index_percent_prev',
            ]
    
            # Remove pedersen if not present.
            if ('pedersen_index_prev' not in df_precinct_change_prev.columns) & ('pedersen_index_percent_prev' not in df_precinct_change_prev.columns):
                cols_to_extract.remove('pedersen_index_prev')
                cols_to_extract.remove('pedersen_index_percent_prev')
    
            df_precinct_change_prev = df_precinct_change_prev[cols_to_extract]
            
            df_precinct_original['standardized_id_num'] = df_precinct_original['standardized_id_num'].astype(str).str.zfill(13)
            df_precinct_change_prev['standardized_id_num'] = df_precinct_change_prev['standardized_id_num'].astype(str).str.zfill(13)
            
            df_precinct_results = pd.merge(df_precinct_original, df_precinct_change_prev, on='standardized_id_num', how='left')
            
            df_precinct_results = df_precinct_results.rename(columns={'partisan_temp_x': 'partisan_temp'})
            df_precinct_results = df_precinct_results.drop(columns=['partisan_temp_y'])
            df_precinct_results.to_file('data/generated_data/df_02_vote_changes_calc_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') +'.geojson', driver='GeoJSON')


# ### Missing historical data in 3rd decade.
# <p>Ward numbers are not recorded properly across time. For example, precinct 171 in detroit is the only precinct numbered 171, 
# <br>but in 2018 that precinct was recorded as within Ward 0, while in 2020 that precinct was record within Ward 5. This gives a
# <br>standard id of 1632200000171 and 1632200005171 respectively. So the vote/share changes cannot be counted, because
# <br>these precincts aren't joined in the data due to the different ids.</p>
# <p>Another example is standard id 0013686000001. Before 2020 it is 0013686001001 because that precinct 001 was listed under 
# <br>Ward 01, but on and after 2020 it is listed under Ward 0 with a standard id of 0013686000001. Similarly, not vote/share 
# <br>changes can be computed.</p>
# 
# **Maybe dropna() at prediction, etc. and not above.**

# In[ ]:


# Sanity check that all rows have historical data
# except for first two cycles.
print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')
    
    dfs = {}
    
    for year in YEARS:
        dfs[year] = {}
        
        if int(year) >= 2020:
            for office in OFFICES:
                office = office.replace(' ', '_').replace('.', '')
                df = gpd.read_file(f'data/generated_data/df_02_vote_changes_calc_{year}_{office}.geojson')

                if 'dem_share_prev' in df.columns:
                    print(f'Missing dem_share_prev in {year} for {office}: {len(df[df['dem_share_prev'].isna()])}')
                    print(f'Missing rep_share_prev in {year} for {office}: {len(df[df['dem_share_prev'].isna()])}')
                    print(f'Missing oth_share_prev in {year} for {office}: {len(df[df['dem_share_prev'].isna()])}')
                    print('----------------------------------')
                else:
                    print(f'Missing dem_share_prev column for office {office} in year {year}')
    
                dfs[year][office] = df


# In[ ]:




