#!/usr/bin/env python
# coding: utf-8

# ### Globals

# In[ ]:


# President: 2016 (Trump), 2020 (Biden), 2024 (Trump)
# U.S. Senate: 2014 (Peters), 2018 (Stabenow), 2020 (Peters), 2024 (Slotkin)
# U.S. House: every cycle
# State Senate: 2014, 2018, 2022
# State House: every cycle

# Do NOT include year 2024; use the openelections dataset instead.

ELECTIONS = {}

ELECTIONS['U.S. House'] =   ['2014', '2016', '2018', '2020', '2022']
ELECTIONS['State House'] =  ['2014', '2016', '2018', '2020', '2022']
ELECTIONS['U.S. Senate'] =  ['2014', '2018', '2020']
ELECTIONS['State Senate'] = ['2014', '2018', '2022']
ELECTIONS['President'] =    ['2016', '2020']


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import geopandas as gpd
import pandas as pd
import numpy as np


# In[ ]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# In[ ]:


def getPartyName(party):
    party_clean = ''

    if party in ['DEM', 'REP']:
        party_clean = party
    else:
        party_clean = 'OTH'

    return party_clean


# ### Load/merge data functions

# In[ ]:


def loadArcGIS(year, office):
    print(f'Loading ArcGIS data for year {year} and office {office.replace(' ', '_').replace('.', '')}...')
    df = gpd.read_file(f'data/generated_data/df_00_election_{year}_{office.replace('.', '').replace(' ', '_')}.geojson')
    
    # Don't convert "Precinct Votes" leave as int64.
    columns_to_convert = ['Candidate ID', 'Michigan County Code', 'City/Township Code', 
                          'Ward Number', 'Precinct Number', 'Precinct Label', 
                          'Office Code', 'District Code', 'Status Code', 'Candidate Last Name', 
                          'Candidate First Name', 'Candidate Middle Name', 'Candidate Party Name', 
                          'Office Description', 'County Name', 'Census County Code', 
                          'Election Year', 'Election Type', 'City/Township Description']
    
    df[columns_to_convert] = df[columns_to_convert].astype(str)
    df['Precinct Votes'] = df['Precinct Votes'].astype(float)
    df['Precinct Votes'] = df['Precinct Votes'].astype(int)
    
    df['Party Name'] = df['Candidate Party Name'].apply(getPartyName)

    print(f'Done.')
    
    return df


# In[ ]:


def aggCandidates(df):
    print(f'Aggregating candidate data...')
    
    columns_to_aggregate = ['Precinct Votes']
    grouping_columns = ['standardized_id_num', 'Candidate ID']
    
    agg_funcs = {col: 'sum' for col in columns_to_aggregate}
    
    # Use the first value for the same/constant value.
    for col in df.columns:
        if col not in columns_to_aggregate:
            agg_funcs[col] = 'first'
    
    # Use census ids and not michigan ids.
    df = df.groupby(grouping_columns, as_index=False).agg(agg_funcs).reset_index(drop=True)

    print(f'Done.')

    return df


# In[ ]:


def aggParties(df):
    print(f'Aggregating party data...')
    
    columns_to_drop = ['Candidate ID', 'Candidate Last Name', 'Candidate First Name', 'Candidate Middle Name']
    
    df = df.drop(columns=columns_to_drop)
    
    # Extract unique geometry before pivoting
    df_geometry = df[['Michigan County Code', 'City/Township Code', 
                      'Ward Number', 'Precinct Number', 'geometry']].drop_duplicates()
    
    # Define index columns **without geometry**
    index_columns = ['Michigan County Code', 'City/Township Code',
                     'Ward Number', 'Precinct Number', 'Precinct Label', 
                     'Office Code', 'District Code', 'Status Code', 
                     'Office Description', 'County Name', 'Census County Code', 
                     'Election Year', 'Election Type', 'City/Township Description',  
                     'standardized_id', 'standardized_id_num']
    
    # Pivot the data
    df = df.pivot_table(index=index_columns, columns='Party Name', values='Precinct Votes', aggfunc='sum').reset_index()
    
    # Rename columns for clarity
    df.columns.name = None  # Remove the multi-index name
    df = df.rename(columns={'DEM': 'dem_votes', 'REP': 'rep_votes', 'OTH': 'oth_votes'})
    
    # NaNs with 0s (if needed)
    df = df.fillna(0)
    
    # Merge back unique geometry
    df = df.merge(df_geometry, on=['Michigan County Code', 'City/Township Code', 'Ward Number', 'Precinct Number'], how='left')

    # Compute votes
    df['total_votes'] = df.apply(lambda row: row['dem_votes'] + row['rep_votes'] + row['oth_votes'], axis=1)

    # Drop rows where no voting occurred
    df = df[df["total_votes"] != 0]

    # Share of vote
    df['dem_share'] = df.apply(lambda row: row['dem_votes'] / row['total_votes'], axis=1)
    df['rep_share'] = df.apply(lambda row: row['rep_votes'] / row['total_votes'], axis=1)
    df['oth_share'] = df.apply(lambda row: row['oth_votes'] / row['total_votes'], axis=1)

    # Make one number to gauge partisan temperature.
    # Left-wing dems are left of 0 (neg nums). Right-wing reps, right of 0 (pos nums).
    df['partisan_temp'] = (-1 * df['dem_share']) + df['rep_share']

    print('Done.')

    return df


# In[ ]:


def calcTurnout(df):
    print(f'Calculating turnout...')
    
    df_registered_voters = pd.read_csv('data/custom_data/registered_voters_count.csv') # these are 2025 numbers though!
    df_registered_voters = df_registered_voters.rename(columns={'count': 'registered_voters'})
    df = pd.merge(df, df_registered_voters, on="standardized_id_num", how="left")
    df['turnout_pct'] = df['total_votes'] / df['registered_voters']

    print('Done.')
    
    return df


# In[ ]:


def makeFinalElections(YEARS, OFFICES):
    print(f'Making new election data...')
    
    election_dfs = {}
    
    for year in YEARS:
        print(f'Processing year {year}')
        
        for office in OFFICES:
            print(f'Processing office {office}')
            
            df = loadArcGIS(year, office)
            df = aggCandidates(df)
            df = aggParties(df)
            df = calcTurnout(df)

            # Somehow we're getting some duplicate precincts.
            df = df.drop_duplicates(subset=['standardized_id_num'], keep='first')
    
            df = gpd.GeoDataFrame(df, geometry="geometry")
            df.to_file('data/generated_data/df_01_election_' + str(year) + '_' + office.replace('.', '').replace(' ', '_') + '.geojson', driver='GeoJSON')
    
            election_dfs[year] = df
            del(df)

            print('----------------------------')

        print('============================')

    print('Done.')
    
    return election_dfs


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    election_final_dfs = makeFinalElections(YEARS, OFFICES)


# In[ ]:




