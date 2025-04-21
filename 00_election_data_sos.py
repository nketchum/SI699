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
ELECTIONS['U.S. Senate'] =  ['2014', '2018', '2020', '2024']
ELECTIONS['State Senate'] = ['2014', '2018', '2022']
ELECTIONS['President'] =    ['2016', '2020']


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import geopandas as gpd
import pandas as pd
import numpy as np
import gc


# In[ ]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# ### Helper functions

# In[ ]:


def getOfficeName(office_code):
    offices = {
        1: 'President',
        2: 'Governor',
        3: 'Secretary of State',
        4: 'Attorney General',
        5: 'U.S. Senator',
        6: 'U.S. House',
        7: 'State Senate',
        8: 'State House',
        9: 'State Board of Education',
        10: 'University of Michigan Board of Regents',
        11: 'Michigan State University Board of Trustees',
        12: 'Wayne State University Board of Governors',
        13: 'Supreme Court',
        90: 'Statewide Ballot Proposals',
    }
    return offices[office_code]


# In[ ]:


def getOfficeCode(office_name):
    offices = {
        "President": 1,
        "Governor": 2,
        "Secretary of State": 3,
        "Attorney General": 4,
        "U.S. Senator": 5,
        "U.S. Senate": 5, # alternate nomenclature
        "U.S. House": 6,
        "State Senate": 7,
        "State House": 8,
        "State Board of Education": 9,
        "University of Michigan Board of Regents": 10,
        "Michigan State University Board of Trustees": 11,
        "Wayne State University Board of Governors": 12,
        "Supreme Court": 13,
        "Statewide Ballot Proposals": 90
    }
    return offices[office_name]


# ### Load/merge data functions

# In[ ]:


def loadCityData(year):
    print(f'Loading city data for {year}...')
    if year == '2024':
        year = '2022' # No 2024 data yet
    city_headers = pd.read_csv('data/elections-data-mi-sos/headers/city_headers.csv', header=None).squeeze().tolist()
    df_city = pd.read_csv(f'data/elections-data-mi-sos/{year}GEN/{year}city.txt', sep='\t', header=None, names=city_headers, index_col=False)
    df_city.columns = df_city.columns.str.strip()
    df_city = df_city.rename(columns={'County Code': 'Michigan County Code'})
    print(f'Done.')
    return df_city


# In[ ]:


def loadCountyData(year):
    print(f'Loading county data for {year}...')
    if year == '2024':
        year = '2022' # No 2024 data yet
    county_headers = pd.read_csv('data/elections-data-mi-sos/headers/county_headers.csv', header=None).squeeze().tolist()
    df_county = pd.read_csv(f'data/elections-data-mi-sos/{year}GEN/county.txt', sep='\t', header=None, names=county_headers, index_col=False)
    df_county.columns = df_county.columns.str.strip()
    # Map the US Census county code to the Michigan county code.
    df_county_codes_census = pd.read_csv('data/custom_data/county_code_mapping.csv')
    df_county_codes_census = df_county_codes_census[['Census County Code', 'Michigan County Code']]
    df_county_codes_census = df_county_codes_census.rename(columns={'Michigan County Code': 'County Code'})
    df_county= pd.merge(df_county, df_county_codes_census, on="County Code", how="inner")
    df_county = df_county.rename(columns={'County Code': 'Michigan County Code'})
    print(f'Done')
    return df_county


# In[ ]:


def loadNameData(year):
    print(f'Loading candidate name data for {year}...')
    if year == '2024':
        year = '2022' # No 2024 data yet
    # Candidate ID are negative for nationwide offices.
    name_headers = pd.read_csv('data/elections-data-mi-sos/headers/name_headers.csv', header=None).squeeze().tolist()
    df_name = pd.read_csv(f'data/elections-data-mi-sos/{year}GEN/{year}name.txt', sep='\t', header=None, names=name_headers, index_col=False)
    df_name.columns = df_name.columns.str.strip()
    df_name["Candidate Middle Name"] = df_name["Candidate Middle Name"].fillna("")
    df_name["Candidate Party Name"] = df_name["Candidate Party Name"].fillna("")
    print(f'Done')
    return df_name


# In[ ]:


def loadOffcData(year):
    print(f'Loading office data for {year}...')
    if year == '2024':
        year = '2022' # No 2024 data yet
    offc_headers = pd.read_csv('data/elections-data-mi-sos/headers/offc_headers.csv', header=None).squeeze().tolist()
    df_offc = pd.read_csv(f'data/elections-data-mi-sos/{year}GEN/{year}offc.txt', sep='\t', header=None, names=offc_headers, index_col=False)
    df_offc.columns = df_offc.columns.str.strip()
    print(f'Done')
    return df_offc


# In[ ]:


def loadVoteData(year):
    print(f'Loading vote data for {year}...')
    if year == '2024':
        year = '2022' # No 2024 data yet
    vote_headers = pd.read_csv('data/elections-data-mi-sos/headers/vote_headers.csv', header=None).squeeze().tolist()
    df_vote = pd.read_csv(f'data/elections-data-mi-sos/{year}GEN/{year}vote.txt', sep='\t', header=None, names=vote_headers, index_col=False)
    df_vote.columns = df_vote.columns.str.strip()
    df_vote = df_vote.rename(columns={'County Code': 'Michigan County Code', 'City/Town Code': 'City/Township Code'})
    df_vote["Precinct Label"] = df_vote["Precinct Label"].fillna("")
    print(f'Done')
    return df_vote


# In[ ]:


def loadPrecinctData(YEARS):
    precinct_dfs = []

    for year in YEARS:
        print(f'Loading precinct data for {year}...')
        
        # TIGER/Line shapefiles DO NOT include precinct-level data, but 
        # the city/township names are much more standardized that those on 
        # ArcGIS, and TIGER/Line goes back to 2007.
        # However, the ArcGIS data DOES include precinct-level data, but
        # the city/township names are a mess. Yet, there is a standard id on those,
        # though ArcGIS only goes back to 2014.
        #
        # We will use TIGER/Line to match the voting info with precinct info to 
        # construct the standard WP ids, and then after that match the ArcGIS
        # files so we can get down to the precinct level.
        filename = f'data/voting_precincts/tl/tl_{year}_26_cousub/tl_{year}_26_cousub.shp'
        df_precinct = gpd.read_file(filename)
        precinct_dfs.append(df_precinct)

        print('Done')

    return precinct_dfs


# In[ ]:


def mergeRegionData(df_county, df_city):
    print(f'Merging counties and cities...')
    # County + City = Region
    df_region = pd.merge(df_county, df_city, on="Michigan County Code", how="inner")
    print('Done')
    return df_region


# In[ ]:


def mergeCampaignData(df_name, df_offc):
    print(f'Merging candidate and office (campaign) data...')
    # Name + Office = Campaign
    df_name_temp = df_name.copy()
    df_name_temp = df_name_temp.drop(columns=['Election Year', 'Election Type'])  # Remove duplicate cols
    df_campaign = pd.merge(df_name_temp, df_offc, on=["Office Code", "District Code", "Status Code"], how="inner")
    print('Done')
    return df_campaign


# In[ ]:


def mergeOutcomeData(df_vote, df_campaign):
    print(f'Merging vote data...')
    # Vote + Campaign = Outcome
    df_vote_temp = df_vote.copy()
    df_vote_temp = df_vote_temp.drop(columns=['Election Year', 'Election Type', 'Office Code', 'District Code', 'Status Code'])  # Remove duplicate cols
    df_outcome = pd.merge(df_vote_temp, df_campaign, on="Candidate ID", how="inner")
    print('Done')
    return df_outcome


# In[ ]:


def mergeElectionData(df_outcome, df_region):
    print(f'Merging election data...')
    # Outcome + Region = Election
    df_outcome_temp = df_outcome.copy()
    df_outcome_temp = df_outcome_temp.drop(columns=["Election Year", "Election Type"])
    df_election = pd.merge(df_outcome_temp, df_region, on=["Michigan County Code", "City/Township Code"], how="inner")
    print('Done')
    return df_election


# In[ ]:


def cleanPlaceNames(df_election, df_precinct):
    print(f'Cleaning place names...')
    
    df_precinct['NAMELSAD'] = df_precinct['NAMELSAD'].apply(lambda x: x.upper().replace('CHARTER ', ''))
    df_precinct['NAMELSAD'] = df_precinct['NAMELSAD'].apply(lambda x: x.replace('CITY CITY', 'CITY'))

    names_to_clean = {
        'DE WITT': 'DEWITT', # Includes city & township
        'CLARKSTON CITY': 'VILLAGE OF CLARKSTON CITY',
        'COLD SPRINGS TOWNSHIP': 'COLDSPRINGS TOWNSHIP',
        'GUNPLAIN TOWNSHIP': 'GUN PLAIN TOWNSHIP',
        'LA GRANGE TOWNSHIP': 'LAGRANGE TOWNSHIP',
        'LANSE TOWNSHIP': "L'ANSE TOWNSHIP",
        'PLEASANT VIEW TOWNSHIP': 'PLEASANTVIEW TOWNSHIP',
        'ST JAMES TOWNSHIP': 'ST. JAMES TOWNSHIP',
    }

    for orig_name, clean_name in names_to_clean.items():
        df_election["City/Township Description"] = df_election["City/Township Description"].apply(lambda x: x.replace(orig_name, clean_name)) 

    print('Done')
    
    return df_election, df_precinct


# In[ ]:


def mergeTigerLine(df_election, df_precinct):
    print(f'Merging Tiger Line boundaries...')
    df_election_precinct = pd.merge(df_election, df_precinct, left_on=["Census County Code", "City/Township Description"], right_on=["COUNTYFP", "NAMELSAD"], how="left")
    print('Done')
    return df_election_precinct


# In[ ]:


def makeStandardIds(df_election_precinct):
    print(f'Making standard ids...')
    df_election_precinct['standardized_id'] = df_election_precinct.apply(lambda row: 'WP-' + str(row['COUNTYFP']).zfill(3) + '-' + str(row['COUSUBFP']).zfill(5) + '-' + str(row['Ward Number']).zfill(2) + str(row['Precinct Number']).zfill(3), axis=1)
    df_election_precinct['standardized_id_num'] = df_election_precinct.apply(lambda row: str(row['COUNTYFP']) + str(row['COUSUBFP']).zfill(5) + str(row['Ward Number']).zfill(2) + str(row['Precinct Number']).zfill(3), axis=1)
    df_election_precinct['standardized_id_num'] = df_election_precinct['standardized_id_num'].astype(str).str.zfill(13)
    print('Done')
    return df_election_precinct


# In[ ]:


def mergeArcGIS(df, year):
    print(f'Merging ArcGIS data...')
    
    columns_to_keep = ['Candidate ID', 'Michigan County Code', 'City/Township Code', 
                       'Ward Number', 'Precinct Number', 'Precinct Label', 'Precinct Votes', 
                       'Office Code', 'District Code', 'Status Code', 'Candidate Last Name', 
                       'Candidate First Name', 'Candidate Middle Name', 'Candidate Party Name', 
                       'Office Description', 'County Name', 'Census County Code', 
                       'Election Year', 'Election Type', 'City/Township Description', 'GEOID', 
                       'standardized_id', 'standardized_id_num', 'geometry']

    # Some cities are not properly recorded, such as missing wards.
    messy_cities = ['DETROIT CITY', 'LIVONIA CITY', 'BATTLE CREEK CITY', 'ROCHESTER HILLS CITY', 
                    'NILES CITY', 'BENTON HARBOR CITY', 'IONIA CITY', 'GRAND LEDGE CITY',
                    'ST. JOHNS CITY', 'LAPEER CITY', 'NORTON SHORES CITY', 'ST. LOUIS CITY',
                    'HOLLAND CITY', 'GLADWIN CITY']

    messy_cities_dfs = []
    for messy_city in messy_cities:
        df_messy_city = df[df['City/Township Description'] == messy_city][columns_to_keep]
        messy_cities_dfs.append(df_messy_city)

    df_precinct_arcgis = gpd.read_file("data/voting_precincts/" + str(year) + "_Voting_Precincts.geojson")

    year = int(year)
    if year == 2016:
        df_precinct_arcgis.rename(columns={'VTD2016': 'precinct_id'}, inplace=True)
    elif year == 2014:
        df_precinct_arcgis.rename(columns={'VP': 'precinct_id'}, inplace=True)
    elif year > 2016:
        df_precinct_arcgis['precinct_id'] = df_precinct_arcgis['PRECINCTID'].apply(lambda x: x.replace('WP', '').replace('-',''))
    else:
        raise Exception('Too far in the past, ArcGIS shapefiles start in 2014')

    df = df[['Candidate ID', 'Michigan County Code', 'City/Township Code', 
             'Ward Number', 'Precinct Number', 'Precinct Label', 'Precinct Votes', 
             'Office Code', 'District Code', 'Status Code', 'Candidate Last Name', 
             'Candidate First Name', 'Candidate Middle Name', 'Candidate Party Name', 
             'Office Description', 'County Name', 'Census County Code', 
             'Election Year', 'Election Type', 'City/Township Description', 'GEOID', 
             'standardized_id', 'standardized_id_num']]

    df_final = pd.merge(df, df_precinct_arcgis, left_on='standardized_id_num', right_on='precinct_id', how='inner')
    
    # Ensure GeoDataFrames before CRS conversion
    df_final = gpd.GeoDataFrame(df_final, geometry="geometry")
    df_final = df_final.to_crs(epsg=4326)

    for df_messy_city in messy_cities_dfs:
        df_messy_city = gpd.GeoDataFrame(df_messy_city, geometry="geometry")
        df_messy_city = df_messy_city.to_crs(epsg=4326)
        df_final = pd.concat([df_final, df_messy_city], ignore_index=True)

    # Convert back into geodf.
    df_final = gpd.GeoDataFrame(df_final, geometry="geometry")

    print('Done')

    return df_final


# In[ ]:


# TigerLine Shapefiles
# https://www.census.gov/programs-surveys/geography/technical-documentation/complete-technical-documentation/tiger-geo-line.html

def makeElections(YEARS, OFFICES):
    print(f'Making elections...')
    
    election_dfs = {}
    
    for year in YEARS:
        print(f'Processing year {year}...')
        
        for office in OFFICES:
            office_code = getOfficeCode(office)
            print(f'Processing office {office}...')
            
            # Load each component
            df_city = loadCityData(year)
            df_county = loadCountyData(year)
            df_name = loadNameData(year)
            df_offc = loadOffcData(year)
            df_vote = loadVoteData(year)
    
            # Filter by office
            if office_code != None:
                df_name = df_name[df_name['Office Code'] == office_code]
                df_offc = df_offc[df_offc['Office Code'] == office_code]
                df_vote = df_vote[df_vote['Office Code'] == office_code]
    
            # Region = County + City
            df_region = mergeRegionData(df_county, df_city)
    
            # Campaign = Name + Office
            df_campaign = mergeCampaignData(df_name, df_offc)
    
            # Outcome = Campaign + Votes
            df_outcome = mergeOutcomeData(df_vote, df_campaign)
    
            # Election = Outcomes + Region
            df_election = mergeElectionData(df_outcome, df_region)
    
            # Drop non-location names
            df_election = df_election[df_election["City/Township Description"] != "{Statistical Adjustments}"]
    
            # Convert all columns to str
            df_election = df_election.astype(str)
            df_election['Census County Code'] = df_election['Census County Code'].apply(lambda x: x.zfill(3)) # always 3 digits
    
            # Clean up
            df_precinct = loadPrecinctData([year])[0] # This takes a list of years, but we're already looping thru years here.
            df_election, df_precinct = cleanPlaceNames(df_election, df_precinct) # We need targetline precinct here

            # Join TigerLine municipal data
            df = mergeTigerLine(df_election, df_precinct)

            # Each precinct needs a reliable id.
            df = makeStandardIds(df)

            # Join ArcGIS precinct data
            df = mergeArcGIS(df, year)
            
            # Save to disk
            filename = f'df_election_{year}{('_' + office.replace('.', '').replace(' ', '_')) if office_code != None else ''}.csv'
            df.to_file(f'data/generated_data/df_00_election_{year}_{office.replace('.', '').replace(' ', '_')}.geojson', driver='GeoJSON')
    
            # Add to output
            election_dfs[year] = df

            # Cleanup
            del(df)
            gc.collect()

    print('Done')

    return election_dfs


# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    election_dfs = makeElections(YEARS, OFFICES)


# In[ ]:




