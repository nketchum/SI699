#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# President: 2016 (Trump), 2020 (Biden), 2024 (Trump)
# U.S. Senate: 2014 (Peters), 2018 (Stabenow), 2020 (Peters), 2024 (Slotkin)
# U.S. House: every cycle
# State Senate: 2014, 2018, 2022
# State House: every cycle

YEAR = 2024
DAY_OF_NOV = 5  # Changes each year
WRITE_INS = True
PLOT_MAP = False

OFFICES = ['U.S. House', 'State House', 'U.S. Senate', 'President']

for OFFICE in OFFICES:
    
    import warnings
    warnings.filterwarnings("ignore")
    
    from matplotlib.colors import to_rgba
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    
    pd.set_option("display.max_columns", None)

    
    def getPrecinctResults(file_path):
        df = pd.read_csv(file_path)
        return df
    
    
    def showPrecinctResultSummary(df):
        print("OFFICES")
        print(df['office'].unique())
        print("=========================================================")
        print("PARTIES")
        print(df['party'].unique())
        print("=========================================================")
        print("PRESIDENT")
        print(df[df['office'] == 'President'].head())
        print("=========================================================")
        print("STRAIGHT PARTY")
        print(df[df['office'] == 'Straight Party'].head())
        print("=========================================================")
        print("REGISTERED VOTERS")
        print(df[df['office'] == 'Registered Voters'].head())
        print("=========================================================")
        print("BALLOTS CAST")
        print(df[df['office'] == 'Ballots Cast'].head())
        print("=========================================================")
        print("BALLOTS CAST (BLANK)")
        print(df[df['office'] == 'Ballots Cast Blank'].head())
        print("=========================================================")
        print("STATE HOUSE")
        print(df[df['office'] == 'State House'].head())
        print("=========================================================")
        print("STATE SENATE")
        print(df[df['office'] == 'State Senate'].head())
        print("=========================================================")
        print("U.S. HOUSE")
        print(df[df['office'] == 'U.S. House'].head())
        print("=========================================================")
        print("U.S. SENATE")
        print(df[df['office'] == 'U.S. Senate'].head())
    
    
    def getPrecinctBounds(file_path):
        df = gpd.read_file(file_path)
        return df
    
    
    def showPrecinctBoundsSummary(df):
        print("DF.DESCRIBE")
        print(df.describe())
        print("=========================================================")
        print("DF.DTYPES")
        print(df.dtypes)
        print("=========================================================")
        print("DF.iloc[0]")
        print(df.iloc[0])
    
    
    def plotPrecinctBounds(df):
        color_map = {
            'D': np.array([0, 0, 255]),   # Blue
            'R': np.array([255, 0, 0]),   # Red
            'I': np.array([255, 255, 0])  # Yellow
        }
    
        # Weighted sum of RGB components
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.to_rgba.html
        # https://stackoverflow.com/questions/4255973/calculation-of-a-mixed-color-in-rgb
        # https://stackoverflow.com/questions/61488790/how-can-i-proportionally-mix-colors-in-python
        def compute_mixed_color(row):       
            mixed_rgb = (
                row['dem_share'] * color_map['D'] +
                row['rep_share'] * color_map['R'] +
                row['oth_share'] * color_map['I']
            )
            return tuple(mixed_rgb.astype(int) / 255)
    
        df['color'] = df.apply(compute_mixed_color, axis=1)
    
        fig, ax = plt.subplots(figsize=(80, 80))
        divider = make_axes_locatable(ax)
    
        df.boundary.plot(ax=ax, color="black", linewidth=0.1)
        df.plot(ax=ax, color=df['color'], edgecolor="black", linewidth=0.01)
    
        ax.margins(0)
        ax.set_title(str(YEAR) + " Precinct Outcomes for " + OFFICE, fontsize=64)
        ax.set_axis_off()
    
        if WRITE_INS == True:
            filename_writeins = '_writeins'
        else:
            filename_writeins = ''
    
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig('output/maps/precincts/' + str(YEAR) + "_" + OFFICE + filename_writeins + "_Map_OpenElections.png")
        plt.show()
    
    
    def cleanPartyName(party):
        party_clean = ''
    
        # If already coded, leave it.
        if party in ['DEM', 'REP']:
            return party
    
        # If not coded, code it.
        if (party == 'Democrat') | (party == 'D'):
            party_clean = 'DEM'
        elif (party == 'Republican') | (party == 'R'):
            party_clean = 'REP'
        else:
            party_clean = 'OTH'
    
        return party_clean
    
    
    print('Processing 2024 election data...')
    
    
    df_precinct_bounds = getPrecinctBounds("data/voting_precincts/" + str(YEAR) + "_Voting_Precincts.geojson")
    df_precinct_bounds = df_precinct_bounds.drop(columns=[col for col in ['OBJECTID_1', 'Tabulator_Voter_Assist'] if col in df_precinct_bounds.columns])
    
    if YEAR >= 2022:
        # Remove rows without precinct values.
        df_precinct_bounds = df_precinct_bounds.dropna(subset=['Precinct_Long_Name'])
        df_precinct_bounds = df_precinct_bounds[df_precinct_bounds['Precinct_Long_Name'] != "None"]    
    
    # Load county subdivisions and associated codes.
    # Different census data is formatted differently.
    if YEAR < 2020:
        census_year = 2010
        sep = ','
        names = ['STATESTATEFP', 'COUNTYFP', 'COUNTYNAME', 'COUSUBFP', 'COUSUBNAME', 'FUNCSTAT']
    elif YEAR >= 2020:
        census_year = 2020
        sep = '|'
        names = None
    
    df_census_fips = pd.read_csv('data/census/fips/st26_mi_cousub' + str(census_year) + '.txt', sep=sep, names=names)
    
    # Left-pad numbers with 0s so that we can construct the standard Ward-District codes for precincts.
    df_census_fips['COUNTYFP'] = df_census_fips['COUNTYFP'].astype(str).str.zfill(3)
    df_census_fips['COUSUBFP'] = df_census_fips['COUSUBFP'].astype(str).str.zfill(5)
    
    # Map county subdivisions to precinct boundaries
    df_precinct_bounds_named = pd.DataFrame()
    
    if YEAR > 2016:
        df_precinct_bounds_named = pd.merge(df_precinct_bounds, df_census_fips, left_on=['COUNTYFIPS', 'MCDFIPS'], right_on=['COUNTYFP', 'COUSUBFP'], how='left')
    elif YEAR <= 2016:
        df_precinct_bounds_named = pd.merge(df_precinct_bounds, df_census_fips, left_on=['CountyFips', 'Jurisdicti'], right_on=['COUNTYFP', 'COUSUBFP'], how='left')
        
    df_precinct_bounds_named.drop(['COUNTYFP', 'COUSUBFP'], axis=1, inplace=True)
    
    if YEAR == 2020:
        df_precinct_bounds_named.drop(['STATE', 'FUNCSTAT_x', 'STATEFP_y', 'COUSUBNS', 'FUNCSTAT_y'], axis=1, inplace=True)
    elif YEAR == 2024:
        df_precinct_bounds_named.drop(['STATE', 'COUSUBNS'], axis=1, inplace=True)
    
    df_precinct_bounds_named.rename(columns={'STATEFP_x': 'STATEFP'}, inplace=True)
    
    # Construct a human-readable precinct name to allow
    # joining with the open election voting data which omits
    # well-structured, standardized identification of precincts.
    if YEAR > 2016:
        df_precinct_bounds_named['WARD'] = df_precinct_bounds_named['WARD'].astype(str).str.strip()
        df_precinct_bounds_named['PRECINCT'] = df_precinct_bounds_named['PRECINCT'].astype(str).str.strip()
        df_precinct_bounds_named['precinct_name'] = df_precinct_bounds_named.apply(lambda row: 
        f"{row['COUSUBNAME']}" +
        (f", Ward {row['WARD'].lstrip('0')}" if row['WARD'] != "00" else "") +
        f", Precinct {row['PRECINCT'].lstrip('0')}",
        axis=1)
    elif YEAR == 2016:
        vtd_column_name = 'VTD' + str(YEAR)
        df_precinct_bounds_named.rename(columns={vtd_column_name: 'VTD'}, inplace=True)
        df_precinct_bounds_named['WARD'] = df_precinct_bounds_named['VTD'].apply(lambda x: str(x)[-5:-3])
        df_precinct_bounds_named['PRECINCT'] = df_precinct_bounds_named['VTD'].apply(lambda x: str(x)[-3:])
    elif YEAR == 2014:
        df_precinct_bounds_named.rename(columns={'VP': 'VTD'}, inplace=True)
        df_precinct_bounds_named['WARD'] = df_precinct_bounds_named['VTD'].apply(lambda x: str(x)[-5:-3])
        df_precinct_bounds_named['PRECINCT'] = df_precinct_bounds_named['VTD'].apply(lambda x: str(x)[-3:])
    
    if YEAR <= 2016:
        df_precinct_bounds_named['WARD'] = df_precinct_bounds_named['WARD'].astype(str).str.zfill(2)
        df_precinct_bounds_named['PRECINCT'] = df_precinct_bounds_named['PRECINCT'].astype(str).str.zfill(3)

    df_precinct_results = getPrecinctResults("data/openelections-data-mi/" + str(YEAR) + "/" + str(YEAR) + "11" + str(DAY_OF_NOV).zfill(2) + "__mi__general__precinct.csv")
    
    # Tool: https://regex101.com/
    def clean_column(value):
        # Remove "AVCB" anywhere in the string
        value = re.sub(r'\bAVCB\b', '', value, flags=re.IGNORECASE)
        # Remove hyphen + uppercase letters
        value = re.sub(r'\s*-\s*[A-Z]+\b', '', value)
        # Remove text inside parentheses at end
        value = re.sub(r'\s*\([^)]*\)$', '', value)
        # Remove three-letter uppercase endings after  last space
        value = re.sub(r'\s+[A-Z]{3}$', '', value)
        return value.strip() # Remove trailing spaces
    
    df_precinct_results_cleaned = df_precinct_results.copy()
    
    # Make voting-related columns numerical only.
    df_precinct_results_cleaned['votes'] = df_precinct_results_cleaned['votes'].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)
    if YEAR == 2024:
        cols_to_convert = ['votes', 'election_day', 'absentee', 'av_counting_boards',
                       'early_voting', 'mail', 'provisional', 'pre_process_absentee']
    elif (YEAR == 2022) | (YEAR == 2020):
        cols_to_convert = ['votes', 'election_day', 'absentee']
    else:
        cols_to_convert = ['votes']
    
    df_precinct_results_cleaned[cols_to_convert] = df_precinct_results_cleaned[cols_to_convert].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    
    df_precinct_results_cleaned['precinct_original'] = df_precinct_results_cleaned['precinct']
    df_precinct_results_cleaned['precinct'] = df_precinct_results_cleaned['precinct'].apply(clean_column)
    df_precinct_results_cleaned['district'] = df_precinct_results_cleaned['district'].fillna(0)
    df_precinct_results_cleaned['party'] = df_precinct_results_cleaned['party'].fillna('')
    df_precinct_results_cleaned['candidate'] = df_precinct_results_cleaned['candidate'].fillna('')
    if OFFICE != 'Straight Party':
        # Remove rows with blank candidates.
        df_precinct_results_cleaned = df_precinct_results_cleaned[df_precinct_results_cleaned['candidate'].notna() & (df_precinct_results_cleaned['candidate'].astype(str).str.strip() != "")]
    
    df_precinct_results_office = df_precinct_results_cleaned[df_precinct_results_cleaned['office'] == OFFICE]
    df_precinct_results_office['party_clean'] = df_precinct_results_office['party'].apply(cleanPartyName)
    
    if len(df_precinct_results_office) == 0:
        raise TypeError("No results for office " + str(OFFICE) + " for the " + str(YEAR) + " cycle.")
    
    
    # ONE-OFF CORRECTIONS
    if YEAR == 2024:
        # For counties below, "early voting" column is bad. Each cell contains a constant number that throws off the whole county.
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Ionia', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Lapeer', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Livingston', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Oceana', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Oakland', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Menominee', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Missaukee', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Washtenaw', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Tuscola', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'St. Clair', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Sanilac', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Marquette', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Mackinac', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Luce', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Keweenaw', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Houghton', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Grand Traverse', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Gladwin', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Crawford', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Chippewa', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Cheboygan', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Cass', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Benzie', 'early_voting'] = 0
        df_precinct_results_office.loc[df_precinct_results_office['county'] == 'Baraga', 'early_voting'] = 0
    
    # Sum up all votes from all vote types into the "votes" column.
    if YEAR == 2024:
        df_precinct_results_office['votes'] = df_precinct_results_office['votes'] + df_precinct_results_office[['election_day', 'absentee', 'av_counting_boards', 'early_voting', 'mail', 'provisional', 'pre_process_absentee']].sum(axis=1)
    elif YEAR == 2022:
        df_precinct_results_office['votes'] = df_precinct_results_office['votes'] + df_precinct_results_office[['election_day', 'absentee']].sum(axis=1)
    
    
    if WRITE_INS == False:
        non_candidates = ['absentee', 'Ballots Cast', 'Blank', 'Blank (W)', 'Cast Votes', 'election_day',
                          'Not Assigned', 'Over Vote Count', 'Over Votes', 'Overvotes', 'Registered Voters', 
                          'Rejected write-in votes', 'Rejected write-ins', 'Total Votes', 'Total Write-In', 
                          'Turnout Pct', 'Unassigned write-ins', 'Under Vote Count', 'Under Votes', 'Undervotes', 
                          'Unqualified Write-Ins', 'Unresolved write-in votes:', 'Write-In', 'write-in', 
                          'Write-In Totals', 'Write-ins', 'Write-Ins', 'Write_ins', 'Yes']
    elif WRITE_INS == True:
        non_candidates = ['absentee', 'Ballots Cast', 'Cast Votes', 'election_day', 'Over Vote Count', 
                          'Over Votes', 'Overvotes', 'Registered Voters', 'Total Votes', 'Total Write-In', 
                          'Turnout Pct', 'Under Vote Count', 'Under Votes', 'Undervotes', 'Write-In Totals', 'Yes']
    
    df_precinct_results_office = df_precinct_results_office[~df_precinct_results_office['candidate'].isin(non_candidates)]
    
    
    # Some votes are in string format, which is quite bad.
    df_precinct_results_office['dem_votes'] = None
    df_precinct_results_office.loc[df_precinct_results_office['party_clean'] == 'DEM', 'dem_votes'] = df_precinct_results_office['votes']
    
    df_precinct_results_office['rep_votes'] = None
    df_precinct_results_office.loc[df_precinct_results_office['party_clean'] == 'REP', 'rep_votes'] = df_precinct_results_office['votes']
    
    df_precinct_results_office['oth_votes'] = None
    df_precinct_results_office.loc[(df_precinct_results_office['party_clean'] != 'DEM') & (df_precinct_results_office['party_clean'] != 'REP'), 'oth_votes'] = df_precinct_results_office['votes']
    
    # Aggregate results by party
    columns_to_convert = ['votes', 'dem_votes', 'rep_votes', 'oth_votes']
    df_precinct_results_office[columns_to_convert] = df_precinct_results_office[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    
    df_precinct_results_office_grouped = df_precinct_results_office.groupby(['county', 'precinct', 'office', 'party_clean']).agg({
        'district': 'first',
        'votes': 'sum',
        'dem_votes': 'sum',
        'rep_votes': 'sum',
        'oth_votes': 'sum',
    }).reset_index()
    
    
    # Aggregate results by precinct
    df_precinct_results_office_grouped = df_precinct_results_office_grouped.groupby(['county', 'precinct', 'office'], as_index=False)[['dem_votes', 'rep_votes', 'oth_votes']].sum()
    
    df_precinct_results_office_grouped['dem_share'] = None
    df_precinct_results_office_grouped['rep_share'] = None
    df_precinct_results_office_grouped['oth_share'] = None
    
    dem_share = df_precinct_results_office_grouped['dem_votes'] / (df_precinct_results_office_grouped['dem_votes'] + df_precinct_results_office_grouped['rep_votes'] + df_precinct_results_office_grouped['oth_votes'])
    rep_share = df_precinct_results_office_grouped['rep_votes'] / (df_precinct_results_office_grouped['dem_votes'] + df_precinct_results_office_grouped['rep_votes'] + df_precinct_results_office_grouped['oth_votes'])
    oth_share = df_precinct_results_office_grouped['oth_votes'] / (df_precinct_results_office_grouped['dem_votes'] + df_precinct_results_office_grouped['rep_votes'] + df_precinct_results_office_grouped['oth_votes'])
    
    df_precinct_results_office_grouped['dem_share'] = dem_share
    df_precinct_results_office_grouped['rep_share'] = rep_share
    df_precinct_results_office_grouped['oth_share'] = oth_share
    
    
    # Make a list of precinct names from open data.
    df_precinct_ids = pd.DataFrame()
    df_precinct_ids[['county', 'precinct']] = df_precinct_results_office_grouped[['county', 'precinct']]
    

    def standardize_voting_place(name, county):
        # Remove all punctuation except hypens.
        name = re.sub(r'[^a-zA-Z0-9\- ]', '', name) # https://regex101.com/
    
        # Get rid of bad rows.
        if len(str(name)) < 5:
            return
        
        # Make lowercase
        name = name.lower()
    
        # Mt. Cityname should be Mount Cityname.
        name = name.replace('mt ', 'mount ')
        
        # Determine city or township
        is_city = not any(t in name for t in ['township', 'twp', 'twsp'])
        
        # Determine if ward exists
        has_ward = 'ward' in name
        ward_num = 0
    
        # Extract ward number if present
        if has_ward:
            ward_match = re.search(r'ward (\d+)', name)
            if ward_match:
                ward_num = int(ward_match.group(1))
            elif re.search(r'ward I', name):
                ward_num = 1
            elif re.search(r'ward II', name):
                ward_num = 2
            elif re.search(r'ward III', name):
                ward_num = 3
            elif re.search(r'ward IV', name):
                ward_num = 4
                
        # Determine if district exists, instead of a ward.
        has_district = False
        
        if 'district' in name:
            has_district = True
        elif 'dist' in name:
            has_district = True
        if has_district:
            # District may be spelled out or abbreviated.
            district_match = re.search(r'district (\d+)', name)
            dist_match = re.search(r'dist (\d+)', name)
    
            if district_match:
                ward_num = int(district_match.group(1))
            elif dist_match:
                ward_num = int(dist_match.group(1))
        
        # Remove stop words
        stopwords = {'township', 'twp', 'twsp', 'city', 'of', 'the', 'charter', 'ward', 'district', 'dist', 'precinct', 'prec', 'pct', 'pr', 'p'}
        words = [word for word in name.split() if word not in stopwords]
        
        # Extract precinct number (last number in the string)
        precinct_num_match = re.search(r'(\d+[a-zA-Z]?)$', ' '.join(words))
        precinct_num = precinct_num_match.group(1) if precinct_num_match else "0"
        
        # Determine the correct zero-padding for precinct number
        if any(c.isalpha() for c in precinct_num):
            precinct_num = precinct_num.zfill(4)  # An appended letter.
        else:
            precinct_num = precinct_num.zfill(3)  # No appended letters.
    
        # For extra fun, years 2018 and previous show wards AFTER precincts.
        # So, for wards we must swap the two numbers when "number ward number" is detected
        if YEAR <= 2018:
            swap_match = re.search(r'(\d+)\s+ward\s+(\d+)', name) # https://regex101.com/
            if swap_match:
                precinct_num = swap_match.group(1).zfill(3)  # First number → Precinct
                ward_num = int(swap_match.group(2))  # Second number → Ward
        
        # Extract locale name (everything before the first digit)
        locale_match = re.match(r'([a-z ]+)', ' '.join(words)) # https://regex101.com/
        locale_name = locale_match.group(1).strip().replace(' ', '_') if locale_match else "unknown"
        
        # Check if locale name originally ended with 'city'
        ends_with_city = bool(re.search(r'\b' + re.escape(locale_name.replace('-', '')) + r' city\b', name)) # https://regex101.com/
        
        if is_city and ends_with_city:
            locale_name += "_city"
    
        if locale_name == 'detroit_cb':
            # CB's in Detroit aren't included in precinct shapefiles.
            # There are no votes to count anyhow.
            return
        
        # Special case: Check if the name ends in an integer-hyphen-space-integer pattern
        # As seen in Owosso.
        hyphenated_match = re.search(r'(\d+)-\s*(\d+)$', name)
        
        if hyphenated_match:
            ward_num = int(hyphenated_match.group(1))
            precinct_num = hyphenated_match.group(2).zfill(3)
    
        # Some small precincts are unnumbered.
        if precinct_num == '000':
            precinct_num = '001'
    
        # Get the county
        county_name = county.replace(' ', '_')
        county_name = county_name.lower()
        
        # Construct the standardized ID
        id_type = "city-" if is_city else "township-"
        standardized_id = f"{county_name}--{id_type}{locale_name}--{int(ward_num):02}--{precinct_num}"
        
        return standardized_id
    
    
    df_precinct_ids['standardized_id'] = df_precinct_ids.apply(lambda row: standardize_voting_place(row['precinct'], row['county']), axis=1)
    df_precinct_ids = df_precinct_ids.loc[df_precinct_ids['standardized_id'].notna()]
    
    
    def makeLocaleName(id_str):
        type_match = re.search(r"--(city|township)-", id_str)
        if type_match:
            locale_type = type_match.group(1)
            
        locale_match = re.search(r"--(?:city|township)-([^--]+)", id_str)
        locale_name = locale_match.group(1).replace('_', ' ')
        output = locale_name + " " + locale_type
        return output
    
        
    def getWardNumber(id_str):
        # There are many one-off exceptions to note
        # that are located elsewhere in this notebook.
        pattern = re.compile(r"(?:city|township)-[^-]+--(\d{2})--")
        match = re.search(pattern, id_str)
        ward_num = match.group(1)
        return ward_num
    
    
    def getPrecinctNumber(id_str):
        pattern = re.compile(r"--(\d{2})--(.+)$")
        match = re.search(pattern, id_str)
        precinct_num = match.group(2)
        return precinct_num
    
    
    df_precinct_ids['county_full'] = df_precinct_ids['county'].apply(lambda x: x.replace("'s", "") + ' County')
    df_precinct_ids['county_full'] = df_precinct_ids['county'].apply(lambda x: x.replace("Gd. Traverse", "Grand Traverse") + ' County')
    df_precinct_ids['locale_full'] = df_precinct_ids['standardized_id'].apply(lambda x: makeLocaleName(x))
    df_precinct_ids['ward_num'] = df_precinct_ids['standardized_id'].apply(lambda x: getWardNumber(x))
    df_precinct_ids['precinct_num'] = df_precinct_ids['standardized_id'].apply(lambda x: getPrecinctNumber(x))
    
    
    if 'subdivision_fips' not in df_precinct_ids.columns:
        df_precinct_ids['subdivision_fips'] = None
    
    for index, row in df_precinct_ids.iterrows():
        # Get county fips.
        county_fips_series = df_census_fips[df_census_fips['COUNTYNAME'].str.lower() == row['county_full'].lower()]['COUNTYFP']
        county_fips = county_fips_series.iloc[0] if not county_fips_series.empty else None
        df_precinct_ids.at[index, 'county_fips'] = county_fips
            
        # Get city fips despite ambiguities.
        subdivision_fips = None
        subdivision_fips_series = df_census_fips[
            (df_census_fips['COUSUBNAME'].str.lower().str.replace('.', '', regex=False) == row['locale_full'].lower()) &
            (df_census_fips['COUNTYNAME'].str.lower().str.replace('.', '', regex=False) == row['county_full'].lower().replace('.', ''))
        ]['COUSUBFP']
        
        if not subdivision_fips_series.empty:
            subdivision_fips = subdivision_fips_series.iloc[0] if not subdivision_fips_series.empty else None
        else:
            # Sometimes cities do not have duplicate suffixes (no "city city").
            subdivision_fips = None
            subdivision_fips_series = df_census_fips[
                (df_census_fips['COUSUBNAME'].str.lower().str.replace('.', '', regex=False) == row['locale_full'].lower().replace('city city', 'city')) &
                (df_census_fips['COUNTYNAME'].str.lower().str.replace('.', '', regex=False) == row['county_full'].lower().replace('.', ''))
            ]['COUSUBFP']
            
            if not subdivision_fips_series.empty:
                subdivision_fips = subdivision_fips_series.iloc[0] if not subdivision_fips_series.empty else None
            else:
                # Sometimes an ambiguous place name (which we default classify as cities) is a charter township.
                subdivision_renamed = row['locale_full'].lower().replace('city', 'charter township')
                subdivision_fips_series = df_census_fips[
                    (df_census_fips['COUSUBNAME'].str.lower().str.replace('.', '', regex=False) == subdivision_renamed) &
                    (df_census_fips['COUNTYNAME'].str.lower().str.replace('.', '', regex=False) == row['county_full'].lower().replace('.', ''))
                ]['COUSUBFP']
                
                if not subdivision_fips_series.empty:
                    subdivision_fips = subdivision_fips_series.iloc[0] if not subdivision_fips_series.empty else None
                else:
                    # Or sometimes an ambiguous township place name leaves out "charter".
                    subdivision_renamed = row['locale_full'].lower().replace('city', 'township')
                    subdivision_fips_series = df_census_fips[
                        (df_census_fips['COUSUBNAME'].str.lower().str.replace('.', '', regex=False) == subdivision_renamed) & 
                        (df_census_fips['COUNTYNAME'].str.lower().str.replace('.', '', regex=False) == row['county_full'].lower().replace('.', ''))
                    ]['COUSUBFP']
                    
                    if not subdivision_fips_series.empty:
                        subdivision_fips = subdivision_fips_series.iloc[0] if not subdivision_fips_series.empty else None
                    else:
                        # And another edge case is a township in one dataset labeled as a charter township in the second dataset.
                        subdivision_renamed = row['locale_full'].lower().replace('township', 'charter township')
                        subdivision_fips_series = df_census_fips[
                            (df_census_fips['COUSUBNAME'].str.lower().str.replace('.', '', regex=False) == subdivision_renamed) & 
                            (df_census_fips['COUNTYNAME'].str.lower().str.replace('.', '', regex=False) == row['county_full'].lower().replace('.', ''))
                        ]['COUSUBFP']
                        if not subdivision_fips_series.empty:
                            subdivision_fips = subdivision_fips_series.iloc[0] if not subdivision_fips_series.empty else None
    
        # ============================== #
        # ONE-OFF ADJUSTMENTS BEFORE STANDARDIZING AND MATCHING
        # ============================== #
        
        # COLLAPSE PRECINCT FOR CERTAIN PLACES WHERE LAST CHARACTER IS A SINGLE ALPHA CHARACTER,
        # AND OTHER ONE-OFF CORRECTIONS.
        if YEAR == 2024:
            if row['county'] == 'Jackson':
                df_precinct_ids.at[index, 'precinct_num'] = re.sub(r'^(.*\d)(?:\s*[A-Za-z]+)?$', r'\1', row['precinct_num'])
                
        if YEAR == 2022:
            if (row['locale_full'] == 'allendale township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'blendon township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'chester township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'crockery township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'georgetown township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'hastings city') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'jamestown township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'merritt r township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'port sheldon township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'tallmadge township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
            if (row['locale_full'] == 'wright township') & (row['precinct_num'][-1].isalpha()):
                df_precinct_ids.at[index, 'precinct_num'] = row['precinct_num'][:-1]
        
        # ============================== #
        
        # FIND WARDS OMITTED IN PRECINCT NAMES after 2020
        # Some precincts have names where wards aren't clearly specified in name strings like the other places.
        # We will have to dig up the ward number from the geojson shape file for each MCDFIPS + PRECINCT.
    
        # Wards for Battle Creek
        if YEAR >= 2018:
            if (row['locale_full'] == 'battle creek city'):
                df_battlecreek_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '05920') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_battlecreek_precinct.empty:
                    battlecreek_ward_num = df_battlecreek_precinct.iloc[0]['WARD']
                else:
                    battlecreek_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = battlecreek_ward_num
        
        # Wards for Benton Harbor
        if YEAR >= 2018:
            if (row['locale_full'] == 'benton harbor city'):
                df_bentonharbor_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '07520') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_bentonharbor_precinct.empty:
                    bentonharbor_ward_num = df_bentonharbor_precinct.iloc[0]['WARD']
                else:
                    bentonharbor_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = bentonharbor_ward_num
    
        # Wards for Coldwater
        if YEAR > 2020:
            if (row['locale_full'] == 'coldwater city'):
                df_precinct_ids.at[index, 'ward_num'] = row['precinct_num'][-2:] # The ward number should be whatever the precinct number is from open elections, but only two characters.
                df_precinct_ids.at[index, 'precinct_num'] = '001' # The geojson precincts numbers are all "1", and it defines the wards using precinct numbers from open elections.
        
        # Wards for Detroit
        if (row['locale_full'] == 'detroit city') | (row['locale_full'] == 'detroit city city'):
            if YEAR > 2016:
                df_detroit_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '22000') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
            else:
                df_detroit_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['Jurisdicti'] == '22000') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
                
            if (not df_detroit_precinct.empty) & (YEAR > 2016):    
                detroit_ward_num = df_detroit_precinct.iloc[0]['WARD']
            elif (not df_detroit_precinct.empty) & (YEAR <= 2016):
                detroit_ward_num = df_detroit_precinct.iloc[0]['VTD'][-5:-3]
            else:
                detroit_ward_num = None
                
            df_precinct_ids.at[index, 'ward_num'] = detroit_ward_num
    
        # Wards for Dowagiac, which doesn't have them in geojson.
        if YEAR > 2020:
            if (row['locale_full'] == 'dowagiac city'):
                df_precinct_ids.at[index, 'ward_num'] = '00'
            
        # Wards for Flint
        if YEAR > 2020:
            if (row['locale_full'] == 'flint city'):
                df_flint_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '29000') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_flint_precinct.empty:
                    flint_ward_num = df_flint_precinct.iloc[0]['WARD']
                else:
                    flint_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = flint_ward_num
    
        # Wards for Lapeer
        if YEAR > 2020:
            if (row['locale_full'] == 'lapeer city'):
                df_lapeer_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '46040') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_lapeer_precinct.empty:
                    lapeer_ward_num = df_lapeer_precinct.iloc[0]['WARD']
                else:
                    lapeer_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = lapeer_ward_num
        
        # Wards for Ludington
        if YEAR > 2020:
            if (row['locale_full'] == 'ludington city'):
                df_ludington_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '49640') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_ludington_precinct.empty:
                    ludington_ward_num = df_ludington_precinct.iloc[0]['WARD']
                else:
                    ludington_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = ludington_ward_num
    
        # Wards for Midland
        if YEAR > 2020:
            if (row['locale_full'] == 'midland city'):
                df_midland_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '53780') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_midland_precinct.empty:
                    midland_ward_num = df_midland_precinct.iloc[0]['WARD']
                else:
                    midland_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = midland_ward_num
        
        # Wards for Muskegon
        if YEAR > 2020:
            if (row['locale_full'] == 'muskegon city'):
                df_muskegon_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '56320') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_muskegon_precinct.empty:
                    muskegon_ward_num = df_muskegon_precinct.iloc[0]['WARD']
                else:
                    muskegon_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = muskegon_ward_num
        
        # Wards for Pontiac
        if YEAR > 2020:
            if (row['locale_full'] == 'pontiac city'):
                df_pontiac_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '65440') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_pontiac_precinct.empty:
                    pontiac_ward_num = df_pontiac_precinct.iloc[0]['WARD']
                else:
                    pontiac_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = pontiac_ward_num
    
        # Wards for Rochester Hills
        if YEAR > 2020:
            if (row['locale_full'] == 'rochester hills city'):
                df_rochesterhills_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '69035') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_rochesterhills_precinct.empty:
                    rochesterhills_ward_num = df_rochesterhills_precinct.iloc[0]['WARD']
                else:
                    rochesterhills_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = rochesterhills_ward_num
    
        # Wards for Tecumseh
        if YEAR > 2020:
            if (row['locale_full'] == 'tecumseh city'):
                df_tecumseh_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '79120') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_tecumseh_precinct.empty:
                    tecumseh_ward_num = df_tecumseh_precinct.iloc[0]['WARD']
                else:
                    tecumseh_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = tecumseh_ward_num
        
        # Wards for Warren
        if YEAR > 2020:
            if (row['locale_full'] == 'warren city'):
                df_warren_precinct = df_precinct_bounds_named[(df_precinct_bounds_named['MCDFIPS'] == '84000') & (df_precinct_bounds_named['PRECINCT'] == str(row['precinct_num']))]
    
                if not df_warren_precinct.empty:
                    warren_ward_num = df_warren_precinct.iloc[0]['WARD']
                else:
                    warren_ward_num = None
    
                df_precinct_ids.at[index, 'ward_num'] = warren_ward_num
    
        # ============================== #
        
        # Add final subdivision code.
        df_precinct_ids.at[index, 'subdivision_fips'] = subdivision_fips
    
    df_precinct_ids['precinct_wp_id'] = df_precinct_ids.apply(lambda row: 'WP-' + str(row['county_fips']) + '-' + str(row['subdivision_fips']) + '-' + str(row['ward_num']) + str(row['precinct_num']), axis=1)
    df_precinct_ids['precinct_wp_id'] = df_precinct_ids['precinct_wp_id'].astype(str).apply(lambda x: x.upper())
    df_precinct_ids['vtd'] = df_precinct_ids.apply(lambda row: str(row['county_fips']) + str(row['subdivision_fips']) + str(row['ward_num']) + str(row['precinct_num']), axis=1)
    df_precinct_ids['vtd'] = df_precinct_ids['vtd'].str.upper()
    
    # THESE ONE-OFFS SHOULD BE STRUCTURED INTO A MUCH BETTER FUNCTION ABOVE.
    if YEAR == 2018:
        # Jonesville needs a MCDFIPS. Data from the 2010 census fips.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Jonesville City 1', 'standardized_id'] = 'jackson--township-summit---00--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Jonesville City 1', 'subdivision_fips'] = 41920
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Jonesville City 1', 'precinct_wp_id'] = 'WP-059-41920-00001'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Jonesville City 1', 'vtd'] = '0594192000001'
        
    if YEAR == 2022:
        # Merritt township has a nasty name format.
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'merritt r township', 'standardized_id'] = 'bay--township-merritt--00--001'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'merritt r township', 'subdivision_fips'] = 53220
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'merritt r township', 'precinct_wp_id'] = 'WP-017-53220-00001'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'merritt r township', 'vtd'] = '0175322000001'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'merritt r township', 'locale_full'] = 'merritt township'
    
    if YEAR >= 2020:
        # Dowagiac county geojson does not show any wards, they are 0.
        df_precinct_ids.loc[df_precinct_ids['standardized_id'] == 'cass--city-dowagiac--01--001', 'standardized_id'] = 'cass--city-dowagiac--00--001'
        df_precinct_ids.loc[df_precinct_ids['standardized_id'] == 'cass--city-dowagiac--02--002', 'standardized_id'] = 'cass--city-dowagiac--00--002'
        df_precinct_ids.loc[df_precinct_ids['standardized_id'] == 'cass--city-dowagiac--03--003', 'standardized_id'] = 'cass--city-dowagiac--00--003'
    
        # Jackson school district precincts and wards need hand-setting.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Summit Township, Precinct 3JPS', 'standardized_id'] = 'jackson--township-summit---00--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Summit Township, Precinct 3JPS', 'precinct_wp_id'] = 'WP-075-77200-00003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Summit Township, Precinct 3JPS', 'vtd'] = '0757720000003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Summit Township, Precinct 4JPS', 'standardized_id'] = 'jackson--township-summit---00--004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Summit Township, Precinct 4JPS', 'precinct_wp_id'] = 'WP-075-77200-00004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Summit Township, Precinct 4JPS', 'vtd'] = '0757720000004'
    
        # Blackman township precincts 2, 3, 4, and 6 have extra letters appended to precincts.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 2NW', 'standardized_id'] = 'jackson--township-blackman---00--002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 2NW', 'precinct_num'] = '002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 2NW', 'precinct_wp_id'] = 'WP-075-08760-00002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 2NW', 'vtd'] = '0750876000002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 2NW', 'standardized_id'] = 'jackson--township-summit---00--002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 2NW', 'precinct_wp_id'] = 'WP-075-08760-00002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 2NW', 'vtd'] = '0750876000002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 3NW', 'standardized_id'] = 'jackson--township-blackman---00--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 3NW', 'precinct_num'] = '003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 3NW', 'precinct_wp_id'] = 'WP-075-08760-00003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 3NW', 'vtd'] = '0750876000003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 3NW', 'standardized_id'] = 'jackson--township-summit---00--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 3NW', 'precinct_wp_id'] = 'WP-075-08760-00003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 3NW', 'vtd'] = '0750876000003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 4NW', 'standardized_id'] = 'jackson--township-blackman---00--004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 4NW', 'precinct_num'] = '004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 4NW', 'precinct_wp_id'] = 'WP-075-08760-00004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 4NW', 'vtd'] = '0750876000004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 4NW', 'standardized_id'] = 'jackson--township-summit---00--004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 4NW', 'precinct_wp_id'] = 'WP-075-08760-00004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 4NW', 'vtd'] = '0750876000004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 6NW', 'standardized_id'] = 'jackson--township-blackman---00--006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 6NW', 'precinct_num'] = '006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 6NW', 'precinct_wp_id'] = 'WP-075-08760-00006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 6NW', 'vtd'] = '0750876000006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 6NW', 'standardized_id'] = 'jackson--township-summit---00--006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 6NW', 'precinct_wp_id'] = 'WP-075-08760-00006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Blackman Charter Township, Precinct 6NW', 'vtd'] = '0750876000006'
    
        # Columbia township precinct 1 has extra letters appended to precincts.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Columbia Township, Precinct 3COL', 'standardized_id'] = 'jackson--township-columbia--00--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Columbia Township, Precinct 3COL', 'precinct_num'] = '003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Columbia Township, Precinct 3COL', 'precinct_wp_id'] = 'WP-075-17400-00003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Columbia Township, Precinct 3COL', 'vtd'] = '0751740000003'
    
        # Norvell township precinct 1 has extra letters appended to precincts.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Norvell Township, Precinct 2COL', 'standardized_id'] = 'jackson--township-norvell--00--002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Norvell Township, Precinct 2COL', 'precinct_num'] = '002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Norvell Township, Precinct 2COL', 'precinct_wp_id'] = 'WP-075-59180-00002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Norvell Township, Precinct 2COL', 'vtd'] = '0755918000003'
    
        # Sandstone township precinct 2 has extra letters appended to precincts.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Sandstone Charter Township, Precinct 2WEST', 'standardized_id'] = 'jackson--township-norvell--00--002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Sandstone Charter Township, Precinct 2WEST', 'precinct_num'] = '002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Sandstone Charter Township, Precinct 2WEST', 'precinct_wp_id'] = 'WP-075-71500-00002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Sandstone Charter Township, Precinct 2WEST', 'vtd'] = '0757150000002'
        
        # Napoleon townships precincts 1 and 3 have extra letters appended to precincts, but we only need to correct precinct 3 details.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Napoleon Township, Precinct 3NAP', 'standardized_id'] = 'jackson--township-napoleon---00--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Napoleon Township, Precinct 3NAP', 'precinct_num'] = '003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Napoleon Township, Precinct 3NAP', 'precinct_wp_id'] = 'WP-075-56640-00003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Napoleon Township, Precinct 3NAP', 'vtd'] = '0755664000003'
    
        # Grass lake city is oddly-named in the open_election data.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Grass Lake Charter Township, Precinct 3VIL', 'standardized_id'] = 'jackson--township-grass_lake---00--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Grass Lake Charter Township, Precinct 3VIL', 'precinct_wp_id'] = 'WP-075-34500-00003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Grass Lake Charter Township, Precinct 3VIL', 'vtd'] = '0753450000003'
    
        # Manchester city is treated as a township (precinct #1) in the geojson data.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Manchester, Precinct 1', 'standardized_id'] = 'washtenaw--township-manchester---00--001'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Manchester, Precinct 1', 'subdivision_fips'] = 50660
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Manchester, Precinct 1', 'precinct_wp_id'] = 'WP-161-50660-00001'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Manchester, Precinct 1', 'vtd'] = '1615066000001'
        
        # Coldwater city geojson and open_election wards and precincts do not match.
        # Match it to geojson, where each precinct is "1" and the ward number is the thing to look at.
        df_precinct_ids.loc[df_precinct_ids['standardized_id'] == 'branch--city-coldwater--00--001', 'standardized_id'] = 'branch--city-coldwater--01--001'
        df_precinct_ids.loc[df_precinct_ids['standardized_id'] == 'branch--city-coldwater--00--002', 'standardized_id'] = 'branch--city-coldwater--02--001'
        df_precinct_ids.loc[df_precinct_ids['standardized_id'] == 'branch--city-coldwater--00--003', 'standardized_id'] = 'branch--city-coldwater--03--001'
        df_precinct_ids.loc[df_precinct_ids['standardized_id'] == 'branch--city-coldwater--00--004', 'standardized_id'] = 'branch--city-coldwater--04--001'
        
        # Hillsdale uses roman numerals for precincts.
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'precinct'] = 'City of Hillsdale, Ward I, Precinct 1'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'standardized_id'] = 'hillsdale--city-hillsdale--01--001'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'precinct_num'] = '001'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'subdivision_fips'] = 38460
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'precinct_wp_id'] = 'WP-059-38460-01001'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'vtd'] = '0593846001001'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale i city', 'locale_full'] = 'hillsdale city'
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'City of Hillsdale, Ward I, Precinct 1', 'precinct'] = 'City of Hillsdale, Ward I, Precinct 1'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'precinct'] = 'City of Hillsdale, Ward II, Precinct 2'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'standardized_id'] = 'hillsdale--city-hillsdale--02--002'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'precinct_num'] = '002'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'subdivision_fips'] = 38460
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'precinct_wp_id'] = 'WP-059-38460-02002'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'vtd'] = '0593846002002'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale ii city', 'locale_full'] = 'hillsdale city'
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'City of Hillsdale, Ward II, Precinct 1', 'precinct'] = 'City of Hillsdale, Ward II, Precinct 2'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'precinct'] = 'City of Hillsdale, Ward III, Precinct 3'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'standardized_id'] = 'hillsdale--city-hillsdale--03--003'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'precinct_num'] = '003'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'subdivision_fips'] = 38460
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'precinct_wp_id'] = 'WP-059-38460-03003'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'vtd'] = '0593846003003'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iii city', 'locale_full'] = 'hillsdale city'
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'City of Hillsdale, Ward III, Precinct 1', 'precinct'] = 'City of Hillsdale, Ward III, Precinct 3'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'precinct'] = 'City of Hillsdale, Ward IV, Precinct 4'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'standardized_id'] = 'hillsdale--city-hillsdale--04--004'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'precinct_num'] = '004'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'subdivision_fips'] = 38460
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'precinct_wp_id'] = 'WP-059-38460-04004'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'vtd'] = '0593846004004'
        df_precinct_ids.loc[df_precinct_ids['locale_full'] == 'hillsdale iv city', 'locale_full'] = 'hillsdale city'
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'City of Hillsdale, Ward IV, Precinct 1', 'precinct'] = 'City of Hillsdale, Ward IV, Precinct 4'
    
        # Midland city open_election data split precinct 2 into two parts. Update precinct 2A in Ward 1 so it matches geojson, delete 2B in Ward 1.
        df_precinct_ids.drop(df_precinct_ids[df_precinct_ids['precinct'] == 'City of Midland, Ward 1, Precinct 2B'].index, inplace=True)
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 1, Precinct 2A', 'standardized_id'] = 'midland--city-midland--01--002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 1, Precinct 2A', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 1, Precinct 2A', 'precinct_num'] = '002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 1, Precinct 2A', 'precinct_wp_id'] = 'WP-111-53780-01002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 1, Precinct 2A', 'vtd'] = '1115378001002'
        # Some other Midland precincts need wards.
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 4, Precinct 7', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 4, Precinct 7', 'precinct_wp_id'] = 'WP-111-53780-04007'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 4, Precinct 7', 'vtd'] = '1115378004007'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 4, Precinct 8', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 4, Precinct 8', 'precinct_wp_id'] = 'WP-111-53780-04008'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 4, Precinct 8', 'vtd'] = '1115378004008'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 5, Precinct 9', 'ward_num'] = '05'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 5, Precinct 9', 'precinct_wp_id'] = 'WP-111-53780-05009'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 5, Precinct 9', 'vtd'] = '1115378005009'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 5, Precinct 10', 'ward_num'] = '05'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 5, Precinct 10', 'precinct_wp_id'] = 'WP-111-53780-05010'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'City of Midland, Ward 5, Precinct 10', 'vtd'] = '1115378005010'
    
    
    # Handle Rochester Hills separately.
    if (YEAR == 2014) | (YEAR == 2016) | (YEAR == 2018):
        # Manually assign correct wards to Rochester Hills, the geojson dataset has few columns.
        # Ignore the precincts labeled similar to this: "Rochester City 5", those are redundant.
        # The remaining 32 rows in the open_election dataset nicely correspond to the 32 rows in the geojson set.
        # Precint 1, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 1 Ward 1', 'precinct'] = 'Rochester Hills City 1 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 1 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--001'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 1 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 1 Ward 1', 'precinct_num'] = '001'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 1 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-03001'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 1 Ward 1', 'vtd'] = '1256903503001'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 1 Ward 1', 'precinct'] = 'Rochester Hills City 1 Ward 3'
        # Precinct 2, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 2 Ward 1', 'precinct'] = 'Rochester Hills City 2 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 2 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 2 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 2 Ward 1', 'precinct_num'] = '002'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 2 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 2 Ward 1', 'vtd'] = '1256903502003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 2 Ward 1', 'precinct'] = 'Rochester Hills City 2 Ward 3'
        # Precinct 3, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 3 Ward 1', 'precinct'] = 'Rochester Hills City 3 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 3 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 3 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 3 Ward 1', 'precinct_num'] = '003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 3 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-03003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 3 Ward 1', 'vtd'] = '1256903503003'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 3 Ward 1', 'precinct'] = 'Rochester Hills City 3 Ward 3'
        # Precinct 4, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 4 Ward 1', 'precinct'] = 'Rochester Hills City 4 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 4 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 4 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 4 Ward 1', 'precinct_num'] = '004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 4 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 4 Ward 1', 'vtd'] = '1256903502004'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 4 Ward 1', 'precinct'] = 'Rochester Hills City 4 Ward 2'
        # Precinct 5, Ward 4
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 5 Ward 1', 'precinct'] = 'Rochester Hills City 5 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 5 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--04--005'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 5 Ward 1', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 5 Ward 1', 'precinct_num'] = '005'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 5 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-04005'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 5 Ward 1', 'vtd'] = '1256903504005'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 5 Ward 1', 'precinct'] = 'Rochester Hills City 5 Ward 4'
        # Precinct 6, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 6 Ward 1', 'precinct'] = 'Rochester Hills City 6 Ward 1'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 6 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 6 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 6 Ward 1', 'precinct_num'] = '006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 6 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01006'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 6 Ward 1', 'vtd'] = '1256903501006'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 6 Ward 1', 'precinct'] = 'Rochester Hills City 6 Ward 1' # same name
        # Precinct 7, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 7 Ward 1', 'precinct'] = 'Rochester Hills City 7 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 7 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--007'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 7 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 7 Ward 1', 'precinct_num'] = '007'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 7 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02007'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 7 Ward 1', 'vtd'] = '1256903502007'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 7 Ward 1', 'precinct'] = 'Rochester Hills City 7 Ward 2'
        # Precinct 8, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 8 Ward 1', 'precinct'] = 'Rochester Hills City 8 Ward 1'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 8 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--008'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 8 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 8 Ward 1', 'precinct_num'] = '008'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 8 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01008'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 8 Ward 1', 'vtd'] = '1256903501008'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 8 Ward 1', 'precinct'] = 'Rochester Hills City 8 Ward 1' # same name
        # Precinct 9, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 9 Ward 1', 'precinct'] = 'Rochester Hills City 9 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 9 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--009'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 9 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 9 Ward 1', 'precinct_num'] = '009'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 9 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02009'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 9 Ward 1', 'vtd'] = '1256903502009'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 9 Ward 1', 'precinct'] = 'Rochester Hills City 9 Ward 2'
        # Precinct 10, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 10 Ward 1', 'precinct'] = 'Rochester Hills City 10 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 10 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--010'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 10 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 10 Ward 1', 'precinct_num'] = '010'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 10 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02010'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 10 Ward 1', 'vtd'] = '1256903502010'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 10 Ward 1', 'precinct'] = 'Rochester Hills City 10 Ward 2'
        # Precinct 11, Ward 4
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 11 Ward 1', 'precinct'] = 'Rochester Hills City 11 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 11 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--04--011'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 11 Ward 1', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 11 Ward 1', 'precinct_num'] = '011'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 11 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-04011'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 11 Ward 1', 'vtd'] = '1256903504011'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 11 Ward 1', 'precinct'] = 'Rochester Hills City 11 Ward 4'
        # Precinct 12, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 12 Ward 1', 'precinct'] = 'Rochester Hills City 12 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 12 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--012'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 12 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 12 Ward 1', 'precinct_num'] = '012'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 12 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-03012'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 12 Ward 1', 'vtd'] = '1256903503012'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 12 Ward 1', 'precinct'] = 'Rochester Hills City 12 Ward 3'
        # Precinct 13, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 13 Ward 1', 'precinct'] = 'Rochester Hills City 13 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 13 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--013'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 13 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 13 Ward 1', 'precinct_num'] = '013'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 13 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-03013'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 13 Ward 1', 'vtd'] = '1256903503013'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 13 Ward 1', 'precinct'] = 'Rochester Hills City 13 Ward 3'
        # Precinct 14, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 14 Ward 1', 'precinct'] = 'Rochester Hills City 14 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 14 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--014'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 14 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 14 Ward 1', 'precinct_num'] = '014'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 14 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01014'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 14 Ward 1', 'vtd'] = '1256903501014'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 14 Ward 1', 'precinct'] = 'Rochester Hills City 14 Ward 1' # same name
        # Precinct 15, Ward 4
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 15 Ward 1', 'precinct'] = 'Rochester Hills City 15 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 15 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--04--015'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 15 Ward 1', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 15 Ward 1', 'precinct_num'] = '015'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 15 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-04015'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 15 Ward 1', 'vtd'] = '1256903504015'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 15 Ward 1', 'precinct'] = 'Rochester Hills City 15 Ward 4'
        # Precinct 16, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 16 Ward 1', 'precinct'] = 'Rochester Hills City 16 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 16 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--016'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 16 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 16 Ward 1', 'precinct_num'] = '016'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 16 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-03016'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 16 Ward 1', 'vtd'] = '1256903503016'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 16 Ward 1', 'precinct'] = 'Rochester Hills City 16 Ward 3'
        # Precinct 17, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 17 Ward 1', 'precinct'] = 'Rochester Hills City 17 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 17 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--017'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 17 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 17 Ward 1', 'precinct_num'] = '017'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 17 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01017'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 17 Ward 1', 'vtd'] = '1256903501017'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 17 Ward 1', 'precinct'] = 'Rochester Hills City 17 Ward 1' # same name
        # Precinct 18, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 18 Ward 1', 'precinct'] = 'Rochester Hills City 18 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 18 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--018'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 18 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 18 Ward 1', 'precinct_num'] = '018'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 18 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02018'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 18 Ward 1', 'vtd'] = '1256903502018'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 18 Ward 1', 'precinct'] = 'Rochester Hills City 18 Ward 2'
        # Precinct 19, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 19 Ward 1', 'precinct'] = 'Rochester Hills City 19 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 19 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--019'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 19 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 19 Ward 1', 'precinct_num'] = '019'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 19 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02019'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 19 Ward 1', 'vtd'] = '1256903502019'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 19 Ward 1', 'precinct'] = 'Rochester Hills City 19 Ward 2'
        # Precinct 20, Ward 4
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 20 Ward 1', 'precinct'] = 'Rochester Hills City 20 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 20 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--04--020'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 20 Ward 1', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 20 Ward 1', 'precinct_num'] = '020'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 20 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-04020'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 20 Ward 1', 'vtd'] = '1256903504020'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 20 Ward 1', 'precinct'] = 'Rochester Hills City 20 Ward 4'
        # Precinct 21, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 21 Ward 1', 'precinct'] = 'Rochester Hills City 21 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 21 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--021'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 21 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 21 Ward 1', 'precinct_num'] = '021'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 21 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02021'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 21 Ward 1', 'vtd'] = '1256903502021'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 21 Ward 1', 'precinct'] = 'Rochester Hills City 21 Ward 2'
        # Precinct 22, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 22 Ward 1', 'precinct'] = 'Rochester Hills City 22 Ward 1'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 22 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--022'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 22 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 22 Ward 1', 'precinct_num'] = '022'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 22 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01022'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 22 Ward 1', 'vtd'] = '1256903501022'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 22 Ward 1', 'precinct'] = 'Rochester Hills City 22 Ward 1' # same name
        # Precinct 23, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 23 Ward 1', 'precinct'] = 'Rochester Hills City 23 Ward 1'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 23 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--023'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 23 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 23 Ward 1', 'precinct_num'] = '023'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 23 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01023'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 23 Ward 1', 'vtd'] = '1256903501023'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 23 Ward 1', 'precinct'] = 'Rochester Hills City 23 Ward 1' # same name
        # Precinct 24, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 24 Ward 1', 'precinct'] = 'Rochester Hills City 24 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 24 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--024'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 24 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 24 Ward 1', 'precinct_num'] = '024'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 24 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-03024'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 24 Ward 1', 'vtd'] = '1256903503024'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 24 Ward 1', 'precinct'] = 'Rochester Hills City 24 Ward 3'
        # Precinct 25, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 25 Ward 1', 'precinct'] = 'Rochester Hills City 25 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 25 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--025'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 25 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 25 Ward 1', 'precinct_num'] = '025'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 25 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02025'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 25 Ward 1', 'vtd'] = '1256903502025'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 25 Ward 1', 'precinct'] = 'Rochester Hills City 25 Ward 2'
        # Precinct 26, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 26 Ward 1', 'precinct'] = 'Rochester Hills City 26 Ward 1'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 26 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--026'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 26 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 26 Ward 1', 'precinct_num'] = '026'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 26 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01026'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 26 Ward 1', 'vtd'] = '1256903501026'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 26 Ward 1', 'precinct'] = 'Rochester Hills City 26 Ward 1' # same name
        # Precinct 27, Ward 4
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 27 Ward 1', 'precinct'] = 'Rochester Hills City 27 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 27 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--04--027'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 27 Ward 1', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 27 Ward 1', 'precinct_num'] = '027'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 27 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-04027'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 27 Ward 1', 'vtd'] = '1256903504027'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 27 Ward 1', 'precinct'] = 'Rochester Hills City 27 Ward 4'
        # Precinct 28, Ward 4
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 28 Ward 1', 'precinct'] = 'Rochester Hills City 28 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 28 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--04--028'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 28 Ward 1', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 28 Ward 1', 'precinct_num'] = '028'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 28 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-04028'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 28 Ward 1', 'vtd'] = '1256903504028'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 28 Ward 1', 'precinct'] = 'Rochester Hills City 28 Ward 4'
        # Precinct 29, Ward 1
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 29 Ward 1', 'precinct'] = 'Rochester Hills City 29 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 29 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--01--029'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 29 Ward 1', 'ward_num'] = '01'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 29 Ward 1', 'precinct_num'] = '029'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 29 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-01029'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 29 Ward 1', 'vtd'] = '1256903501029'
        # df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 29 Ward 1', 'precinct'] = 'Rochester Hills City 29 Ward 1' # same name
        # Precinct 30, Ward 4
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 30 Ward 1', 'precinct'] = 'Rochester Hills City 30 Ward 4'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 30 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--04--030'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 30 Ward 1', 'ward_num'] = '04'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 30 Ward 1', 'precinct_num'] = '030'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 30 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-04030'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 30 Ward 1', 'vtd'] = '1256903504030'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 30 Ward 1', 'precinct'] = 'Rochester Hills City 30 Ward 4'
        # Precinct 31, Ward 3
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 31 Ward 1', 'precinct'] = 'Rochester Hills City 31 Ward 3'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 31 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--03--031'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 31 Ward 1', 'ward_num'] = '03'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 31 Ward 1', 'precinct_num'] = '031'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 31 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-03031'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 31 Ward 1', 'vtd'] = '1256903503031'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 31 Ward 1', 'precinct'] = 'Rochester Hills City 31 Ward 3'
        # Precinct 32, Ward 2
        df_precinct_results_office_grouped.loc[df_precinct_results_office_grouped['precinct'] == 'Rochester Hills City 32 Ward 1', 'precinct'] = 'Rochester Hills City 32 Ward 2'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 32 Ward 1', 'standardized_id'] = 'oakland--city-rochester_hills--02--032'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 32 Ward 1', 'ward_num'] = '02'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 32 Ward 1', 'precinct_num'] = '032'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 32 Ward 1', 'precinct_wp_id'] = 'WP-125-69035-02032'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 32 Ward 1', 'vtd'] = '1256903502032'
        df_precinct_ids.loc[df_precinct_ids['precinct'] == 'Rochester Hills City 32 Ward 1', 'precinct'] = 'Rochester Hills City 32 Ward 2'
    
    
    # Compute a consistent numerical id for all cycles.
    df_precinct_ids['standardized_id_num'] = df_precinct_ids['precinct_wp_id'].apply(lambda x: x.replace('WP', '').replace('-', ''))
    
    if YEAR <= 2016:
        df_precinct_results_office_grouped = pd.merge(df_precinct_results_office_grouped, df_precinct_ids[['county', 'precinct', 'vtd']], on=['county', 'precinct'], how='left')
    
    df_precinct_results_office_grouped_precinct_matched = pd.merge(df_precinct_results_office_grouped, df_precinct_ids, left_on=['county', 'precinct'], right_on=['county', 'precinct'], how='left')
    
    # Year 2016 and earlier contain less data, so we need
    # to use the vtd column to figure out precinct numbers and ward numbers.
    if YEAR == 2016:
        vtd_column_name = 'VTD' + str(YEAR)
        df_precinct_bounds.rename(columns={vtd_column_name: 'vtd'}, inplace=True)
    
    if YEAR > 2016:
        df_precinct_bounds_with_results = pd.merge(df_precinct_bounds, df_precinct_results_office_grouped_precinct_matched, left_on='PRECINCTID', right_on='precinct_wp_id', how='left')
    elif YEAR == 2016:
        df_precinct_bounds_with_results = pd.merge(df_precinct_bounds, df_precinct_results_office_grouped_precinct_matched, left_on='vtd', right_on='vtd_x', how='left')
    elif YEAR == 2014:
        df_precinct_bounds_with_results = pd.merge(df_precinct_bounds, df_precinct_results_office_grouped_precinct_matched, left_on='VP', right_on='vtd_x', how='left')
    
    
    def cleanPrecinctResultsData(df):
        df.columns = df.columns.str.lower()
        columns = ['active_voters', 'countyfips', 'county_full', 'funcstat', 'id', 
                   'jurisdicti', 'jurisdiction_name', 'label', 'lsad', 'mcdfips', 'name',
                   'precinct', 'precinct_long_name', 'precinct_short_name', 'precinctla', 'precinctid',
                   'shape_star', 'shape_stle', 'statefp', 'fp', 'vtd', 'vtdi', 'vtdst',
                   'vtd_x', 'vtd_x', 'vtd_y', 'vtd_y', 'ward']
    
        df = df.drop(columns=[col for col in columns if col in df.columns])
        df = df[sorted(df.columns)]
        
        return df
    
    
    # Standardize columns for all cycles.
    df_precinct_bounds_with_results_clean = cleanPrecinctResultsData(df_precinct_bounds_with_results)
    df_precinct_bounds_with_results_clean = df_precinct_bounds_with_results_clean.dropna(subset=["standardized_id_num"]).loc[df_precinct_bounds_with_results_clean["standardized_id_num"] != ""]
    
    df_precinct_bounds_with_results_clean.to_file('data/generated_data/df_00_election_' + str(YEAR) + '_' + OFFICE.replace('.', '').replace(' ', '_') + '.geojson', driver='GeoJSON')
    
    if PLOT_MAP == True:
        plotPrecinctBounds(df_precinct_bounds_with_results_clean)
    
    
    def getOfficeCode(office_name):
        offices = {
            "President": 1,
            "Governor": 2,
            "Secretary of State": 3,
            "Attorney General": 4,
            "U.S. Senator": 5,
            "U.S. Senate": 5, # open elections format
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
    
    
    df_precinct_new = df_precinct_bounds_with_results_clean.copy()
    df_precinct_new = df_precinct_new.drop(columns=['objectid', 'standardized_id', 'subdivision_fips'])
    
    df_precinct_new['Office Code'] = df_precinct_new['office'].apply(getOfficeCode)
    df_precinct_new['Election Type'] = 'GEN'
    df_precinct_new['total_votes'] = df_precinct_new['dem_share'] + df_precinct_new['rep_share'] + df_precinct_new['oth_share']
    df_precinct_new['turnout_pct'] = df_precinct_new['total_votes'] / df_precinct_new['registered_voters']
    
    # Make one number to gauge partisan temperature.
    # Left-wing dems are left of 0 (neg nums). Right-wing reps, right of 0 (pos nums).
    df_precinct_new['partisan_temp'] = (-1 * df_precinct_new['dem_share']) + df_precinct_new['rep_share']
    
    df_precinct_new = df_precinct_new.rename(columns={
        'county': 'County Name', 
        'county_fips': 'Census County Code', 
        'precinct_num': 'Precinct Number', 
        'ward_num': 'Ward Number', 
        'locale_full': 'City/Township Description',
        'electionye': 'Election Year',
        'office': 'Office Description',
        'precinct_wp_id': 'standardized_id'
    })
    
    df_precinct_new['County Name'] = df_precinct_new['County Name'].str.upper()
    df_precinct_new['City/Township Description'] = df_precinct_new['City/Township Description'].str.upper()
    df_precinct_new['Office Description'] = df_precinct_new['Office Description'].str.upper()
    
    desired_order = [
        'Ward Number', 'Precinct Number', 'Office Code', 'Office Description', 
        'County Name', 'Census County Code', 'Election Year', 'Election Type', 'City/Township Description',
        'standardized_id', 'standardized_id_num', 'dem_votes', 'oth_votes', 'rep_votes', 'geometry',
        'total_votes', 'dem_share', 'rep_share', 'oth_share', 'partisan_temp', 'registered_voters', 'turnout_pct',
    ]
    
    df_precinct_new = df_precinct_new[desired_order]
    
    gdf_precinct_new = gpd.GeoDataFrame(df_precinct_new, geometry='geometry')
    gdf_precinct_new.set_crs(epsg=4326, inplace=True)
    gdf_precinct_new.to_file(f'data/generated_data/df_01_election_{YEAR}_{OFFICE.replace(' ', '_').replace('.', '')}.geojson', driver='GeoJSON')
    
print('Done.')


# In[ ]:




