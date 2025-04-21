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

TARGETS = [
    'dem_share',
    'rep_share',
    'oth_share',
    'dem_share_change_curr',
    'rep_share_change_curr',
    'oth_share_change_curr',
    'partisan_temp',
    'partisanship_lean_curr',
    'partisanship_lean_change_amount_curr',
    'partisan_temp_change_curr',
]


# In[ ]:


drop_features = [
    'standardized_id', 'standardized_id_num',
    'aland_tract', 'awater_tract', 'geoid_tract', 'geoidfq_tract', 
    'geometry', 'geometry_tract', 'name_tract', 'tractce_tract',
    'nearest_bound_census_tract', 'nearest_bound_school_district', 'nearest_bound_zipcode',
]


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import shap


# In[ ]:


# pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)


# In[ ]:


## Ranks features over all historic data for each target. Each target's
# most influencial features are saved independently, but without years
# or offices in the filename. These are aggregate rankings per target.

for target in TARGETS:
    print(f'Processing target {target}')
    print(f'Num. of offices to process: {len(ELECTIONS)}')

    # Best features per target computed by all offices across all years
    top_features_list = []
    
    for key, value in ELECTIONS.items():
        print(f'Num. of years to process: {len(value)}')
        
        OFFICES = [key]
        YEARS = value
    
        print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')
        
        # # Rank all features for the target defined above
        # # using several different metrics as well
        # # as an average score across metrics to help
        # # test many different combinations of features
        # # and targets.
        for year in YEARS:
            print(f'Processing year {year}...')
            
            for office in OFFICES:
                print(f'Processing year {office}...')
    
                office = office.replace(' ', '_').replace('.', '')
                
                df = pd.read_csv(f'data/generated_data/07_ml_features_{year}_{office}_with_geometry.csv', low_memory=False)
                # df = df.drop(columns=drop_features)
                
                # Target and features
                y = df[target]
        
                # Categorical targets need to be encoded
                if y.dtype == 'object' or y.dtype.name == 'category':
                    label_encoder = LabelEncoder()
                    y = pd.Series(label_encoder.fit_transform(y), name=target)
                
                X = df.drop(columns=[target])
        
                # Combine X and y, drop rows where y is NaN
                df_model = pd.concat([X, y], axis=1)
                df_model = df_model.dropna(subset=[target])
                
                # Separate again
                y = df_model[target]
                X = df_model.drop(columns=[target])
                
                # Keep only numeric features
                X_numeric = X.select_dtypes(include=[np.number]).copy()
                
                # Drop any columns with all NaNs or constant values
                X_numeric = X_numeric.dropna(axis=1, how='all')
                X_numeric = X_numeric.loc[:, X_numeric.nunique() > 1]
                
                # Fill remaining NaNs with mean
                X_numeric = X_numeric.fillna(X_numeric.mean(numeric_only=True))
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
        
                # Begin running models to compute corresonding accuracies.
                
                print('Running Correlation')
                correlations =  X_numeric.corrwith(y).abs().sort_values(ascending=False)

                # https://medium.com/@prasannarghattikar/using-random-forest-for-feature-importance-118462c40189
                print('Running Random Forest')
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, max_features='sqrt')
                rf.fit(X_train, y_train)
                rf_importances = pd.Series(rf.feature_importances_, index=X_numeric.columns)

                # https://medium.com/@agrawalsam1997/feature-selection-using-lasso-regression-10f49c973f08
                print('Running LassoCV')
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_numeric)
                lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
                lasso.fit(X_scaled, y)
                lasso_importances = pd.Series(np.abs(lasso.coef_), index=X_numeric.columns)

                # https://guhanesvar.medium.com/feature-selection-based-on-mutual-information-gain-for-classification-and-regression-d0f86ea5262a
                print('Running Mutual Information')
                mi = mutual_info_regression(X_numeric, y, random_state=42)
                mi_importances = pd.Series(mi, index=X_numeric.columns)

                # https://medium.com/@msvs.akhilsharma/unlocking-the-power-of-shap-analysis-a-comprehensive-guide-to-feature-selection-f05d33698f77
                print('Running SHAP')
                explainer = shap.Explainer(rf, X_train)
                shap_values = explainer(X_test, check_additivity=False)
                shap_importances = pd.Series(np.abs(shap_values.values).mean(0), index=X_numeric.columns)
                
                df_importances = pd.DataFrame({
                    'Correlation': correlations,
                    'RandomForest': rf_importances,
                    'LassoCV': lasso_importances,
                    'MutualInfo': mi_importances,
                    'SHAP': shap_importances
                })
                
                # df_importances['Average'] = df_importances.mean(axis=1)  # Commented out b/c skip average for each office.
                
                df_importances = df_importances.reset_index()
                df_importances.rename(columns={'index': 'Feature name'}, inplace=True)
        
                top_features_list.append(df_importances)
        
    # Combine the feature columns
    df_combined = pd.concat(top_features_list, axis=0)
    
    # Aggregate features to compute averages.
    df_aggregated = df_combined.groupby('Feature name').mean(numeric_only=True).reset_index()
    df_aggregated['Average'] = df_aggregated.select_dtypes(include=[np.number]).mean(axis=1) # Average across all offices.
    df_aggregated = df_aggregated.sort_values(by='Average', ascending=False)
    
    df_aggregated.to_csv(f'data/generated_data/df_importances_{target}.csv', index=False)

    # Plot features
    import matplotlib.pyplot as plt
    top_n = 20
    df_plot = df_aggregated.head(top_n).set_index('Feature name')
    
    # Drop 'Average' to plot metrics separately
    metrics = df_plot.drop(columns='Average')
    
    # Plot
    ax = metrics.plot(kind='barh', figsize=(12, 10), width=0.85)
    plt.gca().invert_yaxis()  # highest at top
    plt.title(f'Top {top_n} Feature Importances by Metric')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(f'output/figures/features_ranking_{target}.png')
    plt.close()
    # plt.show()


# In[ ]:




