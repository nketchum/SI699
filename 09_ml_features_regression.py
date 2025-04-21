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

# NUMERIC ONLY
TARGETS = ['partisan_temp']

TOP_N_FEATURES = 100
TOP_N_FEATURES_TO_DISPLAY = 15


# In[ ]:


from functools import reduce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split,cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)


# In[ ]:


# Socioeconomic data as features in addition
# to the original and the engineered features.
census_datasets = [
    'b02001_race', 'b04007_ancestry', 'b05012_nativity_us', 'b08303_travel_time_work', 'b25003_housing_rentership', 
    'dp02_selected_social_characteristics', 'dp03_selected_economic_characteristics', 'dp04_housing_characteristics', 'dp05_age_race', 
    's0101_age_sex', 's1101_households_families', 's1201_marital_status', 's1501_educational_attainment', 's1701_income_poverty', 
    's1903_median_income', 's2101_veteran_status', 's2201_food_stamps', 's2301_employment_status', 's2401_occupation_sex', 
    's2403_industry_sex', 's2501_occupancy_characteristics', 's2701_health_insurance', 's2503_financial_characteristics',
]

# These key-like columns just add noise.
drop_features_required = [
    'standardized_id', 'standardized_id_num',
    'aland_tract', 'awater_tract', 'geoid_tract', 'geoidfq_tract', 
    'geometry', 'geometry_tract', 'name_tract', 'tractce_tract',
    'nearest_bound_census_tract', 'nearest_bound_school_district', 'nearest_bound_zipcode',
]

# Optionally drop one or more of these during 
# train/test/prediction.

# If one of these are a target, they must be enabled in this
# list so that it can be dropped later.
drop_features_optional = [
    'office_code', 
    'dem_share_prev', 
    'rep_share_prev', 'oth_share_prev', 
    'dem_share_change_prev', 'rep_share_change_prev', 'oth_share_change_prev', 
    'dem_votes_change_prev', 'rep_votes_change_prev', 'oth_votes_change_prev', 
    'registered_voters_change_prev', 'turnout_pct_change_prev', 
    # 'partisan_temp_prev', 
    'partisan_temp_change_prev', 
    'partisanship_lean_prev', 
    'partisanship_lean_change_prev', 
    'partisanship_lean_change_amount_prev',
]

# Seen features that may or may not be used as
# targets as well.
drop_features_seen = [
    'dem_votes', 'oth_votes', 'rep_votes', 'total_votes', 
    'dem_share', 'rep_share', 'oth_share',  'turnout_pct',
    'dem_share_change_curr','rep_share_change_curr', 'oth_share_change_curr', 
    'dem_votes_change_curr','rep_votes_change_curr', 'oth_votes_change_curr', 
    'partisan_temp', 'partisanship_lean_curr', 'registered_voters',
    'registered_voters_change_curr','turnout_pct_change_curr',
    'partisan_temp_category', 'partisan_temp_change_curr',
    'pedersen_index_percent', 'pedersen_index',
    'partisanship_lean_change_amount_curr',
]


# In[ ]:


# # # DO NOT EDIT BELOW THIS LINE
for target in TARGETS:
    if target in drop_features_seen:
        drop_features_seen.remove(target) # Keep target in features for later extraction
    if target in drop_features_optional:
        drop_features_optional.remove(target) # Keep target in features for later extraction

drop_features = drop_features_required + drop_features_optional + drop_features_seen

# Store dropped features for each target
DROP_FEATURES_DICT = {}

for target in TARGETS:
    drop_features_copy = drop_features.copy()
    
    if target in drop_features_copy:
        drop_features_copy.remove(target)
        
    DROP_FEATURES_DICT[target] = drop_features_copy


# In[ ]:


def removeUncommonColumns(nested_dict):
    print("Removing uncommon columns...")
    
    # Flatten and find common columns
    all_dfs = [df for year in nested_dict for df in nested_dict[year].values()]
    common_cols = set(all_dfs[0].columns)
    for df in all_dfs[1:]:
        common_cols &= set(df.columns)
    
    # Safely trim all dataframes
    for year in nested_dict:
        for office in nested_dict[year]:
            df = nested_dict[year][office]
            existing_cols = [col for col in common_cols if col in df.columns]
            nested_dict[year][office] = df[existing_cols]

    print('Done.')
    
    return nested_dict


def makeDatasets(years, offices):
    print('Making datasets...')
    
    df_datasets = {}
    
    for year in years:
        print(f'Processing year {year}...')
        df_datasets[year] = {}
        
        for office in offices:
            office = office.replace(' ', '_').replace('.', '')
            print(f'Processing office {office}...')

            df = pd.read_csv('data/generated_data/07_ml_features_' + year + '_' + office + '.csv', low_memory=False)
            df_datasets[year][office] = df
    
    # df_datasets = removeUncommonColumns(df_datasets)
    print('Done.')
    
    return df_datasets


def makeFeaturesTargets(df, target):
    print(f'Making features and target...')
    
    y = df[[target]]
    
    X = df.drop(columns=['standardized_id_num', 'partisan_temp', 'partisan_temp_change_curr']) #KETCHUM aren't we already droppin gthese
    X = X.replace(['-', '(X)', 'N/A', 'null', ''], pd.NA)
    
    X, y = X.align(y.dropna(), join='inner', axis=0)
    
    print('Done.')
    return X, y


def fitModel(X, y, k=5):
    print(f'Fitting model...')

    categorical_cols = [
        'office_code',
        'partisanship_lean_curr',
        'partisanship_lean_prev',
        'partisanship_lean_change_prev',
    ]

    # Format the columns
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]

    # https://medium.com/@oluwabukunmige/implementing-linear-regression-with-sci-kit-learn-b6f87edc3150
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # https://www.kaggle.com/code/lovely94/pipelines-for-linear-regression
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)
    ])
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
    mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

    print(f'Average R² across {k} folds: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}')
    print(f'Average MSE across {k} folds: {mse_scores.mean():.4f} ± {mse_scores.std():.4f}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    print('Final model fitted on training split.')
    return model, X_train, X_test, y_train, y_test, numeric_cols


def plotAccuracy(y_test, y_pred):
    print(f'Plotting accuracy...')
    plt.figure(figsize=(12, 9))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction Accuracy")
    plt.grid(True)
    print('Done.')
    return plt


def featureCoeff(model):
    print(f'Computing feature coefficients from pipeline...')

    # https://medium.com/@dvdhartsman/extracting-coefficients-and-feature-names-from-scikit-learn-pipelines-331d578b8450
    regressor = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    coef = regressor.coef_.flatten()

    # Inline feature name extraction
    output_features = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == 'drop' or transformer is None:
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                names = transformer.get_feature_names_out(columns)
            except:
                names = columns
        else:
            names = columns
        output_features.extend(names)

    feature_names = output_features

    if len(coef) != len(feature_names):
        raise ValueError(f"Mismatch: {len(coef)} coefficients vs {len(feature_names)} feature names")

    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef)
    })

    top_features = df.sort_values(by='abs_coefficient', ascending=False).head(TOP_N_FEATURES_TO_DISPLAY)
    print('Done.')
    return top_features


def plotFeatureCoeff(features):
    print(f'Plotting feature coefficients...')
    plt.figure(figsize=(12, 18))
    bars = plt.barh(features['feature'], features['coefficient'])
    plt.xlabel('Coefficient Value')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(f'Most Influential Features (Linear Regression)')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid(True, axis='x', linestyle=':', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    print('Done.')
    return plt


def mergeTopFeatures(top_features_lists):
    print(f'Creating common top features using clean names...')
    from itertools import chain

    # Make clean names
    normalized_lists = []
    for item in top_features_lists:
        if isinstance(item, list):
            normalized_lists.append(item)
        elif hasattr(item, 'columns') and 'feature' in item.columns:
            normalized_lists.append(item['feature'].tolist())
        else:
            raise ValueError("Each item must be a list or a DataFrame with a 'feature' column")

    # Find intersection
    common_features = set(normalized_lists[0])
    for feature_list in normalized_lists[1:]:
        common_features.intersection_update(feature_list)

    # Preserve order
    seen = set()
    merged_common_ordered = []

    for item in chain.from_iterable(normalized_lists):
        if item in common_features and item not in seen:
            seen.add(item)
            merged_common_ordered.append(item)

    print('Done.')
    return merged_common_ordered


# In[ ]:


def getRankedFeatureList(target):
    df_ranked_features = pd.read_csv(f'data/generated_data/df_importances_{target}.csv')
    
    df_ranked_features = df_ranked_features.sort_values(by='Average', ascending=False)
    drop_features = DROP_FEATURES_DICT[target]

    df_ranked_features = df_ranked_features[~df_ranked_features['Feature name'].isin(drop_features)]
    df_ranked_features_top = df_ranked_features.head(TOP_N_FEATURES)

    ranked_features_top_list = df_ranked_features_top['Feature name'].tolist()
    
    return ranked_features_top_list


# In[ ]:


def predDatasetsIndiv(df_datasets, years, offices):
    for year in years:
        print(f'Processing year {year}...')
        
        for office in offices:
            office = office.replace(' ', '_').replace('.', '')
            print(f'Processing office {office}...')
    
            df = df_datasets[year][office].copy()

            for target in TARGETS:
                print(f'Processing target {target}...')

                X, y = makeFeaturesTargets(df, target)
                ranked_features_top_list = getRankedFeatureList(target).copy()

                if target in ranked_features_top_list:
                    ranked_features_top_list.remove(target)
                
                features_to_exclude = set(TARGETS)
                ranked_features_top_list = [f for f in ranked_features_top_list if f not in features_to_exclude]
                
                X = X[ranked_features_top_list]
        
                print(f"Training over {len(X.columns)} features...")
                
                model, X_train, X_test, y_train, y_test, numeric_cols = fitModel(X, y)

                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)
        
                r2 = model.score(X_test, y_test)
                print("R2 Score:", r2)
        
                plt = plotAccuracy(y_test, y_pred)
                plt.savefig(f'output/figures/regression_accuracy_{year}_{office}_{target}_individual.png')
                # plt.show()
                plt.close()
                
                top_features = featureCoeff(model)
                plt = plotFeatureCoeff(top_features)
                plt.savefig(f'output/figures/regression_top_features_{year}_{office}_{target}_individual.png')
                # plt.show()
                plt.close()


# #### Predict Individual Datasets

# In[ ]:


print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    print(f'Num. of years to process: {len(value)}')
    
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    df_datasets = makeDatasets(YEARS, OFFICES)
    predDatasetsIndiv(df_datasets, YEARS, OFFICES)


# #### Fit/Train Final Model
# Do we have data leakage here? Maybe not, if we train on historical data and
# <br>run a separate test on newer data in a following step. Say, here we
# train 2018-2022, <br>and then in another cell test 2024 on the same model

# In[ ]:


HOLDOUT_YEAR = '2024'


# #### Fit aggregated data
# This functionality produces the <code>model</code> object to be used later. Make sure no datasets from this
# <br>cell is NOT included in holdout testing.

# In[ ]:


def aggDatasets(df_datasets, years, offices):
    dfs = []
    
    for year in years:
        print(f'Processing year {year}...')
        for office in offices:
            office = office.replace(' ', '_').replace('.', '')
            
            print(f'Processing office {office}...')
            dfs.append(df_datasets[year][office].copy())
    if len(dfs) > 1:
        df = pd.concat(dfs, axis=0, ignore_index=True)
    elif len(dfs) == 0:
        raise ValueError("No dataframes found to aggregate.")
    elif len(dfs) == 1:
        return dfs[0]
        
    return df


# In[ ]:


def getAggDataset(holdout_year):
    df_agg_dataset_list = []
    
    for office, years in ELECTIONS.items():
        for year in years:
            if year == holdout_year:
                continue
            office = office.replace(' ', '_').replace('.', '')
            df = pd.read_csv('data/generated_data/07_ml_features_' + year + '_' + office + '.csv', low_memory=False)
            df_agg_dataset_list.append(df)
    
    df_agg = pd.concat(df_agg_dataset_list, axis=0, ignore_index=True)

    return df_agg


# In[ ]:


df_agg = getAggDataset(HOLDOUT_YEAR)

model_dict = {}

for target in TARGETS:
    
    X, y = makeFeaturesTargets(df_agg, target)

    ranked_features_top_list = getRankedFeatureList(target)
    
    if target in ranked_features_top_list:
        ranked_features_top_list.remove(target)
    
    features_to_exclude = set(TARGETS)
    ranked_features_top_list = [f for f in ranked_features_top_list if f not in features_to_exclude]
    
    X = X[ranked_features_top_list]
    X = X.replace({pd.NA: np.nan})
    
    model, X_train, X_test, y_train, y_test, numeric_cols = fitModel(X, y)

    model_dict[target] = model
    
    y_pred = model.predict(X_test)
    
    expected_columns = X.columns.tolist()
    
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    
    r2 = model.score(X_test, y_test)
    print("R2 Score:", r2)
    
    plt = plotAccuracy(y_test, y_pred)
    plt.savefig(f'output/figures/regression_accuracy_{target}_aggregate.png')
    plt.close()
    # plt.show()
    
    top_features = featureCoeff(model)
    plt = plotFeatureCoeff(top_features)
    plt.savefig(f'output/figures/regression_top_features_{target}_aggregate.png')
    plt.close()
    # plt.show()


# #### Holdout Prediction
# This functionality requires the <code>model_dict</code> object from a previous cell. Make sure this holdout
# <br>dataset was not included in the model's training.

# In[ ]:


def getHoldoutDataset(holdout_year=HOLDOUT_YEAR, holdout_office=None):
    df_holdout_dataset_list = []

    if holdout_office == None:
        for office, years in ELECTIONS.items():
            for year in years:
                if year == holdout_year:
                    office = office.replace(' ', '_').replace('.', '')
                    df = pd.read_csv('data/generated_data/07_ml_features_' + year + '_' + office + '.csv', low_memory=False)
                    df_holdout_dataset_list.append(df)
                else:
                    continue
    else:
        office = holdout_office
        office = office.replace(' ', '_').replace('.', '')
        df = pd.read_csv('data/generated_data/07_ml_features_' + HOLDOUT_YEAR + '_' + office + '.csv', low_memory=False)
        df_holdout_dataset_list.append(df)
    
    df_holdout = pd.concat(df_holdout_dataset_list, axis=0, ignore_index=True)

    return df_holdout


# In[ ]:


HOLDOUT_OFFICE = 'U.S. House'


# In[ ]:


df_holdout = getHoldoutDataset(HOLDOUT_YEAR, HOLDOUT_OFFICE)

for target in TARGETS:
    
    df_orig = df_holdout.copy()
    
    X, y = makeFeaturesTargets(df_holdout, target)
    
    ranked_features_top_list = getRankedFeatureList(target)
    
    if target in ranked_features_top_list:
        ranked_features_top_list.remove(target)
    
    features_to_exclude = set(TARGETS)
    ranked_features_top_list = [f for f in ranked_features_top_list if f not in features_to_exclude]
    
    X = X[ranked_features_top_list]
    
    X = X.replace({pd.NA: np.nan})
        
    model = model_dict[target]
    y_pred = model.predict(X)
    
    expected_columns = X.columns.tolist()
    
    mse = mean_squared_error(y, y_pred)
    print("Mean Squared Error:", mse)
    
    r2 = model.score(X, y)
    print("R2 Score:", r2)
    
    print(f"Holdout MSE ({HOLDOUT_YEAR}):", mse)
    print(f"Holdout R² ({HOLDOUT_YEAR}):", r2)
    
    with open(f'output/reports/prediction_eval_{target}_holdout_regression.txt', 'w') as f:
        output = f"R2 Score: {r2}"
        output += f"\nMSE: {mse}"
        f.write(output)
    
    plt = plotAccuracy(y, y_pred)
    plt.savefig(f'output/figures/regression_accuracy_{target}_holdout.png')
    plt.close()
    # plt.show()
    
    top_features = featureCoeff(model)
    plt = plotFeatureCoeff(top_features)
    plt.savefig(f'output/figures/regressio_top_features_{target}_holdout.png')
    plt.close()
    # plt.show()
    
    # Save holdout predictions
    df_orig['predicted_label'] = pd.Series(y_pred.ravel(), index=X.index)
    df_export = df_orig[["standardized_id_num", target, "predicted_label"]].copy()
    df_export["standardized_id_num"] = df_export["standardized_id_num"].astype(str).str.replace('.0', '').str.zfill(13)
    df_export.rename(columns={target: 'true_label'}, inplace=True)
    
    filename = f"data/generated_data/predicted_{target}_holdout.csv"
    df_export.to_csv(filename, index=False)
    print(f"Holdout predictions saved to {filename}")


# ### Grid search

# In[ ]:


def gridSearch(df, target, ranked_features_top_list):
    df = df.copy()
    
    X, y = makeFeaturesTargets(df, target)

    if target in ranked_features_top_list:
        ranked_features_top_list.remove(target)
    
    features_to_exclude = set(TARGETS)
    ranked_features_top_list = [f for f in ranked_features_top_list if f not in features_to_exclude]
    
    X = X[ranked_features_top_list]
    X = X.replace({pd.NA: np.nan})
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # https://www.geeksforgeeks.org/grid-searching-from-scratch-using-python/
    # https://towardsdatascience.com/grid-search-in-python-from-scratch-hyperparameter-tuning-3cca8443727b/
    # https://www.w3schools.com/python/python_ml_grid_search.asp
    # https://www.kdnuggets.com/2022/10/hyperparameter-tuning-grid-search-random-search-python.html
    # etc.
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    models = {
        'LinearRegression': (
            LinearRegression(), 
            {}
        ),
        'Ridge': (
            Ridge(),
            {'regressor__alpha': [0.01, 0.1, 1.0, 10]}
        ),
        'Lasso': (
            Lasso(max_iter=10000),
            {'regressor__alpha': [0.01, 0.1, 1.0, 10]}
        ),
        'ElasticNet': (
            ElasticNet(max_iter=10000), 
            {
            'regressor__alpha': [0.01, 0.1, 1.0],
            'regressor__l1_ratio': [0.2, 0.5, 0.8]
            }
        ),
        'DecisionTree': (
            DecisionTreeRegressor(), 
            {
            'regressor__max_depth': [5, 10, None],
            'regressor__min_samples_split': [2, 10]
            }
        ),
        'RandomForest': (
            RandomForestRegressor(n_jobs=-1), 
            {
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [5, 10, None]
            }
        ),
        'GradientBoosting': (
            GradientBoostingRegressor(), 
            {
            'regressor__n_estimators': [50, 100],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__max_depth': [3, 5]
            }
        ),
    }
    
    y = y.values.ravel() 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_models = {}
    results = []
    
    for name, (model, params) in models.items():
        print(f"Tuning: {name}")
        
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        grid = GridSearchCV(pipe, param_grid=params, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
    
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
    
        mse = mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        results.append({
            'Model': name,
            'Best Params': grid.best_params_,
            'MSE': mse,
            'R2': r2
        })
    
        best_models[name] = grid 
    
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)

    print('Done.')

    return best_models, results_df


# In[ ]:


df_agg = getAggDataset(HOLDOUT_YEAR)

best_models_dict = {}
results_df_dict = {}

for target in TARGETS:    
    ranked_features_top_list = getRankedFeatureList(target)
    best_models, results_df = gridSearch(df_agg, target, ranked_features_top_list)

    best_models_dict[target] = best_models
    results_df_dict[target] = results_df


# In[ ]:


def fitBestModel(df, target, params, ranked_features_top_list):
    df = df.copy()
    
    X, y = makeFeaturesTargets(df, target)

    if target in ranked_features_top_list:
        ranked_features_top_list.remove(target)
    
    features_to_exclude = set(TARGETS)
    ranked_features_top_list = [f for f in ranked_features_top_list if f not in features_to_exclude]
    
    X = X[ranked_features_top_list]
    X = X.replace({pd.NA: np.nan})
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # See citations from above.
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    gbr_final = GradientBoostingRegressor(
        random_state=42,
        **params
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', gbr_final)
    ])
    
    model.fit(X, y.values.ravel())

    print('Done.')

    return model


# In[ ]:


def cleanParams(best_models, model_name):
    params = best_models[model_name].best_params_

    # https://stackoverflow.com/questions/71290888/for-k-v-in-data-dict-items-attributeerror-list-object-has-no-attribute-i
    clean_params = {k.replace('regressor__', ''): v for k, v in params.items()}
    
    print(f'Cleaned params for {model_name}: {clean_params}')
    
    return clean_params


# In[ ]:


df_agg = getAggDataset(HOLDOUT_YEAR)

fitted_models_dict = {}

for target in TARGETS:
    clean_params = cleanParams(best_models, 'GradientBoosting')
    ranked_features_top_list = getRankedFeatureList(target)
    model = fitBestModel(df_agg, target, clean_params, ranked_features_top_list)
    fitted_models_dict[target] = model


# In[ ]:


def makePredsBestModel(df, target, model, ranked_features_top_list):
    df = df.copy()
    
    X, y = makeFeaturesTargets(df, target)

    ranked_features_top_list = [f.strip() for f in ranked_features_top_list if f.strip() not in TARGETS]
    ranked_features_top_list = [f for f in ranked_features_top_list if f in X.columns]
    if not ranked_features_top_list:
        raise ValueError("No valid features left for prediction — check feature list or holdout data.")

    X = X[ranked_features_top_list]
    X = X.replace({pd.NA: np.nan})
    X = X.dropna()
    
    # Align datagrame to cleaned X
    df = df.loc[X.index]
    
    y_pred = model.predict(X)
    
    y_true = df[target]
    
    df[f"predicted_{target}"] = y_pred
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return df, y_pred, y_true


# In[ ]:


df_holdout = getHoldoutDataset(HOLDOUT_YEAR)

for target in TARGETS:
    print(f'Processing target {target}...')

    ranked_features_top_list = getRankedFeatureList(target)

    model = fitted_models_dict[target]

    df_pred_best, y_pred_best, y_true_best = makePredsBestModel(df_holdout, target, model, ranked_features_top_list)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_best, y_pred_best, alpha=0.4)
    plt.plot([-1, 1], [-1, 1], color='red', linestyle='--')  # Diag. line
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted Temperature vs. Actual Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output/figures/regression_best_model_{target}_{HOLDOUT_YEAR}_holdout_preds.png')
    plt.close()
    # plt.show()


# #### Benchmark 1 Uniform (U.S. House 2024)

# In[ ]:


years = [HOLDOUT_YEAR]
offices = ['U.S. House']
dfs = makeDatasets(years, offices)

df_benchmark = dfs[HOLDOUT_YEAR][offices[0].replace(' ', '_').replace('.', '')]


# In[ ]:


BENCHMARK_TYPE = 'uniform'

df_benchmark1 = df_benchmark.copy()

for target in TARGETS:
    
    X, y = makeFeaturesTargets(df_benchmark1, target)

    if target in ranked_features_top_list:
        ranked_features_top_list.remove(target)
    
    features_to_exclude = set(TARGETS)
    ranked_features_top_list = [f for f in ranked_features_top_list if f not in features_to_exclude]
    
    X = X[ranked_features_top_list]
    X = X.replace({pd.NA: np.nan})
    X = X.dropna()
    
    df_benchmark1 = df_benchmark1.loc[X.index]
    y_true = df_benchmark1[target]
    
    if BENCHMARK_TYPE == 'uniform':
        low, high = y_true.min(), y_true.max()
        y_dummy = np.random.uniform(low, high, size=len(y_true))
    elif BENCHMARK_TYPE == 'permutation':
        y_dummy = np.random.permutation(y_true) # https://numpy.org/doc/2.1/reference/random/generated/numpy.random.permutation.html
    elif BENCHMARK_TYPE == 'median':
        y_dummy = np.full(len(y_true), np.median(y_true))
    else:
        mean_value = y_true.mean()
        y_dummy = [mean_value] * len(y_true)  # https://medium.com/@eskandar.sahel/a-dummy-classifier-a-baseline-classifier-or-a-null-model-71df50fd8947
    
    mse = mean_squared_error(y_true, y_dummy)
    r2 = r2_score(y_true, y_dummy)
    
    print(f"Dummy {BENCHMARK_TYPE.capitalize()} Squared Error: {mse:.4f}")
    print(f"Dummy {BENCHMARK_TYPE.capitalize()} R² Score: {r2:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_dummy, alpha=0.4)
    plt.plot([-1, 1], [-1, 1], color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Benchmark")
    plt.title("Benchmark vs. Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output/figures/regression_benchmark_{BENCHMARK_TYPE}_{target}_1.png')
    plt.close()
    # plt.show()


# #### Benchmark 2 Rule-of-Thumb (U.S. House 2024)

# In[ ]:


RULE_OF_THUMB_FEATURES = [
    'partisan_temp', 
    'partisan_temp_prev'
]

COMBINE_RULE_OF_THUMB_FEATURES = False

df_benchmark2 = df_benchmark.copy()

if COMBINE_RULE_OF_THUMB_FEATURES == False:
    rule_features_list = RULE_OF_THUMB_FEATURES
else:
    rule_features_list = [RULE_OF_THUMB_FEATURES]

for rule_features in rule_features_list:
    if COMBINE_RULE_OF_THUMB_FEATURES == True:
        X_rule = df_benchmark2[rule_features].copy()
    else:
        X_rule = df_benchmark2[[rule_features]].copy()
    
    y_true = df_benchmark2[target].copy()
    
    mask = X_rule.notna().all(axis=1) & y_true.notna()
    X_rule = X_rule.loc[mask]
    y_true = y_true.loc[mask]
    
    rule_model = LinearRegression()
    rule_model.fit(X_rule, y_true)
    y_pred = rule_model.predict(X_rule)
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("Linear Rule-of-Thumb Benchmark")
    print(f"Feature(s) used: {rule_features}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([-1, 1], [-1, 1], color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted (Rule-of-Thumb)")
    plt.title("Rule-of-Thumb Prediction vs. Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output/figures/regression_benchmark_linear_{target}_2.png')
    plt.close()
    # plt.show()


# In[ ]:




