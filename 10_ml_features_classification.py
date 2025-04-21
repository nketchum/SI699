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

TARGETS = ['partisanship_lean_curr']

TOP_N_FEATURES = 20
FEATURES_ALREADY_RANKED = True


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from tqdm import tqdm
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

# If one of these are a target, they must be enabled in this
# list so that it can be dropped later.
drop_features_optional = [
    'office_code', 
    'dem_share_prev', 
    'rep_share_prev', 'oth_share_prev', 
    'dem_share_change_prev', 'rep_share_change_prev', 'oth_share_change_prev', 
    'dem_votes_change_prev', 'rep_votes_change_prev', 'oth_votes_change_prev', 
    'registered_voters_change_prev', 'turnout_pct_change_prev', 
    'partisan_temp_prev', 
    'partisan_temp_change_prev', 
    'partisanship_lean_prev', 'partisanship_lean_change_prev', 'partisanship_lean_change_amount_prev',
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


# DO NOT EDIT BELOW THIS LINE
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


# ### XG Boost Classifier

# In[ ]:


''' Pull the engineered feature data along with its
    target for each year and office.'''
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
    
    df_datasets = removeUncommonColumns(df_datasets)
    print('Done.')
    
    return df_datasets


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



''' Remove top features not shared between 
    different datasets to prevent errors.'''
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


# In[ ]:


# INDIVIDUAL ELECTIONS PER OFFICE/YEAR
print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    print(f'Num. of years to process: {len(value)}')
    
    OFFICES = [key]
    YEARS = value

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')
    
    for year in YEARS:
        print(f'Processing year {year}...')
    
        for office in OFFICES:
            office = office.replace(' ', '_').replace('.', '')
            print(f'Processing office {office}...')
            
            df = pd.read_csv(f'data/generated_data/07_ml_features_{year}_{office}.csv')

            for target in TARGETS:
                print(f'Processing target {target}')
            
                if FEATURES_ALREADY_RANKED:
                    print(f'Features already ranked, select top...')
                    feature_importance_file = f'data/generated_data/df_importances_{target}.csv'
                    top_feature_columns = pd.read_csv(feature_importance_file)['Feature name'].head(TOP_N_FEATURES).tolist()
                    required_columns = ['standardized_id_num', target]
                    selected_columns = [col for col in top_feature_columns + required_columns if col in df.columns]
                    df = df[selected_columns].dropna(subset=[target])
                
                # Clean ID
                df['standardized_id_num'] = df['standardized_id_num'].astype(str).str.zfill(13)
                
                # Drop rows w/o targets
                df = df.dropna(subset=[target])
                
                # Save standardized IDs
                X_stids = df['standardized_id_num']
                
                # Define y and encode
                y = df[target]
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                
                # Drop unneeded columns after saving id
                cols_to_drop = [col for col in drop_features if col in df.columns]
                df = df.drop(columns=cols_to_drop)
                X = df.drop(columns=[target])
                
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
                
                for col in categorical_cols:
                    X[col] = X[col].astype(str)
                
                categorical_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                numeric_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])
                
                preprocessor = ColumnTransformer([
                    ('cat', categorical_transformer, categorical_cols),
                    ('num', numeric_transformer, numeric_cols)
                ])
                
                model = XGBClassifier(
                    objective="multi:softmax",  # multi:softmax or multi:softprob
                    num_class=len(y.unique()),
                    use_label_encoder=False,
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="mlogloss"
                )
                
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

                class_counts = pd.Series(y).value_counts()

                valid_classes = class_counts[class_counts > 1].index

                mask = y.isin(valid_classes)
                X = X[mask]
                y = y[mask]
                y_encoded = label_encoder.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
        
                decoded_y_test = label_encoder.inverse_transform(y_test)
                decoded_y_pred = label_encoder.inverse_transform(y_pred)
                
                print(classification_report(decoded_y_test, decoded_y_pred))

                cm = confusion_matrix(decoded_y_test, decoded_y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=label_encoder.classes_, 
                            yticklabels=label_encoder.classes_)
                
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                plt.savefig(f'output/figures/confusion_matrix_{year}_{office}.png')
                plt.close()
                # plt.show()
                
                # Save predictions
                results_df = pd.DataFrame({
                    'standardized_id_num': X_stids.loc[X_test.index],
                    'true_label': decoded_y_test,
                    'predicted_label': decoded_y_pred
                })
                results_df['standardized_id_num'] = results_df['standardized_id_num'].astype(str).str.zfill(13)
                
                filename = f'data/generated_data/prediction_results_{year}_{office}_classification.csv'
                results_df.to_csv(filename, index=False)
        
                #############################
                # FEATURE PERFORMANCE
                #############################
                
                feature_performance = []
                is_continuous = y.dtype.kind in 'fc' # float or continuous
                
                for feature in tqdm(X.columns):
                    X_feature = X[[feature]].copy()
                    
                    # Handle missing values
                    if X_feature[feature].dtype == 'object':
                        X_feature = X_feature.fillna(X_feature.mode().iloc[0])
                        X_feature = pd.get_dummies(X_feature, drop_first=True)
                    
                        # Need at least 1 column after one-hot encoding
                        if X_feature.shape[1] == 0:
                            continue  # Leave as is
                    else:
                        X_feature = X_feature.fillna(X_feature.mean(numeric_only=True))
                    
                    # Force one column, even if empty
                    if X_feature.shape[1] == 0:
                        continue
                    
                    X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(X_feature, y, test_size=0.2, random_state=42)
                    
                    # if is_continuous:  # Regression
                    #     model = LinearRegression()
                    #     model.fit(X_train_feat, y_train_feat)
                    #     y_pred = model.predict(X_test_feat)
                    #     score = r2_score(y_test_feat, y_pred)  # R² for regression
                    #     metric = "Score"
                        
                    # else:  # Classification
                    model = LogisticRegression(max_iter=200)  # or DecisionTreeClassifier()
                    model.fit(X_train_feat, y_train_feat)
                    y_pred = model.predict(X_test_feat)
                    score = accuracy_score(y_test_feat, y_pred)
                    metric = "Score"
                        
                    feature_performance.append({"Feature name": feature, metric: score})
                    
                feature_performance_df = pd.DataFrame(feature_performance).sort_values(by=metric, ascending=False)
                filename = f'data/generated_data/feature_rankings_{target}_{year}_{office}.csv'
                feature_performance_df.to_csv(filename, index=None)


# In[ ]:


HOLDOUT_YEAR = '2024'


# In[ ]:


# AGGREGATE TRAIN/TEST
# Take several years and aggregate into a larger single dataset
# for maximum training data, perhaps leaving out one holdout set.

from xgboost import XGBClassifier

print(f'Num. of offices to process: {len(ELECTIONS)}')

for key, value in ELECTIONS.items():
    print(f'Num. of years to process: {len(value)}')

    OFFICES = [key]
    YEARS = value.copy()

    if HOLDOUT_YEAR in YEARS:
        YEARS.remove(HOLDOUT_YEAR)

    if not YEARS:
        print(f"Skipping {key} — no years left after removing holdout.")
        continue

    print(f'Process office(s): {key} for year(s): {', '.join(YEARS)}')

    df_datasets = makeDatasets(YEARS, OFFICES)
    df = aggDatasets(df_datasets, YEARS, OFFICES)

    for target in TARGETS:
        print(f'Processing target {target}')
    
        # If features have been ranked, use only those features ranked
        # high for the target in quest.
        if FEATURES_ALREADY_RANKED:
            print(f'Features already ranked, select top...')
            feature_importance_file = f'data/generated_data/df_importances_{target}.csv'
            top_feature_columns = pd.read_csv(feature_importance_file)['Feature name'].head(TOP_N_FEATURES).tolist()
            required_columns = ['standardized_id_num', target]
            selected_columns = [col for col in top_feature_columns + required_columns if col in df.columns]
            df = df[selected_columns].dropna(subset=[target])
        
        # Ensure clean 13-character left-padded ID
        df['standardized_id_num'] = df['standardized_id_num'].astype(str).str.zfill(13)

        df = df.dropna(subset=[target])
        
        # Save standardized IDs
        X_stids = df['standardized_id_num']
        
        y = df[target]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        cols_to_drop = [col for col in drop_features if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        X = df.drop(columns=[target])
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        for col in categorical_cols:
            X[col] = X[col].astype(str)

        # https://goodboychan.github.io/python/datacamp/machine_learning/2020/07/07/03-Using-XGBoost-in-pipelines.html
        # https://www.kaggle.com/code/carlosdg/xgboost-with-scikit-learn-pipeline-gridsearchcv
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer([
            ('cat', categorical_transformer, categorical_cols),
            ('num', numeric_transformer, numeric_cols)
        ])
        
        xgb_classifier = XGBClassifier(
            objective="multi:softmax",  # multi:softmax or multi:softprob
            num_class=len(y.unique()),
            use_label_encoder=False,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
        
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb_classifier)
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)
        
        xgb_pipeline.fit(X_train, y_train)
        y_pred = xgb_pipeline.predict(X_test)
        
        decoded_y_test = label_encoder.inverse_transform(y_test)
        decoded_y_pred = label_encoder.inverse_transform(y_pred)
        
        class_report = classification_report(decoded_y_test, decoded_y_pred, output_dict=True)
        print(class_report)
        
        df_class_report = pd.DataFrame(class_report).transpose()
        df_class_report = df_class_report.round(3)
        with open(f'output/reports/classification_report_{target}_aggregate_classification.md', 'w') as f:
            f.write("# Classification Report\n\n")
            f.write(df_class_report.to_markdown())
        
        cm = confusion_matrix(decoded_y_test, decoded_y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'output/figures/confusion_matrix_{target}_aggregate.png')
        plt.close()
        # plt.show()
        
        results_df = pd.DataFrame({
            'standardized_id_num': X_stids.loc[X_test.index],
            'true_label': decoded_y_test,
            'predicted_label': decoded_y_pred
        })
        results_df['standardized_id_num'] = results_df['standardized_id_num'].astype(str).str.zfill(13)
        
        filename = f'data/generated_data/prediction_results_{target}_aggregate_classification.csv'
        results_df.to_csv(filename, index=False)
        
        #############################
        # FEATURE PERFORMANCE
        #############################
        from sklearn.linear_model import LogisticRegression
        
        feature_performance = []
        
        for feature in tqdm(X.columns):
            X_feature = X[[feature]].copy()
            
            # Handle missing values
            if X_feature[feature].dtype == 'object':
                X_feature = X_feature.fillna(X_feature.mode().iloc[0])
                X_feature = pd.get_dummies(X_feature, drop_first=True)
            
                # Need at least 1 column after one-hot encoding
                if X_feature.shape[1] == 0:
                    continue  # Leave as is
            else:
                X_feature = X_feature.fillna(X_feature.mean(numeric_only=True))
            
            # Force one column, even if empty
            if X_feature.shape[1] == 0:
                continue
            
            X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(X_feature, y, test_size=0.2, random_state=42)
            
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_feat, y_train_feat)
            y_pred = model.predict(X_test_feat)
            score = accuracy_score(y_test_feat, y_pred)
            metric = "Score"
                
            feature_performance.append({"Feature name": feature, metric: score})
            
        feature_performance_df = pd.DataFrame(feature_performance).sort_values(by=metric, ascending=False)
        filename = f'data/generated_data/feature_rankings_{target}_aggregate_classification.csv'
        feature_performance_df.to_csv(filename, index=None)


# In[ ]:


def getHoldoutDataset(holdout_year=HOLDOUT_YEAR):
    df_holdout_dataset_list = []
    
    for office, years in ELECTIONS.items():
        for year in years:
            if year == holdout_year:
                office = office.replace(' ', '_').replace('.', '')
                df = pd.read_csv('data/generated_data/07_ml_features_' + year + '_' + office + '.csv', low_memory=False)
                df_holdout_dataset_list.append(df)
            else:
                continue
    
    df_holdout = pd.concat(df_holdout_dataset_list, axis=0, ignore_index=True)

    return df_holdout


# In[ ]:


df_benchmark = getHoldoutDataset(HOLDOUT_YEAR)


# In[ ]:


# RULE-OF-THUMB BENCHMARKING
for target in TARGETS:
    RULE_OF_THUMB_FEATURES = ['partisanship_lean_prev']
    
    # Compute average rule-of-thumb if both exist
    rule_cols = RULE_OF_THUMB_FEATURES
    available_rule_cols = [col for col in rule_cols if col in df_benchmark.columns]
    
    if len(available_rule_cols) == 0:
        raise Exception("No rule-of-thumb columns found.")
    
    if len(available_rule_cols) == 2:
        df_benchmark['rule_of_thumb'] = df_benchmark[available_rule_cols].mean(axis=1)
    else:
        df_benchmark['rule_of_thumb'] = df_benchmark[available_rule_cols[0]]
    
    # Round and get back categorical labels
    df_benchmark['rule_of_thumb_rounded'] = df_benchmark['rule_of_thumb'].round()
    
    mask = df_benchmark['rule_of_thumb_rounded'].notna() & df_benchmark[target].notna()
    y_true = df_benchmark.loc[mask, target]
    y_benchmark = df_benchmark.loc[mask, 'rule_of_thumb_rounded']
    
    label_encoder.fit(df_benchmark[target].dropna().astype(str).unique())
    y_true_encoded = label_encoder.transform(y_true.astype(str))
    y_benchmark_encoded = label_encoder.transform(y_benchmark.astype(str))
    
    decoded_y_true = label_encoder.inverse_transform(y_true_encoded)
    decoded_y_benchmark = label_encoder.inverse_transform(y_benchmark_encoded)
    
    print(classification_report(decoded_y_true, decoded_y_benchmark))

    cm = confusion_matrix(decoded_y_true, decoded_y_benchmark)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'output/figures/confusion_matrix_{target}_aggregate_benchmark.png')
    plt.close()
    # plt.show()
    
    df_benchmark_results = pd.DataFrame({
        'standardized_id_num': df_benchmark.loc[mask, 'standardized_id_num'],
        'true_label': y_true,
        'benchmark_label': y_benchmark
    })
    
    filename = f"data/generated_data/benchmark_results_{target}_aggregate.csv"
    df_benchmark_results.to_csv(filename, index=False)
    print(f"Saved benchmark predictions to {filename}")


# In[ ]:


# Predict targets using holdout – U.S. House only
df_holdout = getHoldoutDataset(HOLDOUT_YEAR)

for target in TARGETS:
    years = [HOLDOUT_YEAR]
    offices = ['U.S. House']
    
    df_holdout_benchmark = df_holdout.copy()
    
    if FEATURES_ALREADY_RANKED:
        print(f'Features already ranked, select top...')
        feature_importance_file = f'data/generated_data/df_importances_{target}.csv'
        top_feature_columns = pd.read_csv(feature_importance_file)['Feature name'].head(TOP_N_FEATURES).tolist()
        required_columns = ['standardized_id_num', target]
        selected_columns = [col for col in top_feature_columns + required_columns if col in df_holdout.columns]
        df_holdout = df_holdout[selected_columns].dropna(subset=[target])
    
    df_holdout['standardized_id_num'] = df_holdout['standardized_id_num'].astype(str).str.zfill(13)
    df_holdout = df_holdout.dropna(subset=[target])
    
    # Save standardized IDs
    X_stids = df_holdout['standardized_id_num']
    
    y = df_holdout[target]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    cols_to_drop = [col for col in drop_features if col in df_holdout.columns]
    df_holdout = df_holdout.drop(columns=cols_to_drop)
    X = df_holdout.drop(columns=[target])
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    
    print("Predicting on 2024 holdout set...")
    y_pred_encoded = xgb_pipeline.predict(X)
    
    decoded_y_test = label_encoder.inverse_transform(y_encoded)
    decoded_y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    class_report = classification_report(decoded_y_test, decoded_y_pred, output_dict=True)
    print(classification_report(y_true_encoded, y_benchmark_encoded, target_names=label_encoder.classes_))

    df_class_report = pd.DataFrame(class_report).transpose()
    df_class_report = df_class_report.round(3)
    with open(f'output/reports/classification_report_{target}_holdout.md', 'w') as f:
        f.write("# Classification Report\n\n")
        f.write(df_class_report.to_markdown())
    
    cm = confusion_matrix(decoded_y_test, decoded_y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'output/figures/confusion_matrix_{target}_holdout.png')
    plt.close()
    # plt.show()
    
    results_df = pd.DataFrame({
        'standardized_id_num': X_stids.reset_index(drop=True),
        'true_label': decoded_y_test,
        'predicted_label': decoded_y_pred
    })
    results_df['standardized_id_num'] = results_df['standardized_id_num'].astype(str).str.zfill(13)
    
    filename = f'data/generated_data/predicted_{target}_{HOLDOUT_YEAR}_holdout.csv'
    results_df.to_csv(filename, index=False)


# In[ ]:


# Evaluate benchmark of trained predictor from holdout – U.S. House only
for target in TARGETS:
    years = [HOLDOUT_YEAR]
    offices = ['U.S. House']

    df_holdout_benchmark = df_benchmark.copy()

    RULE_OF_THUMB_FEATURES = ['partisanship_lean_prev']
    
    # Compute average rule-of-thumb if both exist
    rule_cols = RULE_OF_THUMB_FEATURES
    available_rule_cols = [col for col in rule_cols if col in df_holdout_benchmark.columns]
    
    if len(available_rule_cols) == 0:
        raise Exception("No rule-of-thumb columns found.")
    
    if len(available_rule_cols) == 2:
        df_holdout_benchmark['rule_of_thumb'] = df_holdout_benchmark[available_rule_cols].mean(axis=1)
    else:
        df_holdout_benchmark['rule_of_thumb'] = df_holdout_benchmark[available_rule_cols[0]]
    
    # Round and get back categorical labels
    df_holdout_benchmark['rule_of_thumb_rounded'] = df_holdout_benchmark['rule_of_thumb'].round()
    
    mask = df_holdout_benchmark['rule_of_thumb_rounded'].notna() & df_holdout_benchmark[target].notna()
    y_true = df_holdout_benchmark.loc[mask, target]
    y_benchmark = df_holdout_benchmark.loc[mask, 'rule_of_thumb_rounded']
    
    label_encoder.fit(df_holdout_benchmark[target].dropna().astype(str).unique())
    y_true_encoded = label_encoder.transform(y_true.astype(str))
    y_benchmark_encoded = label_encoder.transform(y_benchmark.astype(str))
    
    print("\nRule-of-Thumb Benchmark (Average of Previous Values):")
    print(classification_report(y_true_encoded, y_benchmark_encoded, target_names=label_encoder.classes_))
    
    benchmark_df = pd.DataFrame({
        'standardized_id_num': df_holdout_benchmark.loc[mask, 'standardized_id_num'],
        'true_label': y_true,
        'benchmark_label': y_benchmark
    })
    
    filename = f"data/generated_data/benchmark_results_{target}_{year}_{office}_classification.csv"
    benchmark_df.to_csv(filename, index=False)
    print(f"Saved benchmark predictions to {filename}")


# In[ ]:




