"""Example preprocessing boundary. This does not train or claim a model score."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from furcast_data import AAC_FEATURES, feature_guard, verify_splits

def fit_austin_preprocessor(features, labels):
    feature_guard(features.columns.drop('episode_id'),'austin')
    verify_splits(labels,'episode_id','animal_id')
    merged=features.merge(labels[['episode_id','split']],on='episode_id',validate='one_to_one')
    numeric=['age_days_approx','intake_month']
    categorical=[x for x in AAC_FEATURES if x not in numeric]
    processor=ColumnTransformer([
       ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                        ('scale',StandardScaler())]),numeric),
       ('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='__MISSING__')),
                        ('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
    train=merged.loc[merged.split.eq('train'),AAC_FEATURES]
    if train.empty:raise ValueError('No training rows')
    processor.fit(train)
    # Validation/test only call transform; no refitting or resampling outside train.
    return processor,{s:processor.transform(merged.loc[merged.split.eq(s),AAC_FEATURES])
                      for s in ['train','validation','test']}
