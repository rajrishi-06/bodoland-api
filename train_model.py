import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load and preprocess data
data = pd.read_csv('data/new_synthetic_data_credit_assigned.csv')
X = data.drop(columns=['target'])
y = data['target']

transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), [
            'GENDER', 'MARITAL STATUS', 'ARE YOU THE PRIMARY EARNER OF YOUR FAMILY ?',
            'SKILL 1', 'SKILL 2', 'SKILL 3',
            'DO YOU HAVE ANY CERTIFICATION OF THE ABOVE-MENTIONED SKILL SET?',
            'OWNERSHIP ( includes Land,machine)',
            'Relation with primary earner ?'
        ]),
        ('ordinal', OrdinalEncoder(categories=[[
            'Class III','Class IV','Class V','Class v','ClassV','Class VI','Class VII',
            'Class VIII','Class IX','HS','HSLC','BA Ongoing','BCom','B Com','BA ','BA',
            'BSc','MSc','MCA','PhD'
        ]]), ['WHAT IS YOUR HIGHEST EDUCATIONAL QUALIFICATION?']),
    ],
    remainder='passthrough'
)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit transformer + model
X_train_t = transformer.fit_transform(X_train)
rf = RandomForestClassifier(
    criterion='entropy', max_depth=17, max_features=15,
    min_samples_leaf=6, n_estimators=50, random_state=42
)
rf.fit(X_train_t, y_train)

# Save pipeline
pipeline = {
    'transformer': transformer,
    'model': rf
}
joblib.dump(pipeline, 'pipeline.joblib')
print("Pipeline saved to pipeline.joblib")