import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('4373.csv')
formulas = df['formula'].copy() if 'formula' in df.columns else df['Formula'].copy() if 'Formula' in df.columns else pd.Series(['Material_{}'.format(i+1) for i in range(len(df))])

drop_cols = ['formula', 'Formula']
X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
X = X.fillna(X.median()) if X.isnull().sum().sum() > 0 else X

reg_model = joblib.load('best_model.pkl')
reg_scaler = joblib.load('scaler.pkl')
with open('feature_selector.pkl', 'rb') as f:
    reg_selector = pickle.load(f)
reg_all_features = reg_selector['all_features']
reg_features = reg_selector['stacking_features']

missing_reg = set(reg_all_features) - set(X.columns)
for f in missing_reg:
    X[f] = 0
X_reg = X[reg_all_features].copy()
X_reg_scaled = reg_scaler.transform(X_reg)
X_reg_final = pd.DataFrame(X_reg_scaled, columns=reg_all_features)[reg_features]

cls_model = joblib.load('best_classifier.pkl')
cls_scaler = joblib.load('scaler_classification.pkl')
with open('feature_selector_classification.pkl', 'rb') as f:
    cls_selector = pickle.load(f)
cls_all_features = cls_selector['all_features']
cls_features = cls_selector['stacking_features']

missing_cls = set(cls_all_features) - set(X.columns)
for f in missing_cls:
    X[f] = 0
X_cls = X[cls_all_features].copy()
X_cls_scaled = cls_scaler.transform(X_cls)
X_cls_final = pd.DataFrame(X_cls_scaled, columns=cls_all_features)[cls_features]

Z2 = reg_model.predict(X_reg_final)

meta = cls_model.final_estimator_
coefs = meta.coef_[0]
intercept = meta.intercept_[0]

rf = cls_model.named_estimators_['randomforest']
xgb = cls_model.named_estimators_['xgboost']
lr = cls_model.named_estimators_['logisticregression']

proba_rf = rf.predict_proba(X_cls_final)[:, 1]
proba_xgb = xgb.predict_proba(X_cls_final)[:, 1]
proba_lr = lr.predict_proba(X_cls_final)[:, 1]

Z1 = intercept + coefs[0] * proba_rf + coefs[1] * proba_xgb + coefs[2] * proba_lr

results = pd.DataFrame({
    'Formula': formulas.values,
    'Z1': Z1,
    'Z2': Z2
})

results.to_csv('Z1Z2_predictions.csv', index=False)
print('Z1Z2_predictions.csv')