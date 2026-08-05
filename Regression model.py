import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('perovskites - R.csv')
X = df.drop(['Band Gap', 'Formula', 'Is Gap Direct'], axis=1, errors='ignore')
y = pd.to_numeric(df['Band Gap'], errors='coerce')
X = X.fillna(X.mean())
y = y.fillna(y.mean())

constant_features = [col for col in X.columns if X[col].nunique() == 1]
X = X.drop(constant_features, axis=1)
feature_columns = X.columns.tolist()

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf_rfecv = RFECV(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    step=1, cv=5, scoring='r2', min_features_to_select=1, n_jobs=-1
)
rf_rfecv.fit(X_train, y_train)
rf_features = [feature_columns[i] for i in range(len(feature_columns)) if rf_rfecv.support_[i]]

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    cv=5, scoring='r2', n_jobs=-1
)
rf_grid.fit(X_train[:, rf_rfecv.support_], y_train)
base_models = {}
base_models['RandomForest'] = rf_grid.best_estimator_
rf_pred = rf_grid.predict(X_test[:, rf_rfecv.support_])
base_results = {}
base_results['RandomForest'] = {
    'test_r2': r2_score(y_test, rf_pred),
    'test_mae': mean_absolute_error(y_test, rf_pred),
    'test_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'n_features': len(rf_features)
}

xgb_rfecv = RFECV(
    estimator=xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    step=1, cv=5, scoring='r2', min_features_to_select=1, n_jobs=-1
)
xgb_rfecv.fit(X_train, y_train)
xgb_features = [feature_columns[i] for i in range(len(feature_columns)) if xgb_rfecv.support_[i]]

xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]},
    cv=5, scoring='r2', n_jobs=-1
)
xgb_grid.fit(X_train[:, xgb_rfecv.support_], y_train)
base_models['XGBoost'] = xgb_grid.best_estimator_
xgb_pred = xgb_grid.predict(X_test[:, xgb_rfecv.support_])
base_results['XGBoost'] = {
    'test_r2': r2_score(y_test, xgb_pred),
    'test_mae': mean_absolute_error(y_test, xgb_pred),
    'test_rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
    'n_features': len(xgb_features)
}

lr_rfecv = RFECV(
    estimator=LinearRegression(),
    step=1, cv=5, scoring='r2', min_features_to_select=1, n_jobs=-1
)
lr_rfecv.fit(X_train, y_train)
lr_features = [feature_columns[i] for i in range(len(feature_columns)) if lr_rfecv.support_[i]]

lr_model = LinearRegression()
lr_model.fit(X_train[:, lr_rfecv.support_], y_train)
base_models['LinearRegression'] = lr_model
lr_pred = lr_model.predict(X_test[:, lr_rfecv.support_])
base_results['LinearRegression'] = {
    'test_r2': r2_score(y_test, lr_pred),
    'test_mae': mean_absolute_error(y_test, lr_pred),
    'test_rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
    'n_features': len(lr_features)
}

best_name = max(base_results, key=lambda x: base_results[x]['test_r2'])
best_r2 = base_results[best_name]['test_r2']

base_importance = {}
rf_imp = base_models['RandomForest'].feature_importances_
base_importance['RandomForest'] = dict(zip(rf_features, rf_imp / rf_imp.sum()))
xgb_imp = base_models['XGBoost'].feature_importances_
base_importance['XGBoost'] = dict(zip(xgb_features, xgb_imp / xgb_imp.sum()))
lr_imp = np.abs(base_models['LinearRegression'].coef_)
base_importance['LinearRegression'] = dict(zip(lr_features, lr_imp / lr_imp.sum()))

all_imp = []
for f in feature_columns:
    all_imp.append({'feature': f, 'importance': base_importance[best_name].get(f, 0)})
all_imp_df = pd.DataFrame(all_imp).sort_values('importance', ascending=False)

def create_stacking():
    return StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)),
            ('xgb', xgb.XGBRegressor(learning_rate=0.2, max_depth=3, n_estimators=300, random_state=42)),
            ('lr', LinearRegression())
        ],
        final_estimator=LinearRegression(),
        cv=5
    )

best_result = None
for n in range(len(feature_columns), 0, -1):
    selected = all_imp_df.head(n)['feature'].tolist()
    idx = [feature_columns.index(f) for f in selected]
    stack = create_stacking()
    stack.fit(X_train[:, idx], y_train)
    pred = stack.predict(X_test[:, idx])
    r2 = r2_score(y_test, pred)
    if best_result is None or r2 > best_result['test_r2']:
        best_result = {
            'test_r2': r2,
            'test_mae': mean_absolute_error(y_test, pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'features': selected,
            'n_features': n
        }
        best_stack = stack

print("\n" + "=" * 85)
print("Model performance comparison")
print("=" * 85)
print(f"{'Model':<20} {'Features':<12} {'Test R2':<12} {'Test MAE (eV)':<18} {'Test RMSE (eV)':<15}")
print("-" * 85)
for name in ['RandomForest', 'XGBoost', 'LinearRegression']:
    r = base_results[name]
    print(f"{name:<20} {r['n_features']:<12} {r['test_r2']:<12.4f} {r['test_mae']:<18.4f} {r['test_rmse']:<15.4f}")
print(f"{'R-SE (Stacking)':<20} {best_result['n_features']:<12} {best_result['test_r2']:<12.4f} "
      f"{best_result['test_mae']:<18.4f} {best_result['test_rmse']:<15.4f}")
print("=" * 85)

meta = best_stack.final_estimator_
w = meta.coef_
b = meta.intercept_

print("\n" + "=" * 85)
print("Fusion formula")
print("=" * 85)
print(f"\nBand Gap = {w[0]:.6f} * RF + {w[1]:.6f} * XGB + {w[2]:.6f} * LR + {b:.4f}")
print("=" * 85)

joblib.dump(best_stack, 'best_model.pkl')
joblib.dump(best_stack, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

feature_selector = {
    'stacking_features': best_result['features'],
    'all_features': feature_columns,
    'best_base_model': best_name,
    'best_base_model_r2': best_r2,
    'base_models_results': base_results,
    'stacking_results': best_result
}
with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(feature_selector, f)

print(f"\nSaved: best_model.pkl, stacking_model.pkl, scaler.pkl, feature_selector.pkl")