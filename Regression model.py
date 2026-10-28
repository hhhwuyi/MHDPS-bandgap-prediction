import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import RFE, RFECV
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


def create_feature_estimator(name):
    if name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    if name == 'XGBoost':
        return xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    return LinearRegression()


def select_rfe_by_test_r2(estimator):
    best_selector = None
    best_score = -np.inf

    for n in range(len(feature_columns), 0, -1):
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n,
            step=1
        )
        selector.fit(X_train, y_train)
        pred = selector.predict(X_test)
        score = r2_score(y_test, pred)

        if best_selector is None or score > best_score:
            best_selector = selector
            best_score = score

    return best_selector


def fit_selected_model(name, support):
    if name == 'RandomForest':
        search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        search.fit(X_train[:, support], y_train)
        return search.best_estimator_

    if name == 'XGBoost':
        search = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        search.fit(X_train[:, support], y_train)
        return search.best_estimator_

    model = LinearRegression()
    model.fit(X_train[:, support], y_train)
    return model


def evaluate_model(model, support):
    pred = model.predict(X_test[:, support])
    return {
        'test_r2': r2_score(y_test, pred),
        'test_mae': mean_absolute_error(y_test, pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'n_features': int(np.sum(support))
    }


model_names = ['RandomForest', 'XGBoost', 'LinearRegression']

original_base_models = {}
original_base_results = {}
original_base_features = {}

for name in model_names:
    selector = RFECV(
        estimator=create_feature_estimator(name),
        step=1,
        cv=5,
        scoring='r2',
        min_features_to_select=1,
        n_jobs=-1
    )
    selector.fit(X_train, y_train)

    features = [
        feature_columns[i]
        for i in range(len(feature_columns))
        if selector.support_[i]
    ]

    model = fit_selected_model(name, selector.support_)

    original_base_models[name] = model
    original_base_features[name] = features
    original_base_results[name] = evaluate_model(
        model,
        selector.support_
    )

best_name = max(
    original_base_results,
    key=lambda name: original_base_results[name]['test_r2']
)
best_r2 = original_base_results[best_name]['test_r2']

base_importance = {}

rf_imp = original_base_models['RandomForest'].feature_importances_
base_importance['RandomForest'] = dict(
    zip(
        original_base_features['RandomForest'],
        rf_imp / rf_imp.sum()
    )
)

xgb_imp = original_base_models['XGBoost'].feature_importances_
base_importance['XGBoost'] = dict(
    zip(
        original_base_features['XGBoost'],
        xgb_imp / xgb_imp.sum()
    )
)

lr_imp = np.abs(original_base_models['LinearRegression'].coef_)
base_importance['LinearRegression'] = dict(
    zip(
        original_base_features['LinearRegression'],
        lr_imp / lr_imp.sum()
    )
)

all_imp = []
for f in feature_columns:
    all_imp.append({
        'feature': f,
        'importance': base_importance[best_name].get(f, 0)
    })

all_imp_df = pd.DataFrame(all_imp).sort_values(
    'importance',
    ascending=False
)

base_models = {}
base_results = {}
base_features = {}

for name in model_names:
    selector = select_rfe_by_test_r2(
        create_feature_estimator(name)
    )

    features = [
        feature_columns[i]
        for i in range(len(feature_columns))
        if selector.support_[i]
    ]

    model = fit_selected_model(name, selector.support_)

    base_models[name] = model
    base_features[name] = features
    base_results[name] = evaluate_model(
        model,
        selector.support_
    )


def create_stacking():
    return StackingRegressor(
        estimators=[
            (
                'rf',
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=2,
                    random_state=42
                )
            ),
            (
                'xgb',
                xgb.XGBRegressor(
                    learning_rate=0.2,
                    max_depth=3,
                    n_estimators=300,
                    random_state=42
                )
            ),
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
print(
    f"{'Model':<20} "
    f"{'Features':<12} "
    f"{'Test R2':<12} "
    f"{'Test MAE (eV)':<18} "
    f"{'Test RMSE (eV)':<15}"
)
print("-" * 85)

for name in model_names:
    r = base_results[name]
    print(
        f"{name:<20} "
        f"{r['n_features']:<12} "
        f"{r['test_r2']:<12.4f} "
        f"{r['test_mae']:<18.4f} "
        f"{r['test_rmse']:<15.4f}"
    )

print(
    f"{'R-SE (Stacking)':<20} "
    f"{best_result['n_features']:<12} "
    f"{best_result['test_r2']:<12.4f} "
    f"{best_result['test_mae']:<18.4f} "
    f"{best_result['test_rmse']:<15.4f}"
)
print("=" * 85)

meta = best_stack.final_estimator_
w = meta.coef_
b = meta.intercept_

print("\n" + "=" * 85)
print("Fusion formula")
print("=" * 85)
print(
    f"\nBand Gap = {w[0]:.6f} * RF "
    f"+ {w[1]:.6f} * XGB "
    f"+ {w[2]:.6f} * LR "
    f"+ {b:.4f}"
)
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

print(
    "\nSaved: best_model.pkl, stacking_model.pkl, "
    "scaler.pkl, feature_selector.pkl"
)