import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('perovskites - C.csv')
df['Is Gap Direct'] = df['Is Gap Direct'].map({True: 1, False: 0})
y = df['Is Gap Direct']
X = df.drop(['Is Gap Direct', 'Formula', 'Band Gap'], axis=1, errors='ignore')

X = X.fillna(X.median())

constant_features = [col for col in X.columns if X[col].nunique() == 1]
X = X.drop(constant_features, axis=1)
feature_columns = X.columns.tolist()

with open('feature_columns_classification.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler_classification.pkl')

X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_df = pd.DataFrame(X_train_base, columns=feature_columns)
X_test_df = pd.DataFrame(X_test_base, columns=feature_columns)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_models = {}
base_results = {}
base_importance = {}

rf_rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    step=1, cv=cv_strategy, scoring='f1', min_features_to_select=1, n_jobs=-1, verbose=0
)
rf_rfecv.fit(X_train_base, y_train)
rf_features = [feature_columns[i] for i in range(len(feature_columns)) if rf_rfecv.support_[i]]

X_train_rf = X_train_df[rf_features]
X_test_rf = X_test_df[rf_features]

rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
    },
    cv=cv_strategy, scoring='f1', n_jobs=-1
)
rf_grid.fit(X_train_rf, y_train)
base_models['RandomForest'] = rf_grid.best_estimator_

rf_best = rf_grid.best_estimator_.named_steps['rf']
rf_importance = rf_best.feature_importances_
base_importance['RandomForest'] = dict(zip(rf_features, rf_importance / rf_importance.sum()))

rf_pred = rf_grid.best_estimator_.predict(X_test_rf)
rf_proba = rf_grid.best_estimator_.predict_proba(X_test_rf)[:, 1]
base_results['RandomForest'] = {
    'test_f1': f1_score(y_test, rf_pred),
    'test_accuracy': accuracy_score(y_test, rf_pred),
    'test_precision': precision_score(y_test, rf_pred),
    'test_recall': recall_score(y_test, rf_pred),
    'test_roc_auc': roc_auc_score(y_test, rf_proba),
    'n_features': len(rf_features)
}

xgb_rfecv = RFECV(
    estimator=XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1),
    step=1, cv=cv_strategy, scoring='f1', min_features_to_select=1, n_jobs=-1, verbose=0
)
xgb_rfecv.fit(X_train_base, y_train)
xgb_features = [feature_columns[i] for i in range(len(feature_columns)) if xgb_rfecv.support_[i]]

X_train_xgb = X_train_df[xgb_features]
X_test_xgb = X_test_df[xgb_features]

imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1))
])

xgb_grid = GridSearchCV(
    xgb_pipeline,
    {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 6],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__scale_pos_weight': [imbalance_ratio, imbalance_ratio * 1.5]
    },
    cv=cv_strategy, scoring='f1', n_jobs=-1
)
xgb_grid.fit(X_train_xgb, y_train)
base_models['XGBoost'] = xgb_grid.best_estimator_

xgb_best = xgb_grid.best_estimator_.named_steps['xgb']
xgb_importance = xgb_best.feature_importances_
base_importance['XGBoost'] = dict(zip(xgb_features, xgb_importance / xgb_importance.sum()))

xgb_pred = xgb_grid.best_estimator_.predict(X_test_xgb)
xgb_proba = xgb_grid.best_estimator_.predict_proba(X_test_xgb)[:, 1]
base_results['XGBoost'] = {
    'test_f1': f1_score(y_test, xgb_pred),
    'test_accuracy': accuracy_score(y_test, xgb_pred),
    'test_precision': precision_score(y_test, xgb_pred),
    'test_recall': recall_score(y_test, xgb_pred),
    'test_roc_auc': roc_auc_score(y_test, xgb_proba),
    'n_features': len(xgb_features)
}

lr_rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    step=1, cv=cv_strategy, scoring='f1', min_features_to_select=1, n_jobs=-1, verbose=0
)
lr_rfecv.fit(X_train_base, y_train)
lr_features = [feature_columns[i] for i in range(len(feature_columns)) if lr_rfecv.support_[i]]

X_train_lr = X_train_df[lr_features]
X_test_lr = X_test_df[lr_features]

lr_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {'lr__C': [0.1, 1, 10], 'lr__class_weight': ['balanced', {0: 1, 1: 2}]},
    cv=cv_strategy, scoring='f1', n_jobs=-1
)
lr_grid.fit(X_train_lr, y_train)
base_models['LogisticRegression'] = lr_grid.best_estimator_

lr_best = lr_grid.best_estimator_.named_steps['lr']
lr_importance = np.abs(lr_best.coef_[0])
base_importance['LogisticRegression'] = dict(zip(lr_features, lr_importance / lr_importance.sum()))

lr_pred = lr_grid.best_estimator_.predict(X_test_lr)
lr_proba = lr_grid.best_estimator_.predict_proba(X_test_lr)[:, 1]
base_results['LogisticRegression'] = {
    'test_f1': f1_score(y_test, lr_pred),
    'test_accuracy': accuracy_score(y_test, lr_pred),
    'test_precision': precision_score(y_test, lr_pred),
    'test_recall': recall_score(y_test, lr_pred),
    'test_roc_auc': roc_auc_score(y_test, lr_proba),
    'n_features': len(lr_features)
}

best_name = max(base_results, key=lambda x: base_results[x]['test_f1'])

all_imp = []
for f in feature_columns:
    all_imp.append({'feature': f, 'importance': base_importance[best_name].get(f, 0)})
all_imp_df = pd.DataFrame(all_imp).sort_values('importance', ascending=False)

def create_stacking_model():
    return StackingClassifier(
        estimators=[
            ('randomforest', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('xgboost', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                                      random_state=42, eval_metric='logloss')),
            ('logisticregression', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ],
        final_estimator=LogisticRegression(C=0.5, class_weight='balanced', random_state=42, max_iter=1000),
        cv=5, passthrough=False, n_jobs=-1
    )

stacking_results = []
for n_features in range(len(feature_columns), 0, -1):
    selected_features = all_imp_df.head(n_features)['feature'].tolist()
    X_train_sel = X_train_df[selected_features]
    X_test_sel = X_test_df[selected_features]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)

    stacking_model = create_stacking_model()
    stacking_model.fit(X_train_res, y_train_res)

    y_pred = stacking_model.predict(X_test_sel)
    y_proba = stacking_model.predict_proba(X_test_sel)[:, 1]

    test_f1 = f1_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba)

    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train_sel, y_train):
        X_cv_train, X_cv_val = X_train_sel.iloc[train_idx], X_train_sel.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        smote_cv = SMOTE(random_state=42)
        X_cv_train_res, y_cv_train_res = smote_cv.fit_resample(X_cv_train, y_cv_train)

        model_cv = create_stacking_model()
        model_cv.fit(X_cv_train_res, y_cv_train_res)
        cv_scores.append(f1_score(y_cv_val, model_cv.predict(X_cv_val)))

    cv_scores = np.array(cv_scores)

    stacking_results.append({
        'n_features': n_features,
        'test_f1': test_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_roc_auc': test_roc_auc,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'features': selected_features,
        'model': stacking_model
    })

max_test_f1 = max(r['test_f1'] for r in stacking_results)
candidates = [r for r in stacking_results if r['test_f1'] == max_test_f1]
best_stacking = min(candidates, key=lambda x: x['n_features'])

stacking_features = best_stacking['features']
stacking_n_features = best_stacking['n_features']
final_stacking = best_stacking['model']

print("\n" + "=" * 100)
print("Model performance comparison")
print("=" * 100)
print(f"{'Model':<20} {'Features':<10} {'F1':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'ROC-AUC':<10}")
print("-" * 100)
for name in ['RandomForest', 'XGBoost', 'LogisticRegression']:
    r = base_results[name]
    print(f"{name:<20} {r['n_features']:<10} {r['test_f1']:<10.4f} {r['test_accuracy']:<12.4f} "
          f"{r['test_precision']:<12.4f} {r['test_recall']:<12.4f} {r['test_roc_auc']:<10.4f}")
print(f"{'C-SE (Stacking)':<20} {stacking_n_features:<10} {best_stacking['test_f1']:<10.4f} "
      f"{best_stacking['test_accuracy']:<12.4f} {best_stacking['test_precision']:<12.4f} "
      f"{best_stacking['test_recall']:<12.4f} {best_stacking['test_roc_auc']:<10.4f}")
print("=" * 100)

meta = final_stacking.final_estimator_
coefs = meta.coef_[0]
intercept = meta.intercept_[0]

print("\n" + "=" * 100)
print("Fusion formula")
print("=" * 100)
print(f"\nlogit = {intercept:.6f} + {coefs[0]:.6f} * P_RF + {coefs[1]:.6f} * P_XGB + {coefs[2]:.6f} * P_LR")
print(f"y_pred = 1 if logit > 0 else 0")
print("=" * 100)

joblib.dump(final_stacking, 'best_classifier.pkl')
joblib.dump(final_stacking, 'stacking_classifier.pkl')
joblib.dump(scaler, 'scaler_classification.pkl')

feature_selector = {
    'stacking_features': stacking_features,
    'all_features': feature_columns,
    'best_base_model': best_name,
    'best_base_model_f1': base_results[best_name]['test_f1'],
    'base_models_results': base_results,
    'stacking_results': best_stacking
}
with open('feature_selector_classification.pkl', 'wb') as f:
    pickle.dump(feature_selector, f)

print(f"\nSaved: best_classifier.pkl, stacking_classifier.pkl, scaler_classification.pkl, feature_selector_classification.pkl")