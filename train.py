"""
Wine Quality Prediction - Random Forest
Trains ONLY on WineQT.csv dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("🍷  WINE QUALITY PREDICTION — RANDOM FOREST")
print("=" * 80)

# ============================================================
# CONFIG
# ============================================================
BINARY_CLASSIFICATION = True
QUALITY_THRESHOLD = 6
CV_FOLDS = 5

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n📊 Step 1: Loading WineQT.csv...")

if not os.path.exists("WineQT.csv"):
    print("❌ ERROR: WineQT.csv not found!")
    print("Please make sure WineQT.csv is in the same directory as this script.")
    exit(1)

df = pd.read_csv("WineQT.csv")
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

print(f"✓ Loaded: {len(df)} samples")
print(f"✓ Features: {df.shape[1] - 1}")
print(f"✓ Columns: {list(df.columns)}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n🔧 Step 2: Feature Engineering...")

def create_features(df_input):
    """Create engineered features"""
    df_new = df_input.copy()

    # Interaction features
    df_new['alcohol_sugar'] = df_new['alcohol'] * df_new['residual sugar']
    df_new['alcohol_acidity'] = df_new['alcohol'] * df_new['fixed acidity']
    df_new['alcohol_volatile'] = df_new['alcohol'] * df_new['volatile acidity']
    df_new['density_alcohol'] = df_new['density'] / (df_new['alcohol'] + 0.001)
    df_new['sulphates_alcohol'] = df_new['sulphates'] * df_new['alcohol']

    # Ratio features
    df_new['free_so2_ratio'] = df_new['free sulfur dioxide'] / (df_new['total sulfur dioxide'] + 1)
    df_new['volatile_fixed_ratio'] = df_new['volatile acidity'] / (df_new['fixed acidity'] + 0.001)
    df_new['citric_fixed_ratio'] = df_new['citric acid'] / (df_new['fixed acidity'] + 0.001)
    df_new['sugar_alcohol_ratio'] = df_new['residual sugar'] / (df_new['alcohol'] + 0.001)
    df_new['chlorides_sulphates_ratio'] = df_new['chlorides'] / (df_new['sulphates'] + 0.001)

    # Polynomial features
    df_new['alcohol_squared'] = df_new['alcohol'] ** 2
    df_new['alcohol_cubed'] = df_new['alcohol'] ** 3
    df_new['sulphates_squared'] = df_new['sulphates'] ** 2
    df_new['volatile_squared'] = df_new['volatile acidity'] ** 2
    df_new['density_squared'] = df_new['density'] ** 2

    # Chemical balance
    df_new['total_acidity'] = df_new['fixed acidity'] + df_new['volatile acidity'] + df_new['citric acid']
    df_new['acidity_to_ph'] = df_new['total_acidity'] * df_new['pH']
    df_new['acidity_balance'] = df_new['fixed acidity'] - df_new['volatile acidity']
    df_new['acid_sugar_balance'] = df_new['total_acidity'] / (df_new['residual sugar'] + 1)

    # Sulfur features
    df_new['bound_so2'] = df_new['total sulfur dioxide'] - df_new['free sulfur dioxide']
    df_new['so2_intensity'] = df_new['total sulfur dioxide'] * df_new['sulphates']

    # Quality flags
    df_new['high_alcohol'] = (df_new['alcohol'] > 11).astype(int)
    df_new['low_volatile'] = (df_new['volatile acidity'] < 0.4).astype(int)
    df_new['optimal_ph'] = ((df_new['pH'] >= 3.0) & (df_new['pH'] <= 3.5)).astype(int)
    df_new['quality_score'] = df_new['high_alcohol'] + df_new['low_volatile'] + df_new['optimal_ph']

    # Log transforms
    df_new['log_residual_sugar'] = np.log1p(df_new['residual sugar'])
    df_new['log_chlorides'] = np.log1p(df_new['chlorides'])
    df_new['log_free_sulfur'] = np.log1p(df_new['free sulfur dioxide'])
    df_new['log_total_sulfur'] = np.log1p(df_new['total sulfur dioxide'])

    return df_new

df_eng = create_features(df)
print(f"✓ Created {df_eng.shape[1] - df.shape[1]} new features")
print(f"✓ Total features: {df_eng.shape[1] - 1}")

# ============================================================
# 3. PREPARE DATA
# ============================================================
print("\n📦 Step 3: Preparing Data...")

# Create target
if BINARY_CLASSIFICATION:
    y = (df_eng['quality'] >= QUALITY_THRESHOLD).astype(int)
    class_names = [f'Below Standard (<{QUALITY_THRESHOLD})', f'Good (≥{QUALITY_THRESHOLD})']
else:
    y = df_eng['quality']
    class_names = [str(i) for i in sorted(y.unique())]

print(f"✓ Mode: {'Binary' if BINARY_CLASSIFICATION else 'Multi-class'}")
print(f"✓ Classes: {class_names}")

X = df_eng.drop('quality', axis=1)
feature_names = X.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print(f"✓ SMOTE applied: {len(X_train_bal)} samples")

# Save preprocessing
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(feature_names, 'model/feature_names.pkl')
joblib.dump(create_features, 'model/feature_engineering_fn.pkl')
joblib.dump(BINARY_CLASSIFICATION, 'model/binary_mode.pkl')
joblib.dump(QUALITY_THRESHOLD, 'model/quality_threshold.pkl')

# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================
print("\n🌲 Step 4: Training Random Forest...")

# Hyperparameter grid
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Grid search
print("🔍 Searching for best parameters...")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf, param_grid, cv=skf, scoring='accuracy', 
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train_bal, y_train_bal)

# Best model
best_rf = grid_search.best_estimator_
cv_acc = grid_search.best_score_
test_acc = best_rf.score(X_test_scaled, y_test)

print(f"\n✓ Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n✓ CV Accuracy:   {cv_acc:.4f} ({cv_acc*100:.2f}%)")
print(f"✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n📊 Step 5: Evaluation...")

y_pred = best_rf.predict(X_test_scaled)
print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Random Forest\nAccuracy: {test_acc:.2%}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix saved")

# Feature Importance
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': best_rf.feature_importances_
})
fi_df = fi_df.sort_values('importance', ascending=False)
fi_df.to_csv('model/feature_importance.csv', index=False)

plt.figure(figsize=(12, 8))
top20 = fi_df.head(20)
plt.barh(range(len(top20)), top20['importance'], 
         color=plt.cm.viridis(np.linspace(0, 1, len(top20))))
plt.yticks(range(len(top20)), top20['feature'])
plt.gca().invert_yaxis()
plt.title('Top 20 Important Features', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('model/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Feature importance saved")

# ============================================================
# 6. SAVE MODEL
# ============================================================
print("\n💾 Step 6: Saving Model...")

joblib.dump(best_rf, 'model/best_model.pkl')

metadata = {
    'model_name': 'Random Forest',
    'test_accuracy': float(test_acc),
    'cv_accuracy': float(cv_acc),
    'feature_count': len(feature_names),
    'binary_mode': BINARY_CLASSIFICATION,
    'threshold': QUALITY_THRESHOLD,
    'classes': class_names,
    'best_params': grid_search.best_params_
}
joblib.dump(metadata, 'model/model_metadata.pkl')

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print(f"🌲 Random Forest | {test_acc*100:.2f}% Accuracy")
print("\nFiles saved in model/ directory:")
print("  ✓ best_model.pkl")
print("  ✓ scaler.pkl")
print("  ✓ feature_names.pkl")
print("  ✓ feature_engineering_fn.pkl")
print("  ✓ model_metadata.pkl")
print("  ✓ binary_mode.pkl")
print("  ✓ quality_threshold.pkl")
print("  ✓ confusion_matrix.png")
print("  ✓ feature_importance.csv")
print("  ✓ feature_importance.png")
print("=" * 80)
print("\n🚀 Run:  streamlit run app.py")
