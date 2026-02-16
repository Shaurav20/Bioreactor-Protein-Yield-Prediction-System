"""
Simple Bioreactor Protein Yield Prediction Model
with Data Cleansing Step
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================
# 1. LOAD YOUR DATA (replace with your file path)
# ============================================
df = pd.read_csv('bioreactor_batch_data.csv')

# Intentionally add some data quality issues for demonstration
df.loc[10:15, 'Temperature_C'] = np.nan  # Missing values
df.loc[20:25, 'pH'] = 15.0  # Outlier (impossible pH)
df.loc[30:35, 'DO_percent'] = -10  # Outlier (negative DO)
df.loc[40:45, 'Protein_Yield_gL'] = 50.0  # Outlier (impossible yield)
df.loc[50, 'Feed_Rate_gLh'] = np.nan  # Missing value

print("="*60)
print("BIOREACTOR YIELD PREDICTION MODEL WITH DATA CLEANSING")
print("="*60)

# ============================================
# 2. DATA CLEANSING STEP
# ============================================
print("\n" + "="*60)
print("STEP 1: DATA CLEANSING")
print("="*60)

# Make a copy to avoid modifying original
df_clean = df.copy()

# Display initial data quality
initial_rows = len(df_clean)
print(f"\nInitial dataset: {initial_rows} rows")
print(f"Initial missing values:")
print(df_clean.isnull().sum())

# 2.1 Handle missing values
print("\n1. Handling missing values...")
for col in df_clean.columns:
    if df_clean[col].isnull().any():
        # Skip columns that are completely empty
        if df_clean[col].notna().sum() > 0:
            # Fill numeric columns with median
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"   ✓ {col}: filled {df[col].isnull().sum()} missing values with median ({median_val:.2f})")
        else:
            # If column is all NaN, fill with 0 or drop
            df_clean[col] = df_clean[col].fillna(0)
            print(f"   ⚠ {col}: column was all NaN, filled with 0")

# 2.2 Handle outliers
print("\n2. Handling outliers...")

# Define reasonable ranges for bioreactor parameters - using actual column names
outlier_rules = {
    'Temperature_C': (35, 39),  # °C
    'pH': (6.5, 7.5),           # pH units
    'DO_percent': (30, 100),    # % saturation
    'Feed_Rate_gLh': (0.3, 2.5), # g/L/hr
    'Protein_Yield_gL': (0, 15)  # g/L
}

outliers_count = 0
for col, (lower, upper) in outlier_rules.items():
    if col in df_clean.columns:
        # Cap outliers at reasonable bounds
        before_count = len(df_clean[~df_clean[col].between(lower, upper)])
        df_clean[col] = df_clean[col].clip(lower, upper)
        if before_count > 0:
            print(f"   ✓ {col}: capped {before_count} outliers to [{lower}, {upper}]")
            outliers_count += before_count

if outliers_count == 0:
    print("   ✓ No outliers detected")

# 2.3 Remove duplicates
print("\n3. Removing duplicates...")
duplicates = df_clean.duplicated().sum()
df_clean.drop_duplicates(inplace=True)
print(f"   ✓ Removed {duplicates} duplicate rows")

# 2.4 Drop the artificially created empty columns (if they exist)
columns_to_drop = ['Temperature', 'DO', 'Yield', 'Feed_Rate']
for col in columns_to_drop:
    if col in df_clean.columns:
        df_clean.drop(columns=[col], inplace=True)
        print(f"   ✓ Dropped artificial column: {col}")

# 2.5 Reset index after cleaning
df_clean.reset_index(drop=True, inplace=True)

# Summary of cleansing
print(f"\n📊 Data Cleansing Summary:")
print(f"   - Initial rows: {initial_rows}")
print(f"   - Final rows: {len(df_clean)}")
print(f"   - Rows removed: {initial_rows - len(df_clean)}")
print(f"   - Missing values: 0")
print(f"   - Outliers handled: {outliers_count}")

# ============================================
# 3. PREPARE FEATURES AND TARGET
# ============================================
print("\n" + "="*60)
print("STEP 2: FEATURE PREPARATION")
print("="*60)

feature_cols = ['Feed_Rate_gLh', 'Time_h', 'Temperature_C', 'pH', 'DO_percent']
X = df_clean[feature_cols]
y = df_clean['Protein_Yield_gL']

print(f"\nFeatures: {feature_cols}")
print(f"Target: Protein Yield (g/L)")
print(f"Clean dataset size: {len(df_clean)} samples")

# Quick statistics check
print(f"\nFeature statistics after cleansing:")
print(X.describe().round(2))

# ============================================
# 4. SPLIT DATA (80% train, 20% test)
# ============================================
print("\n" + "="*60)
print("STEP 3: TRAIN/TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# ============================================
# 5. TRAIN RANDOM FOREST MODEL
# ============================================
print("\n" + "="*60)
print("STEP 4: MODEL TRAINING")
print("="*60)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Random Forest model trained successfully")

# ============================================
# 6. MAKE PREDICTIONS
# ============================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ============================================
# 7. EVALUATE MODEL ACCURACY
# ============================================
print("\n" + "="*60)
print("STEP 5: MODEL EVALUATION")
print("="*60)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n{'Metric':<20} {'Training':<15} {'Test':<15}")
print("-"*50)
print(f"{'R² Score':<20} {train_r2:.4f}        {test_r2:.4f}")
print(f"{'RMSE (g/L)':<20} {train_rmse:.4f}        {test_rmse:.4f}")
print(f"{'MAE (g/L)':<20} {train_mae:.4f}        {test_mae:.4f}")

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================
print("\n" + "="*60)
print("STEP 6: FEATURE IMPORTANCE")
print("="*60)

importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature rankings:")
for idx, row in importance.iterrows():
    print(f"  {idx+1}. {row['Feature']:15}: {row['Importance']:.3f} ({row['Importance']*100:.1f}%)")

# ============================================
# 9. PLOT RESULTS
# ============================================
print("\n" + "="*60)
print("STEP 7: GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Bioreactor Yield Prediction Model Performance', fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted (Test)
axes[0,0].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
               'r--', lw=2, label='Perfect prediction')
axes[0,0].set_xlabel('Actual Yield (g/L)', fontsize=11)
axes[0,0].set_ylabel('Predicted Yield (g/L)', fontsize=11)
axes[0,0].set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.3f}', fontsize=12)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_test_pred
axes[0,1].scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0,1].set_xlabel('Predicted Yield (g/L)', fontsize=11)
axes[0,1].set_ylabel('Residual (Actual - Predicted)', fontsize=11)
axes[0,1].set_title(f'Residual Plot\nRMSE = {test_rmse:.3f} g/L', fontsize=12)
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Feature Importance
colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
axes[1,0].barh(importance['Feature'], importance['Importance'], color=colors)
axes[1,0].set_xlabel('Importance', fontsize=11)
axes[1,0].set_title('Feature Importance', fontsize=12)
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Prediction Distribution
axes[1,1].hist(y_test, alpha=0.5, label='Actual', bins=20, color='blue', edgecolor='black')
axes[1,1].hist(y_test_pred, alpha=0.5, label='Predicted', bins=20, color='orange', edgecolor='black')
axes[1,1].set_xlabel('Yield (g/L)', fontsize=11)
axes[1,1].set_ylabel('Frequency', fontsize=11)
axes[1,1].set_title('Actual vs Predicted Distribution', fontsize=12)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_with_cleansing.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Performance plot saved as 'model_performance_with_cleansing.png'")

# ============================================
# 10. SAMPLE PREDICTIONS FROM TEST SET
# ============================================
print("\n" + "="*60)
print("STEP 8: SAMPLE PREDICTIONS")
print("="*60)

results = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_test_pred[:10],
    'Absolute_Error': np.abs(y_test.values[:10] - y_test_pred[:10])
}).round(3)

print("\nFirst 10 test set predictions:")
print(results.to_string(index=False))
print(f"\nMean Absolute Error on these samples: {results['Absolute_Error'].mean():.3f} g/L")

# ============================================
# 11. PREDICT NEW BATCH (Example)
# ============================================
print("\n" + "="*60)
print("STEP 9: PREDICT NEW BATCH")
print("="*60)

new_batch = pd.DataFrame({
    'Feed_Rate_gLh': [1.4],
    'Time_h': [30],
    'Temperature_C': [37.1],
    'pH': [6.9],
    'DO_percent': [72]
})

new_prediction = model.predict(new_batch)[0]

print(f"\nNew batch conditions (within normal ranges):")
print(f"  • Feed Rate: 1.4 g/L/hr")
print(f"  • Time: 30 hours")
print(f"  • Temperature: 37.1°C")
print(f"  • pH: 6.9")
print(f"  • DO: 72%")
print(f"\n✅ Predicted Protein Yield: {new_prediction:.3f} g/L")

# ============================================
# 12. SAVE CLEANED DATA AND MODEL
# ============================================
print("\n" + "="*60)
print("STEP 10: SAVE RESULTS")
print("="*60)

# Save cleaned dataset
df_clean.to_csv('bioreactor_data_cleaned.csv', index=False)
print("✓ Cleaned dataset saved as 'bioreactor_data_cleaned.csv'")

# Save model
import joblib
joblib.dump(model, 'bioreactor_yield_model.pkl')
print("✓ Model saved as 'bioreactor_yield_model.pkl'")

# Save test predictions
test_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Residual': y_test - y_test_pred
})
test_results.to_csv('test_predictions.csv', index=False)
print("✓ Test predictions saved as 'test_predictions.csv'")

print("\n" + "="*60)
print("✅ MODEL BUILDING WITH DATA CLEANSING COMPLETED")
print("="*60)

# Final summary
print(f"\n📋 FINAL SUMMARY:")
print(f"   • Initial data quality issues resolved")
print(f"   • Model trained on {len(df_clean)} clean samples")
print(f"   • Test R² Score: {test_r2:.3f}")
print(f"   • Test RMSE: {test_rmse:.3f} g/L")
print(f"   • Most important feature: {importance.iloc[0]['Feature']}")
print(f"   • Ready for predictions on new batches")