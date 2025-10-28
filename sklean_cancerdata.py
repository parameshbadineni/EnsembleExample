"""
Complete Ensemble Learning Examples in Python
Shows all major ensemble techniques with a real dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Individual Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Ensemble Methods
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier
)

# Load dataset
print("=" * 60)
print("ENSEMBLE LEARNING EXAMPLES")
print("=" * 60)

data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nDataset: {data.filename}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")

# ============================================================================
# 1. INDIVIDUAL MODELS (Baseline)
# ============================================================================
print("\n" + "=" * 60)
print("1. INDIVIDUAL MODELS (Baseline)")
print("=" * 60)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Naive Bayes': GaussianNB()
}

baseline_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    baseline_scores[name] = score
    print(f"{name:25} Accuracy: {score:.4f}")

# ============================================================================
# 2. BAGGING - Random Forest
# ============================================================================
print("\n" + "=" * 60)
print("2. BAGGING - Random Forest")
print("=" * 60)
print("Trains multiple decision trees on random data samples")

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_score:.4f}")
print(f"Improvement over best single model: {rf_score - max(baseline_scores.values()):.4f}")

# ============================================================================
# 3. BAGGING - Bagging Classifier
# ============================================================================
print("\n" + "=" * 60)
print("3. BAGGING - Bagging Classifier")
print("=" * 60)
print("Can use any base model with bagging")

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)
bagging_score = bagging.score(X_test, y_test)
print(f"Bagging Classifier Accuracy: {bagging_score:.4f}")

# ============================================================================
# 4. BOOSTING - AdaBoost
# ============================================================================
print("\n" + "=" * 60)
print("4. BOOSTING - AdaBoost")
print("=" * 60)
print("Sequentially trains models, focusing on misclassified examples")

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
ada_score = ada.score(X_test, y_test)
print(f"AdaBoost Accuracy: {ada_score:.4f}")

# ============================================================================
# 5. BOOSTING - Gradient Boosting
# ============================================================================
print("\n" + "=" * 60)
print("5. BOOSTING - Gradient Boosting")
print("=" * 60)
print("Builds trees to correct residual errors")

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)
print(f"Gradient Boosting Accuracy: {gb_score:.4f}")

# ============================================================================
# 6. VOTING CLASSIFIER - Hard Voting
# ============================================================================
print("\n" + "=" * 60)
print("6. VOTING CLASSIFIER - Hard Voting")
print("=" * 60)
print("Majority vote from multiple models")

voting_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(random_state=42))
    ],
    voting='hard'  # Majority vote
)
voting_hard.fit(X_train, y_train)
voting_hard_score = voting_hard.score(X_test, y_test)
print(f"Hard Voting Accuracy: {voting_hard_score:.4f}")

# ============================================================================
# 7. VOTING CLASSIFIER - Soft Voting
# ============================================================================
print("\n" + "=" * 60)
print("7. VOTING CLASSIFIER - Soft Voting")
print("=" * 60)
print("Averages predicted probabilities")

voting_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='soft'  # Average probabilities
)
voting_soft.fit(X_train, y_train)
voting_soft_score = voting_soft.score(X_test, y_test)
print(f"Soft Voting Accuracy: {voting_soft_score:.4f}")

# ============================================================================
# 8. STACKING
# ============================================================================
print("\n" + "=" * 60)
print("8. STACKING")
print("=" * 60)
print("Uses meta-model to learn how to combine base models")

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)
stacking_score = stacking.score(X_test, y_test)
print(f"Stacking Accuracy: {stacking_score:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS COMPARISON")
print("=" * 60)

results = {
    'Best Single Model': max(baseline_scores.values()),
    'Random Forest (Bagging)': rf_score,
    'Bagging Classifier': bagging_score,
    'AdaBoost': ada_score,
    'Gradient Boosting': gb_score,
    'Hard Voting': voting_hard_score,
    'Soft Voting': voting_soft_score,
    'Stacking': stacking_score
}

# Sort by score
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nRanked by Performance:")
print("-" * 60)
for i, (method, score) in enumerate(sorted_results, 1):
    print(f"{i}. {method:30} {score:.4f}")

print("\n" + "=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)
print("✓ Ensemble methods typically outperform single models")
print("✓ Boosting often works best for structured data")
print("✓ Voting combines diverse models for robustness")
print("✓ Stacking can learn optimal model combinations")

# ============================================================================
# BONUS: Making Predictions with Best Model
# ============================================================================
print("\n" + "=" * 60)
print("BONUS: Making Predictions")
print("=" * 60)

best_model_name = sorted_results[0][0]
print(f"\nUsing best model: {best_model_name}")

# Get the best model
if best_model_name == 'Best Single Model':
    best_model = max(models.items(), key=lambda x: baseline_scores[x[0]])[1]
else:
    best_model = eval(best_model_name.split()[0].lower().replace(' ', '_'))

# Make predictions
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("\n✅ Complete! All ensemble methods demonstrated.")
