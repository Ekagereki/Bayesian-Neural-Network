import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, classification_report,
                             precision_recall_curve, average_precision_score, roc_curve, auc,
                             balanced_accuracy_score, make_scorer)
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from bnn import import_data, preprocess_data, get_bnn_selected_features, set_seeds


def train_optimized_random_forest():

    df = import_data()
    X, y = preprocess_data(df)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Get features selected by BNN
    selected_features = get_bnn_selected_features(X_train, y_train, n_features=12)

    print(f"Selected features for Random Forest ({len(selected_features)}):")
    for feature in selected_features:
        print(f"  - {feature}")

    # Prepare data with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Create a pipeline with SMOTETomek
    pipeline = ImbPipeline([
        ('sampling', SMOTETomek(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter tuning with focus on Class 0
    best_rf = optimize_hyperparameters(X_train_selected, y_train, pipeline)

    # Train the optimized model
    best_rf.fit(X_train_selected, y_train)

    # Predictions
    y_pred = best_rf.predict(X_test_selected)
    y_pred_proba = best_rf.predict_proba(X_test_selected)[:, 1]

    # Comprehensive evaluation
    metrics = evaluate_random_forest(best_rf, X_test_selected, y_test, y_pred, y_pred_proba)

    # Feature importance analysis
    plot_feature_importance(best_rf, selected_features)

    # Cross-validation
    perform_cross_validation(X[selected_features], y, best_rf)

    # Threshold optimization (Class 0 focus)
    optimal_threshold = find_optimal_threshold_class0(best_rf, X_test_selected, y_test)
    print(f"\nOptimal threshold for Class 0 F1: {optimal_threshold:.3f}")

    # Evaluate with Class 0 optimal threshold
    y_pred_optimal = (best_rf.predict_proba(X_test_selected)[:, 1] >= optimal_threshold).astype(int)
    print("\nPerformance with Class 0 optimal threshold:")
    evaluate_threshold_performance(y_test, y_pred_optimal, best_rf.predict_proba(X_test_selected)[:, 1])

    return best_rf, selected_features, metrics


def optimize_hyperparameters(X_train, y_train, pipeline):
    """Optimize Random Forest hyperparameters with Class 0 scoring"""

    param_dist = {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': [None, 10, 20, 30, 40, 50],
        'classifier__min_samples_split': randint(2, 20),
        'classifier__min_samples_leaf': randint(1, 10),
        'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7, None],
        'classifier__bootstrap': [True, False],
        'classifier__class_weight': [
            None,
            'balanced',
            'balanced_subsample',
            {0: 2, 1: 1},
            {0: 3, 1: 1}
        ]
    }

    # Focus scoring on Class 0 F1
    f1_class0 = make_scorer(f1_score, pos_label=0)

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=f1_class0,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("Performing hyperparameter optimization (Class 0 focus)...")
    random_search.fit(X_train, y_train)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best Class 0 F1 score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_


def evaluate_random_forest(model, X_test, y_test, y_pred, y_pred_proba):
    """Comprehensive evaluation"""

    print("\n" + "=" * 60)
    print("OPTIMIZED RANDOM FOREST EVALUATION RESULTS")
    print("=" * 60)

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'specificity': calculate_specificity(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'g_mean': geometric_mean_score(y_test, y_pred)
    }

    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric.upper():<20}: {value:.4f}")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    print("\nImbalanced Classification Report:")
    print(classification_report_imbalanced(y_test, y_pred))

    # ROC & PR curves
    plot_evaluation_curves(y_test, y_pred_proba)

    # Class-wise performance
    analyze_class_performance(y_test, y_pred)

    return metrics


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_evaluation_curves(y_test, y_pred_proba):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR (AP = {avg_precision:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_class_performance(y_test, y_pred):
    print("\nClass-wise Performance Analysis:")
    print("-" * 40)

    class_counts = np.bincount(y_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    for i, count in enumerate(class_counts):
        print(f"\nClass {i}: {count} samples ({count / len(y_test) * 100:.1f}%)")
        print(f"  Precision: {report[str(i)]['precision']:.3f}")
        print(f"  Recall:    {report[str(i)]['recall']:.3f}")
        print(f"  F1-score:  {report[str(i)]['f1-score']:.3f}")


def find_optimal_threshold_class0(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold, best_f1 = 0.5, 0

    for thr in thresholds:
        y_pred = (y_pred_proba >= thr).astype(int)
        f1_class0 = f1_score(y_test, y_pred, pos_label=0)
        if f1_class0 > best_f1:
            best_f1, best_threshold = f1_class0, thr

    return best_threshold


def evaluate_threshold_performance(y_test, y_pred, y_pred_proba):
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def plot_feature_importance(model, feature_names):
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        importance = model.named_steps['classifier'].feature_importances_
    else:
        importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Random Forest Feature Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nDetailed Feature Importance:")
    for _, row in feature_importance_df.iterrows():
        print(f"{row['feature']:.<30} {row['importance']:.4f}")


def perform_cross_validation(X, y, model, n_splits=10):
    print(f"\nPerforming {n_splits}-fold Stratified Cross Validation...")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring_metrics = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'f1_macro': 'f1_macro',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision'
    }

    cv_results = {}
    for metric_name, metric_scorer in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric_scorer, n_jobs=-1)
        cv_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{metric_name}: {scores.mean():.4f} (±{scores.std():.4f})")

    return cv_results


if __name__ == "__main__":
    set_seeds()
    rf_model, selected_features, metrics = train_optimized_random_forest()
    print("\nOptimized Random Forest training completed successfully!")

    # --- Extract Best Parameters ---
    if hasattr(rf_model, "named_steps"):
        best_params = rf_model.named_steps['classifier'].get_params()
    else:
        best_params = rf_model.get_params()

    print("\nBest Random Forest Parameters:")
    print(f"Number of Trees (n_estimators): {best_params['n_estimators']}")
    print(f"Max Depth: {best_params['max_depth']}")
    print(f"Min Samples Split: {best_params['min_samples_split']}")
    print(f"Min Samples Leaf: {best_params['min_samples_leaf']}")
    print(f"Max Features: {best_params['max_features']}")
    print(f"Class Weight: {best_params['class_weight']}")

    # --- Extract Performance Metrics ---
    print("\nPerformance Metrics on Test Set:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # --- Extract Top 5 Features ---
    if hasattr(rf_model, 'named_steps') and 'classifier' in rf_model.named_steps:
        importances = rf_model.named_steps['classifier'].feature_importances_
    else:
        importances = rf_model.feature_importances_

    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance.head(5)
    print("\nTop 5 Important Features:")
    print(top_features.to_string(index=False))

