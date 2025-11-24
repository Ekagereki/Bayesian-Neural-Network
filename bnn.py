import os
import random
import numpy as np
from matplotlib import pyplot as plt
from eda import import_data
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score,
                             precision_recall_curve, average_precision_score, classification_report)
from collections import Counter
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import f_classif, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')



def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors



class BayesianDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, kl_weight=1.0, prior_scale=1.0, **kwargs):
        super(BayesianDense, self).__init__(**kwargs)
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.kl_weight = float(kl_weight)
        self.prior_scale = float(prior_scale)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        initializer = tf.keras.initializers.GlorotNormal(seed=42)

        self.kernel_loc = self.add_weight(
            name='kernel_loc',
            shape=(input_dim, self.units),
            initializer=initializer,
            trainable=True)

        self.kernel_scale = tfp.util.TransformedVariable(
            initial_value=tf.ones((input_dim, self.units)) * 0.01,
            bijector=tfb.Softplus(),
            name='kernel_scale',
            trainable=True)

        self.bias_loc = self.add_weight(
            name='bias_loc',
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)

        self.bias_scale = tfp.util.TransformedVariable(
            initial_value=tf.ones((self.units,)) * 0.01,
            bijector=tfb.Softplus(),
            name='bias_scale',
            trainable=True)

        self.kernel_prior = tfd.Normal(loc=0., scale=self.prior_scale)
        self.bias_prior = tfd.Normal(loc=0., scale=self.prior_scale)

        super(BayesianDense, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        eps_k = tf.random.normal(shape=tf.shape(self.kernel_loc))
        kernel = self.kernel_loc + self.kernel_scale * eps_k

        eps_b = tf.random.normal(shape=tf.shape(self.bias_loc))
        bias = self.bias_loc + self.bias_scale * eps_b

        kernel_kl = tfd.kl_divergence(
            tfd.Normal(loc=self.kernel_loc, scale=self.kernel_scale),
            self.kernel_prior
        )
        bias_kl = tfd.kl_divergence(
            tfd.Normal(loc=self.bias_loc, scale=self.bias_scale),
            self.bias_prior
        )
        self.add_loss(self.kl_weight * (tf.reduce_sum(kernel_kl) + tf.reduce_sum(bias_kl)))

        output = tf.matmul(inputs, kernel) + bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(BayesianDense, self).get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation) if self.activation else None,
            "kl_weight": self.kl_weight,
            "prior_scale": self.prior_scale,
        })
        return config


def preprocess_data(df):
    categorical_cols = ['Form', 'Gender', 'Financial_Status', 'Home', 'Religion',
                        'Fathers_Education', 'Sports', 'Mothers_Education',
                        'Co_Curricular', 'Percieved_Academic_Abilities']

    label_encoder = LabelEncoder()
    df['Depression_Status'] = label_encoder.fit_transform(df['Depression_Status'])

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df['Siblings'] = df['Siblings'].replace('> 4', '5').astype(int)

    X = df.drop(['Depression_Status', 'PHQ_Total'], axis=1)
    y = df['Depression_Status']

    numerical_cols = ['Age', 'Num_parents_dead', 'Siblings', 'GAD_Total', 'MSSS_Total',
                      'EPOCH_Total', 'PCS_Academic_Total', 'UCLA_Total', 'Gratitude_Total']

    for col in numerical_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = np.clip(X[col], lower_bound, upper_bound)

    scaler = RobustScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y


def handle_class_imbalance(X, y, strategy='smote_tomek'):
    if strategy == 'smote_tomek':
        smote_tomek = SMOTETomek(random_state=42, sampling_strategy='auto')
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    elif strategy == 'adasyn':
        adasyn = ADASYN(random_state=42, sampling_strategy='auto')
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
    elif strategy == 'smote':
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif strategy == 'combined':
        over = SMOTE(sampling_strategy=0.8, random_state=42)
        under = TomekLinks()
        pipeline = ImbPipeline(steps=[('o', over), ('u', under)])
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    print(f"Class distribution after {strategy}: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def advanced_feature_selection(X, y, n_features=12):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    gb_importance = pd.Series(gb.feature_importances_, index=X.columns)

    svm = LinearSVC(C=0.01, penalty='l1', dual=False, random_state=42)
    svm.fit(X, y)
    svm_importance = pd.Series(np.abs(svm.coef_[0]), index=X.columns)

    mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
    mi_selector.fit(X, y)
    mi_scores = pd.Series(mi_selector.scores_, index=X.columns)

    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X, y)
    f_scores = pd.Series(f_selector.scores_, index=X.columns)

    combined_scores = (
            rf_importance.rank() * 0.3 +
            gb_importance.rank() * 0.2 +
            svm_importance.rank() * 0.2 +
            mi_scores.rank() * 0.15 +
            f_scores.rank() * 0.15
    )

    selected_features = combined_scores.nlargest(n_features).index.tolist()
    print("Top features by combined importance:")
    for feature in selected_features:
        print(f"  {feature}: RF={rf_importance[feature]:.4f}, GB={gb_importance[feature]:.4f}, "
              f"SVM={svm_importance[feature]:.4f}, MI={mi_scores[feature]:.4f}, F={f_scores[feature]:.4f}")
    return selected_features


def check_multicollinearity(X, selected_features, vif_threshold=5.0):
    if len(selected_features) < 2:
        return selected_features
    try:
        X_selected = X[selected_features].copy()
        X_with_const = add_constant(X_selected)

        vif_scores = {}
        for i, feature in enumerate(selected_features, 1):
            try:
                vif = variance_inflation_factor(X_with_const.values, i)
                vif_scores[feature] = vif
            except:
                vif_scores[feature] = float('inf')

        final_features = [f for f in selected_features if vif_scores.get(f, float('inf')) <= vif_threshold]
        print("VIF scores:")
        for feature in selected_features:
            print(f"  {feature}: VIF={vif_scores.get(feature, float('inf')):.2f}")
        return final_features
    except Exception as e:
        print(f"Error in VIF calculation: {e}")
        return selected_features

def get_bnn_selected_features(X_train, y_train, n_features=15):
    """Get the selected features used by BNN"""
    selected_features = advanced_feature_selection(X_train, y_train, n_features)
    selected_features = check_multicollinearity(X_train, selected_features, vif_threshold=5.0)
    return selected_features

def create_optimized_bayesian_model(input_dim, class_weights=None):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    kl_weight = 1.0 / (input_dim * 100)

    x = BayesianDense(32, activation='relu', kl_weight=kl_weight, prior_scale=0.1)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = BayesianDense(64, activation='relu', kl_weight=kl_weight, prior_scale=0.1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = BayesianDense(32, activation='relu', kl_weight=kl_weight, prior_scale=0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = BayesianDense(1, activation='sigmoid', kl_weight=kl_weight, prior_scale=0.1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if class_weights is not None:
        def weighted_binary_crossentropy(y_true, y_pred):
            weights = class_weights[1] * y_true + class_weights[0] * (1 - y_true)
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            return tf.reduce_mean(weights * bce)
        loss_fn = weighted_binary_crossentropy
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )
    return model


def train_model_with_optimization(model, X_train, y_train, X_val, y_val, patience=15):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_pr_auc', patience=patience, restore_best_weights=True, mode='max', verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_pr_auc', save_best_only=True, mode='max', verbose=1)

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=100, batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    return history


def mc_dropout_predictions(model, X, n_samples=100):
    preds = []
    for _ in range(n_samples):
        y_hat = model(X, training=True).numpy().flatten()
        preds.append(y_hat)
    preds = np.array(preds)
    return preds, preds.mean(axis=0), preds.std(axis=0)


def plot_uncertainty_distribution(y_true, mean_preds, std_preds):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(mean_preds, std_preds, c=y_true, cmap="coolwarm", alpha=0.6)
    plt.colorbar(scatter, label="True Label")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Predictive Std (Uncertainty)")
    plt.title("Uncertainty vs Predicted Probability")
    plt.show()


def evaluate_thresholds(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.plot(thresholds, f1_scores[:-1], label="F1-score")
    plt.axvline(0.5, color="gray", linestyle="--", label="Default=0.5")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall-F1 vs Threshold")
    plt.legend()
    plt.show()

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold = {best_threshold:.2f} "
          f"(Precision={precisions[best_idx]:.3f}, Recall={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f})")
    return best_threshold

def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng(42)
    stats = []

    # Ensure we're working with numpy arrays
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))

    stats = np.array(stats)
    lower = np.percentile(stats, (1 - alpha) / 2 * 100)
    upper = np.percentile(stats, (1 + alpha) / 2 * 100)
    return np.mean(stats), lower, upper

def comprehensive_evaluation(y_true, y_probs, threshold=0.5, n_bootstrap=1000):
    # Convert y_true to numpy array to avoid pandas indexing issues
    y_true_np = np.array(y_true)
    y_pred = (y_probs >= threshold).astype(int)

    acc, acc_l, acc_u = bootstrap_ci(lambda yt, yp: accuracy_score(yt, yp), y_true_np, y_pred, n_bootstrap)
    f1, f1_l, f1_u = bootstrap_ci(lambda yt, yp: f1_score(yt, yp), y_true_np, y_pred, n_bootstrap)
    prec, prec_l, prec_u = bootstrap_ci(lambda yt, yp: precision_score(yt, yp), y_true_np, y_pred, n_bootstrap)
    rec, rec_l, rec_u = bootstrap_ci(lambda yt, yp: recall_score(yt, yp), y_true_np, y_pred, n_bootstrap)

    auc, auc_l, auc_u = bootstrap_ci(lambda yt, yp: roc_auc_score(yt, yp), y_true_np, y_probs, n_bootstrap)
    prauc, prauc_l, prauc_u = bootstrap_ci(lambda yt, yp: average_precision_score(yt, yp), y_true_np, y_probs,
                                               n_bootstrap)

    cm = confusion_matrix(y_true_np, y_pred)
    report = classification_report(y_true_np, y_pred, digits=4)


    print("="*60, " COMPREHENSIVE MODEL EVALUATION ", "="*60)
    print(f"Accuracy: {acc:.4f} (95% CI: {acc_l:.4f}-{acc_u:.4f})")
    print(f"F1 Score: {f1:.4f} (95% CI: {f1_l:.4f}-{f1_u:.4f})")
    print(f"Precision: {prec:.4f} (95% CI: {prec_l:.4f}-{prec_u:.4f})")
    print(f"Recall: {rec:.4f} (95% CI: {rec_l:.4f}-{rec_u:.4f})")
    print(f"ROC AUC: {auc:.4f} (95% CI: {auc_l:.4f}-{auc_u:.4f})")
    print(f"PR AUC: {prauc:.4f} (95% CI: {prauc_l:.4f}-{prauc_u:.4f})")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("="*150)

    return {
        "accuracy": (acc, acc_l, acc_u),
        "f1": (f1, f1_l, f1_u),
        "precision": (prec, prec_l, prec_u),
        "recall": (rec, rec_l, rec_u),
        "roc_auc": (auc, auc_l, auc_u),
        "pr_auc": (prauc, prauc_l, prauc_u),
        "confusion_matrix": cm,
        "report": report
    }

def plot_calibration_curve(y_true, y_probs, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "s-", label="Calibration curve")
    plt.plot([0,1], [0,1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

'''
def compute_shap_feature_influence(model, X_background, X_explain, feature_names):
    Xb = np.array(X_background, dtype=np.float32)
    Xe = np.array(X_explain, dtype=np.float32)
    explainer = shap.DeepExplainer(model, Xb)
    shap_values = explainer.shap_values(Xe, check_additivity=False)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(sv, Xe, feature_names=feature_names)
    return sv
'''

def cross_validate_model(X, y, selected_features, class_weights, n_splits=10, n_epochs=30, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0
    results = []

    for train_index, val_index in kf.split(X):
        fold += 1
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_tr, X_val = X.iloc[train_index][selected_features], X.iloc[val_index][selected_features]
        y_tr, y_v = y.iloc[train_index], y.iloc[val_index]

        X_tr_t = tf.convert_to_tensor(X_tr.values, dtype=tf.float32)
        y_tr_t = tf.convert_to_tensor(y_tr.values, dtype=tf.float32)
        X_val_t = tf.convert_to_tensor(X_val.values, dtype=tf.float32)
        y_val_t = tf.convert_to_tensor(y_v.values, dtype=tf.float32)

        model_cv = create_optimized_bayesian_model(len(selected_features), class_weights)
        model_cv.fit(X_tr_t, y_tr_t, validation_data=(X_val_t, y_val_t),
                     epochs=n_epochs, batch_size=batch_size, verbose=0)

        y_val_prob = model_cv.predict(X_val.values).flatten()
        metric_dict = comprehensive_evaluation(y_v.to_numpy(), y_val_prob, threshold=0.5, n_bootstrap=200)
        results.append(metric_dict)

    return results

def main():
    set_seeds(42)
    df = import_data()
    X, y = preprocess_data(df)

    print("Original class distribution:")
    print(Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train, strategy='smote_tomek')

    print("\nPerforming advanced feature selection...")
    selected_features = advanced_feature_selection(X_train_balanced, y_train_balanced, n_features=15)
    selected_features = check_multicollinearity(X_train_balanced, selected_features, vif_threshold=5.0)
    print(f"\nFinal selected features ({len(selected_features)}): {selected_features}")

    class_counts = Counter(y_train_balanced)
    total_samples = len(y_train_balanced)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_balanced[selected_features], y_train_balanced,
        test_size=0.15, random_state=42, stratify=y_train_balanced)

    X_train_tensor = tf.convert_to_tensor(X_train_final.values, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train_final.values, dtype=tf.float32)
    X_val_tensor = tf.convert_to_tensor(X_val.values, dtype=tf.float32)
    y_val_tensor = tf.convert_to_tensor(y_val.values, dtype=tf.float32)

    print("\nTraining optimized Bayesian neural network...")
    model = create_optimized_bayesian_model(len(selected_features), class_weights)
    history = train_model_with_optimization(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

    # Uncertainty analysis
    print("Running uncertainty analysis")
    X_test_tensor = tf.convert_to_tensor(X_test[selected_features].values, dtype=tf.float32)
    preds_samples, mean_preds, std_preds = mc_dropout_predictions(model, X_test_tensor, n_samples=100)
    plot_uncertainty_distribution(y_test, mean_preds, std_preds)

    # Threshold tuning
    print("Evaluating thresholds")
    best_threshold = evaluate_thresholds(y_test, mean_preds)

    results = comprehensive_evaluation(y_test, mean_preds, threshold=best_threshold, n_bootstrap=1000)

    # Calibration
    plot_calibration_curve(y_test, mean_preds)

    # SHAP Feature Influence
    # X_background = X_train_final.sample(n=min(100, len(X_train_final)), random_state=42)
    # compute_shap_feature_influence(model, X_background, X_test[selected_features], feature_names=selected_features)

    # Cross-validation
    cv_results = cross_validate_model(X, y, selected_features, class_weights, n_splits=10, n_epochs=30)

    return model, selected_features, best_threshold


if __name__ == "__main__":
    main()
