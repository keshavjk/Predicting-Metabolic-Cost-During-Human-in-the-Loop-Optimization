import numpy as np
import pandas as pd
import json, joblib, os
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def evaluate_lasso(X, y, k=10):
    """
    Evaluates Lasso regression with cross-validation to find the best alpha.
    Increased max_iter for better convergence on large datasets.
    """
    print("--- Evaluating Lasso Regression ---")
    lasso_cv = LassoCV(cv=k, random_state=0, n_alphas=100, max_iter=10000, n_jobs=-1)
    lasso_cv.fit(X, y)
    
    # Calculate the final MSE using the best alpha found by LassoCV
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    scores = cross_val_score(lasso_cv, X, y, cv=k, scoring=mse_scorer)
    
    print(f"Lasso finished with best alpha: {lasso_cv.alpha_:.5f}")
    return -np.mean(scores), lasso_cv.alpha_

def evaluate_nn(X, y, k=10, hidden_layer_sizes=[2,3,4,5]):
    """
    Evaluates a single-layer neural network.
    This version is memory-efficient: it trains models sequentially instead of all at once.
    It also uses the 'adam' solver, which is much better for large datasets.
    """
    print("\n--- Evaluating Neural Networks (Memory-Efficient) ---")
    results = {}
    best_mse = float('inf')
    best_cfg = None

    for hls in hidden_layer_sizes:
        print(f"    Training NN with {hls} hidden neurons...")
        
        # Use 'adam' solver and 'early_stopping' for efficiency on large datasets
        mlp = MLPRegressor(
            random_state=0, 
            max_iter=1000,          # Adam converges faster than lbfgs
            solver='adam', 
            hidden_layer_sizes=(hls,), 
            learning_rate_init=0.001,
            early_stopping=True,    # Prevents overfitting and saves time
            n_iter_no_change=10
        )
        
        # cross_val_score is more memory-efficient than GridSearchCV
        neg_mse_scores = cross_val_score(mlp, X, y, cv=k, scoring=make_scorer(mean_squared_error, greater_is_better=False), n_jobs=-1)
        
        current_mse = -np.mean(neg_mse_scores)
        results[hls] = current_mse
        print(f"    > MSE for {hls} neurons: {current_mse:.4f}")

        if current_mse < best_mse:
            best_mse = current_mse
            best_cfg = hls
            
    print(f"NN training finished. Best configuration: {best_cfg} neurons.")
    return best_mse, best_cfg, results

def evaluate_pca_then_nn(X, y, k=10, n_components=10, hidden_layer_sizes=[3]):
    """
    Evaluates a pipeline of PCA followed by a neural network.
    """
    print("\n--- Evaluating PCA + Neural Network ---")
    
    # Define the single NN configuration to use after PCA
    mlp = MLPRegressor(
        random_state=0, 
        max_iter=1000, 
        solver='adam', 
        hidden_layer_sizes=hidden_layer_sizes,
        early_stopping=True,
        n_iter_no_change=10
    )
    
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('mlp', mlp)
    ])
    
    neg_mse_scores = cross_val_score(pipeline, X, y, cv=k, scoring=make_scorer(mean_squared_error, greater_is_better=False), n_jobs=-1)
    
    print("PCA + NN finished.")
    return -np.mean(neg_mse_scores), None, None

def forward_stepwise_selection(X, y, n_features_to_select=10):
    """
    Performs forward stepwise feature selection using Lasso as the base model.
    Returns the names of the selected features.
    """
    print("\n--- Performing Forward Stepwise Feature Selection ---")
    
    # Use a simple Lasso model as the estimator for selection
    lasso = LassoCV(cv=5, max_iter=5000)
    
    sfs = SequentialFeatureSelector(
        lasso, 
        n_features_to_select=n_features_to_select, 
        direction='forward', 
        cv=5,
        n_jobs=-1
    )
    sfs.fit(X, y)
    
    print("Feature selection finished.")
    return list(X.columns[sfs.get_support()])

def plot_results(results, out_prefix='results'):
    """Generates a bar plot of the model MSEs."""
    models = ['Constant', 'Lasso', 'NN', 'PCA+NN']
    mses = [
        results['constant_prediction_mse'],
        results['lasso_mse'],
        results['nn_mse'],
        results['pca_mse']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, mses, color=['gray', 'skyblue', 'salmon', 'lightgreen'])
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Model Performance Comparison (Lower is Better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    plot_filename = f"{out_prefix.replace('.json', '')}_performance.png"
    plt.savefig(plot_filename)
    print(f"\nSaved performance plot to '{plot_filename}'")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='Input preprocessed CSV')
    p.add_argument('--out', default='results.json', help='Output JSON path')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    X = df.drop(columns=['metabolic_rate'])
    y = df['metabolic_rate']

    # --- Run Evaluations ---
    const_pred = np.mean((y - y.mean())**2)
    lasso_mse, best_alpha = evaluate_lasso(X, y, k=10)
    nn_mse, best_cfg, nn_results = evaluate_nn(X, y, k=10, hidden_layer_sizes=[2, 3])
    pca_mse, _, _ = evaluate_pca_then_nn(X, y, k=10, n_components=min(10, X.shape[1]), hidden_layer_sizes=[best_cfg] if best_cfg else [3])
    
    try:
        selected_feats = forward_stepwise_selection(X, y, n_features_to_select=min(10, X.shape[1]))
    except Exception as e:
        print(f"Could not perform feature selection. Error: {e}")
        selected_feats = []

    # --- Save Results ---
    out = {
        'constant_prediction_mse': float(const_pred),
        'lasso_mse': float(lasso_mse),
        'lasso_best_alpha': float(best_alpha),
        'nn_mse': float(nn_mse),
        'nn_best_hidden': int(best_cfg) if best_cfg is not None else None,
        'nn_results_by_hidden': {int(k): float(v) for k, v in nn_results.items()},
        'pca_mse': float(pca_mse),
        'selected_features_forward_stepwise': selected_feats
    }

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    
    print(f"\n✅ Success! All models evaluated. Results saved to '{args.out}'")
    
    # --- Plot Results ---
    plot_results(out, out_prefix=args.out)
