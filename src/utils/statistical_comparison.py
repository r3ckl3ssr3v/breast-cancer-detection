import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def perform_statistical_tests(results_dir, output_dir=None):
    """
    Perform statistical tests to compare model performance.
    
    Args:
        results_dir: Directory containing model results
        output_dir: Directory to save statistical test results
    """
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model comparison data
    comparison_df = pd.read_csv(os.path.join(results_dir, "model_comparison.csv"))
    
    # Extract model names and metrics
    models = comparison_df['Model'].tolist()
    metrics = comparison_df.columns.tolist()[1:]  # Skip 'Model' column
    
    # Perform t-tests for each pair of models for each metric
    t_test_results = {}
    
    for metric in metrics:
        t_test_results[metric] = pd.DataFrame(index=models, columns=models)
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    t_test_results[metric].loc[model1, model2] = 1.0
                    continue
                
                # For a real statistical test, we would need multiple runs or cross-validation results
                # Here we're just comparing the single values (which isn't statistically valid)
                # In a real scenario, you would load multiple run results for each model
                
                val1 = comparison_df.loc[comparison_df['Model'] == model1, metric].values[0]
                val2 = comparison_df.loc[comparison_df['Model'] == model2, metric].values[0]
                
                # Since we don't have multiple samples, we'll just compute the ratio
                # In a real scenario with multiple runs, you would use t-test:
                # t_stat, p_value = stats.ttest_ind(model1_values, model2_values)
                
                ratio = val1 / val2 if val2 != 0 else float('inf')
                t_test_results[metric].loc[model1, model2] = ratio
    
    # Save t-test results
    for metric, result_df in t_test_results.items():
        result_df.to_csv(os.path.join(output_dir, f"{metric}_comparison.csv"))
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(result_df, annot=True, cmap="YlGnBu", vmin=0.5, vmax=1.5)
        plt.title(f"Model Comparison - {metric.capitalize().replace('_', ' ')} Ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison_heatmap.png"))
        plt.close()
    
    # Perform ANOVA test across all models for each metric
    anova_results = {}
    
    for metric in metrics:
        # In a real scenario with multiple runs, you would use:
        # f_stat, p_value = stats.f_oneway(model1_values, model2_values, ...)
        
        # Since we don't have multiple samples, we'll just compute the max/min ratio
        values = comparison_df[metric].values
        max_min_ratio = np.max(values) / np.min(values) if np.min(values) != 0 else float('inf')
        
        anova_results[metric] = {
            'max_value': np.max(values),
            'min_value': np.min(values),
            'max_model': comparison_df.loc[comparison_df[metric] == np.max(values), 'Model'].values[0],
            'min_model': comparison_df.loc[comparison_df[metric] == np.min(values), 'Model'].values[0],
            'max_min_ratio': max_min_ratio
        }
    
    # Save ANOVA results
    anova_df = pd.DataFrame(anova_results).T
    anova_df.to_csv(os.path.join(output_dir, "anova_results.csv"))
    
    # Create summary visualization
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        # Get values for each model
        values = [comparison_df.loc[comparison_df['Model'] == model, metric].values[0] for model in models]
        
        # Create bar plot
        bars = plt.bar(models, values)
        
        # Highlight best model
        best_idx = np.argmax(values)
        bars[best_idx].set_color('green')
        
        plt.title(f"{metric.capitalize().replace('_', ' ')}")
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
    plt.close()
    
    return t_test_results, anova_results

def cross_dataset_comparison(results_dirs, dataset_names, output_dir):
    """
    Compare model performance across different datasets.
    
    Args:
        results_dirs: List of directories containing results for each dataset
        dataset_names: Names of the datasets
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load comparison data for each dataset
    all_comparisons = []
    
    for i, results_dir in enumerate(results_dirs):
        comparison_path = os.path.join(results_dir, "model_comparison.csv")
        if not os.path.exists(comparison_path):
            print(f"Warning: Comparison file not found for dataset {dataset_names[i]}")
            continue
        
        comparison_df = pd.read_csv(comparison_path)
        comparison_df['Dataset'] = dataset_names[i]
        all_comparisons.append(comparison_df)
    
    if not all_comparisons:
        print("Error: No comparison data found")
        return
    
    # Combine all comparisons
    combined_df = pd.concat(all_comparisons, ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, "cross_dataset_comparison.csv"), index=False)
    
    # Extract metrics
    metrics = combined_df.columns.tolist()[1:-1]  # Skip 'Model' and 'Dataset' columns
    
    # Create visualizations for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        sns.barplot(x='Model', y=metric, hue='Dataset', data=combined_df)
        
        plt.title(f"{metric.capitalize().replace('_', ' ')} Across Datasets")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_cross_dataset.png"))
        plt.close()
    
    # Create heatmap for each model showing performance across datasets
    models = combined_df['Model'].unique()
    
    for model in models:
        model_df = combined_df[combined_df['Model'] == model]
        
        # Create pivot table
        pivot_df = model_df.pivot(index='Dataset', columns=None, values=metrics)
        
        # Create heatmap
        plt.figure(figsize=(12, len(dataset_names) * 0.8))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".4f")
        plt.title(f"{model} Performance Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_cross_dataset_heatmap.png"))
        plt.close()
    
    return combined_df