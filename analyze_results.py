import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load results
def load_results(path='results/definitions_results.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    results = load_results()
    df = pd.DataFrame(results)

    # Aggregate mean scores by model and prompt
    metrics = ['bleu', 'rougeL', 'cosine']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=df,
            x='model_name',
            y=metric,
            hue='prompt_version',
            ci='sd'
        )
        plt.title(f'Average {metric.upper()} by Model and Prompt')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f'results/{metric}_by_model_prompt.png')
        plt.close()

    # Boxplot: Distribution of metrics for each model
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.boxplot(
            data=df,
            x='model_name',
            y=metric,
            hue='prompt_version'
        )
        plt.title(f'{metric.upper()} Distribution by Model and Prompt')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f'results/{metric}_boxplot.png')
        plt.close()

    # Heatmap: Model vs Prompt (mean metric)
    for metric in metrics:
        pivot = df.pivot_table(index='model_name', columns='prompt_version', values=metric, aggfunc='mean')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
        plt.title(f'{metric.upper()} Heatmap (Model vs Prompt)')
        plt.tight_layout()
        plt.savefig(f'results/{metric}_heatmap.png')
        plt.close()

    print('Analysis complete. Plots saved in results/.')

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main()
