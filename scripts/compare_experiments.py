"""
Compare multiple DFL experiments
Utility script to analyze and compare different experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
from typing import Dict, List
import numpy as np


class ExperimentComparator:
    """
    Compare multiple DFL experiments and generate comparison reports
    """

    def __init__(self):
        self.experiments = {}
        self.metrics_dfs = {}
        self.global_dfs = {}

    def load_experiment(self, exp_name: str, metrics_file: str, global_metrics_file: str = None):
        """
        Load experiment data

        Args:
            exp_name: Name of experiment
            metrics_file: Path to node metrics CSV
            global_metrics_file: Path to global metrics CSV
        """
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        # Load node metrics
        metrics_df = pd.read_csv(metrics_file)
        self.metrics_dfs[exp_name] = metrics_df

        # Load global metrics if available
        if global_metrics_file and os.path.exists(global_metrics_file):
            global_df = pd.read_csv(global_metrics_file)
            self.global_dfs[exp_name] = global_df

        # Store experiment info
        self.experiments[exp_name] = {
            'total_rounds': metrics_df['round'].max(),
            'num_nodes': metrics_df['node_id'].nunique(),
            'total_samples': metrics_df.groupby('round')['data_size'].sum().iloc[0]
        }

        print(f"✅ Loaded experiment '{exp_name}': "
              f"{self.experiments[exp_name]['num_nodes']} nodes, "
              f"{self.experiments[exp_name]['total_rounds']} rounds")

    def compare_convergence(self, metric: str = 'test_accuracy', save_path: str = None):
        """
        Compare convergence curves across experiments

        Args:
            metric: Metric to compare
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))

        for exp_name, global_df in self.global_dfs.items():
            if f'global_{metric}' in global_df.columns:
                plt.plot(global_df['round'], global_df[f'global_{metric}'],
                        label=exp_name, linewidth=2, marker='o', markersize=4)

                # Add error bars if std is available
                if f'std_{metric}' in global_df.columns:
                    plt.fill_between(global_df['round'],
                                   global_df[f'global_{metric}'] - global_df[f'std_{metric}'],
                                   global_df[f'global_{metric}'] + global_df[f'std_{metric}'],
                                   alpha=0.2)

        plt.xlabel('Communication Round')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Convergence Comparison - {metric.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def compare_final_performance(self, save_path: str = None):
        """
        Compare final performance across experiments

        Args:
            save_path: Path to save plot
        """
        final_metrics = []

        for exp_name, metrics_df in self.metrics_dfs.items():
            final_round = metrics_df[metrics_df['round'] == metrics_df['round'].max()]
            final_metrics.append({
                'experiment': exp_name,
                'avg_test_accuracy': final_round['test_accuracy'].mean(),
                'std_test_accuracy': final_round['test_accuracy'].std(),
                'avg_test_loss': final_round['test_loss'].mean(),
                'std_test_loss': final_round['test_loss'].std(),
                'fairness_std': final_round['test_accuracy'].std(),  # Lower is more fair
                'num_nodes': final_round['node_id'].nunique()
            })

        comparison_df = pd.DataFrame(final_metrics)

        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Test Accuracy
        bars1 = ax1.bar(comparison_df['experiment'], comparison_df['avg_test_accuracy'],
                       yerr=comparison_df['std_test_accuracy'], capsize=5)
        ax1.set_title('Final Test Accuracy')
        ax1.set_ylabel('Test Accuracy')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars1, comparison_df['avg_test_accuracy']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # Test Loss
        bars2 = ax2.bar(comparison_df['experiment'], comparison_df['avg_test_loss'],
                       yerr=comparison_df['std_test_loss'], capsize=5, color='orange')
        ax2.set_title('Final Test Loss')
        ax2.set_ylabel('Test Loss')
        ax2.tick_params(axis='x', rotation=45)

        # Fairness (std of accuracy across nodes)
        bars3 = ax3.bar(comparison_df['experiment'], comparison_df['fairness_std'],
                       capsize=5, color='green')
        ax3.set_title('Fairness (Lower is Better)')
        ax3.set_ylabel('Std of Test Accuracy')
        ax3.tick_params(axis='x', rotation=45)

        # Summary table
        ax4.axis('tight')
        ax4.axis('off')
        table_data = comparison_df[['experiment', 'avg_test_accuracy', 'avg_test_loss', 'fairness_std']].round(4)
        table = ax4.table(cellText=table_data.values,
                         colLabels=['Experiment', 'Test Acc', 'Test Loss', 'Fairness'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to: {save_path}")
        else:
            plt.show()

        plt.close()

        return comparison_df

    def analyze_data_heterogeneity(self, save_path: str = None):
        """
        Analyze data heterogeneity impact across experiments

        Args:
            save_path: Path to save plot
        """
        heterogeneity_stats = []

        for exp_name, metrics_df in self.metrics_dfs.items():
            # Get data sizes for first round (they don't change)
            first_round = metrics_df[metrics_df['round'] == 1]
            data_sizes = first_round['data_size'].values

            # Calculate heterogeneity metrics
            mean_size = np.mean(data_sizes)
            std_size = np.std(data_sizes)
            cv = std_size / mean_size if mean_size > 0 else 0
            gini_coeff = self._calculate_gini(data_sizes)

            # Get final performance
            final_round = metrics_df[metrics_df['round'] == metrics_df['round'].max()]
            avg_accuracy = final_round['test_accuracy'].mean()

            heterogeneity_stats.append({
                'experiment': exp_name,
                'coefficient_variation': cv,
                'gini_coefficient': gini_coeff,
                'data_std': std_size,
                'final_accuracy': avg_accuracy,
                'min_data_size': data_sizes.min(),
                'max_data_size': data_sizes.max()
            })

        hetero_df = pd.DataFrame(heterogeneity_stats)

        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # CV vs Performance
        ax1.scatter(hetero_df['coefficient_variation'], hetero_df['final_accuracy'], s=100)
        ax1.set_xlabel('Coefficient of Variation')
        ax1.set_ylabel('Final Test Accuracy')
        ax1.set_title('Data Heterogeneity vs Performance')
        for i, exp in enumerate(hetero_df['experiment']):
            ax1.annotate(exp, (hetero_df['coefficient_variation'].iloc[i],
                              hetero_df['final_accuracy'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')

        # Gini coefficient comparison
        bars = ax2.bar(hetero_df['experiment'], hetero_df['gini_coefficient'])
        ax2.set_title('Gini Coefficient (Data Distribution)')
        ax2.set_ylabel('Gini Coefficient')
        ax2.tick_params(axis='x', rotation=45)

        # Data size range
        ax3.bar(hetero_df['experiment'], hetero_df['max_data_size'], label='Max')
        ax3.bar(hetero_df['experiment'], hetero_df['min_data_size'], label='Min')
        ax3.set_title('Data Size Range')
        ax3.set_ylabel('Number of Samples')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()

        # Summary statistics
        ax4.axis('tight')
        ax4.axis('off')
        summary_data = hetero_df[['experiment', 'coefficient_variation', 'gini_coefficient', 'final_accuracy']].round(4)
        table = ax4.table(cellText=summary_data.values,
                         colLabels=['Experiment', 'CV', 'Gini', 'Final Acc'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Heterogeneity Analysis')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heterogeneity analysis saved to: {save_path}")
        else:
            plt.show()

        plt.close()

        return hetero_df

    def _calculate_gini(self, values):
        """Calculate Gini coefficient for data distribution fairness"""
        values = np.array(values)
        n = len(values)
        if n == 0:
            return 0

        # Sort values
        sorted_values = np.sort(values)

        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    def generate_report(self, output_dir: str = "results/comparisons"):
        """
        Generate comprehensive comparison report

        Args:
            output_dir: Directory to save comparison results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Generating comparison report in {output_path}")

        # Generate all comparison plots
        self.compare_convergence('test_accuracy',
                                save_path=str(output_path / "convergence_test_accuracy.png"))
        self.compare_convergence('test_loss',
                                save_path=str(output_path / "convergence_test_loss.png"))

        performance_df = self.compare_final_performance(
            save_path=str(output_path / "final_performance.png"))

        hetero_df = self.analyze_data_heterogeneity(
            save_path=str(output_path / "data_heterogeneity.png"))

        # Save summary CSVs
        performance_df.to_csv(output_path / "performance_comparison.csv", index=False)
        hetero_df.to_csv(output_path / "heterogeneity_analysis.csv", index=False)

        print(f"✅ Comparison report generated in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare DFL Experiments")
    parser.add_argument("--experiments", "-e", nargs="+", required=True,
                       help="List of experiment names")
    parser.add_argument("--results-dir", "-r", default="results",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", "-o", default="results/comparisons",
                       help="Output directory for comparison results")

    args = parser.parse_args()

    # Initialize comparator
    comparator = ExperimentComparator()

    # Load experiments
    for exp_name in args.experiments:
        metrics_file = os.path.join(args.results_dir, f"{exp_name}_metrics.csv")
        global_file = os.path.join(args.results_dir, f"{exp_name}_global_metrics.csv")

        try:
            comparator.load_experiment(exp_name, metrics_file, global_file)
        except FileNotFoundError as e:
            print(f"⚠️  Warning: {e}")
            continue

    if not comparator.experiments:
        print("❌ No experiments loaded successfully")
        return 1

    # Generate comparison report
    comparator.generate_report(args.output_dir)

    return 0


if __name__ == "__main__":
    exit(main())