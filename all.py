# ======================================================================
# Advanced LCC & Social Impact Optimization Simulation
# (Complete Suite: Perspectives 1-6 + 5.5 Correlation + ZIP Export)
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numpy.random import Generator, PCG64
from scipy import stats
import warnings
import os
import zipfile
import shutil

# Google Colab specific import
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Warning: Not running in Google Colab. Files will be saved locally.")

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Visualization Style Setup
# ----------------------------------------------------------------------
def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.figsize': (10, 6),
        'figure.dpi': 120,
        'savefig.dpi': 300, # Publication quality
        'savefig.bbox': 'tight'
    })

# 2. Simulator Class
# ----------------------------------------------------------------------
class LCCSimulator:
    def __init__(self, seed=42):
        self.rng = Generator(PCG64(seed))
        self.states = ['A', 'B', 'C', 'D']

        self.default_costs = {
            'inspection': 1,
            'repair_B': 15,
            'repair_C': 60,
            'replace_D': 250,
            'social_loss_D': 500
        }

    def _get_next_state(self, current_state_idx, probs):
        if self.states[current_state_idx] == 'D':
            return 3

        current_s = self.states[current_state_idx]
        next_s = self.states[current_state_idx + 1]
        key = f"{current_s}->{next_s}"
        p = probs.get(key, 0)

        if self.rng.random() < p:
            return current_state_idx + 1
        return current_state_idx

    def run_cohort_simulation(self, strategy, transition_probs, cost_params=None,
                              duration_years=50, num_units=1000):
        if cost_params is None:
            cost_params = self.default_costs

        months = duration_years * 12

        unit_direct_costs = np.zeros(num_units)
        unit_social_costs = np.zeros(num_units)
        unit_failures = np.zeros(num_units)

        for i in range(num_units):
            curr_state = 0
            d_cost = 0
            s_cost = 0
            fails = 0

            for m in range(months):
                d_cost += cost_params['inspection']
                action_taken = False

                if strategy == 'preventive_B':
                    if curr_state >= 1:
                        if curr_state == 1: cost = cost_params['repair_B']
                        elif curr_state == 2: cost = cost_params['repair_C']
                        else: cost = cost_params['replace_D'] + cost_params['social_loss_D']

                        d_cost += cost if curr_state != 3 else cost_params['replace_D']
                        if curr_state == 3:
                            s_cost += cost_params['social_loss_D']
                            fails += 1
                        curr_state = 0
                        action_taken = True

                elif strategy == 'preventive_C':
                    if curr_state >= 2:
                        if curr_state == 2: cost = cost_params['repair_C']
                        else: cost = cost_params['replace_D'] + cost_params['social_loss_D']

                        d_cost += cost if curr_state != 3 else cost_params['replace_D']
                        if curr_state == 3:
                            s_cost += cost_params['social_loss_D']
                            fails += 1
                        curr_state = 0
                        action_taken = True

                elif strategy == 'corrective':
                    if curr_state == 3:
                        d_cost += cost_params['replace_D']
                        s_cost += cost_params['social_loss_D']
                        fails += 1
                        curr_state = 0
                        action_taken = True

                if not action_taken:
                    curr_state = self._get_next_state(curr_state, transition_probs)
                else:
                    curr_state = self._get_next_state(0, transition_probs)

            unit_direct_costs[i] = d_cost
            unit_social_costs[i] = s_cost
            unit_failures[i] = fails

        return {
            'strategy': strategy,
            'direct_costs': unit_direct_costs,
            'social_costs': unit_social_costs,
            'total_costs': unit_direct_costs + unit_social_costs,
            'failures': unit_failures
        }

# 3. Comprehensive Analysis & Export Manager
# ----------------------------------------------------------------------
def run_comprehensive_analysis():
    set_publication_style()
    sim = LCCSimulator()
    base_probs = {'A->B': 0.010, 'B->C': 0.020, 'C->D': 0.030}
    strategies = ['corrective', 'preventive_C', 'preventive_B']
    labels = {
        'corrective': 'Corrective (State D)',
        'preventive_C': 'Preventive (State C)',
        'preventive_B': 'Pre-emptive (State B)'
    }
    colors = {'corrective': '#c44e52', 'preventive_C': '#dd8452', 'preventive_B': '#4c72b0'}

    # ------------------------------------------------------------------
    # Directory Setup for Export
    # ------------------------------------------------------------------
    export_dir = "LCC_Simulation_Results"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir) # Clean up previous runs
    os.makedirs(export_dir)
    print(f"Created export directory: {export_dir}")

    # ==================================================================
    # Perspective 1: Convergence Analysis
    # ==================================================================
    print("\n--- Perspective 1: Convergence Check ---")
    n_steps = [100, 500, 1000, 5000, 10000]
    conv_results = []

    for n in tqdm(n_steps, desc="Calculating Convergence"):
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, base_probs, num_units=n)
            mean_cost = np.mean(res['total_costs'])
            std_error = np.std(res['total_costs']) / np.sqrt(n)
            conv_results.append({'N': n, 'Strategy': labels[strat], 'MeanCost': mean_cost, 'SE': std_error})

    df_conv = pd.DataFrame(conv_results)
    df_conv.to_csv(f"{export_dir}/Perspective1_Convergence.csv", index=False)

    plt.figure(figsize=(10, 5))
    for strat in df_conv['Strategy'].unique():
        subset = df_conv[df_conv['Strategy'] == strat]
        plt.errorbar(subset['N'], subset['MeanCost'], yerr=subset['SE']*1.96,
                     marker='o', label=strat, capsize=5)
    plt.xscale('log')
    plt.xlabel('Number of Simulation Trials (N)')
    plt.ylabel('Mean Total Cost (with 95% CI)')
    plt.title('Perspective 1: Convergence of Calculation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective1_Convergence.png")
    plt.show()

    # ==================================================================
    # Perspective 2: Risk Profile (Distribution)
    # ==================================================================
    print("\n--- Perspective 2: Risk Profile Analysis (N=10,000) ---")
    dist_data = []
    fixed_n = 10000
    dist_csv_dict = {}

    plt.figure(figsize=(10, 6))
    for strat in strategies:
        res = sim.run_cohort_simulation(strat, base_probs, num_units=fixed_n)
        sns.kdeplot(res['total_costs'], label=labels[strat], color=colors[strat], fill=True, alpha=0.3)
        dist_data.append(res)
        dist_csv_dict[labels[strat]] = res['total_costs']

    pd.DataFrame(dist_csv_dict).to_csv(f"{export_dir}/Perspective2_Distributions.csv", index=False)

    plt.xlabel('Total Lifecycle Cost per Unit (Direct + Social)')
    plt.ylabel('Probability Density')
    plt.title('Perspective 2: Risk Profile Distribution (Fat Tail Analysis)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective2_RiskProfile.png")
    plt.show()

    # ==================================================================
    # Perspective 3: Sensitivity Analysis
    # ==================================================================
    print("\n--- Perspective 3: Sensitivity to Social Cost ---")
    social_costs_range = np.linspace(0, 1000, 11)
    sens_results = []

    for sc in tqdm(social_costs_range, desc="Running Sensitivity"):
        current_params = sim.default_costs.copy()
        current_params['social_loss_D'] = sc
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, base_probs, cost_params=current_params, num_units=2000)
            sens_results.append({
                'SocialCostParam': sc,
                'Strategy': labels[strat],
                'MeanTotalCost': np.mean(res['total_costs'])
            })

    df_sens = pd.DataFrame(sens_results)
    df_sens.to_csv(f"{export_dir}/Perspective3_Sensitivity.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sens, x='SocialCostParam', y='MeanTotalCost', hue='Strategy', marker='o', palette=[colors[s] for s in strategies])
    plt.axvline(x=500, color='gray', linestyle=':', label='Baseline Assumption (500)')
    plt.xlabel('Assumed Social Cost of Failure (Unit Price)')
    plt.ylabel('Mean Total Lifecycle Cost')
    plt.title('Perspective 3: Sensitivity Analysis (Tipping Point Identification)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective3_Sensitivity.png")
    plt.show()

    # ==================================================================
    # Perspective 4: Cost-Risk Trade-off
    # ==================================================================
    print("\n--- Perspective 4: Direct Cost vs Social Risk Trade-off ---")
    tradeoff_data = []
    for i, strat in enumerate(strategies):
        res = dist_data[i] # Use N=10000 data
        tradeoff_data.append({
            'Strategy': labels[strat],
            'DirectCost': np.mean(res['direct_costs']),
            'Risk (Failures)': np.mean(res['failures'])
        })

    df_tradeoff = pd.DataFrame(tradeoff_data)
    df_tradeoff.to_csv(f"{export_dir}/Perspective4_Tradeoff.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff, x='Risk (Failures)', y='DirectCost', hue='Strategy', s=300, palette=[colors[s] for s in strategies])

    for i in range(len(df_tradeoff)):
        plt.text(df_tradeoff['Risk (Failures)'][i], df_tradeoff['DirectCost'][i]+5,
                 df_tradeoff['Strategy'][i], ha='center')

    plt.title('Perspective 4: Optimization Landscape (Cost vs Risk)')
    plt.xlabel('Social Risk (Avg Failures per Unit)')
    plt.ylabel('Direct Maintenance Cost')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective4_Tradeoff.png")
    plt.show()

    # ==================================================================
    # Perspective 5: Statistical Significance Testing
    # ==================================================================
    print("\n--- Perspective 5: Statistical Significance ---")
    corr_costs = dist_data[0]['total_costs'] # corrective
    prev_costs = dist_data[1]['total_costs'] # preventive_C

    t_stat, p_val_t = stats.ttest_ind(corr_costs, prev_costs, equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(corr_costs, prev_costs, alternative='two-sided')

    stats_data = [{
        'Comparison': 'Corrective vs Preventive(C)',
        'Test': "Welch's t-test",
        'Statistic': t_stat,
        'P-Value': p_val_t,
        'Significant (p<0.05)': p_val_t < 0.05
    }, {
        'Comparison': 'Corrective vs Preventive(C)',
        'Test': "Mann-Whitney U",
        'Statistic': u_stat,
        'P-Value': p_val_u,
        'Significant (p<0.05)': p_val_u < 0.05
    }]
    pd.DataFrame(stats_data).to_csv(f"{export_dir}/Perspective5_StatisticalTests.csv", index=False)
    print("Statistical tests saved to CSV.")

    # ==================================================================
    # Perspective 5.5: Correlation Analysis (Reviewer Request)
    # ==================================================================
    print("\n--- Perspective 5.5: Correlation Analysis ---")
    res_corr = sim.run_cohort_simulation('corrective', base_probs, num_units=10000)
    df_analysis = pd.DataFrame({
        'Direct_Cost': res_corr['direct_costs'],
        'Social_Cost': res_corr['social_costs'],
        'Total_Cost': res_corr['total_costs'],
        'Failure_Count': res_corr['failures']
    })

    # 1. Spearman Correlation Matrix
    corr_matrix = df_analysis.corr(method='spearman')
    corr_matrix.to_csv(f"{export_dir}/Perspective5_5_CorrelationMatrix.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
    plt.title('Perspective 5.5: Correlation Matrix (Corrective Strategy)\n[Engineering Costs vs. Social Impacts]')
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective5_5_CorrelationHeatmap.png")
    plt.show()

    # 2. Pairplot (Sampled for performance/visibility)
    sns.pairplot(df_analysis.sample(min(1000, len(df_analysis))), kind='reg', diag_kind='kde',
                 plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.suptitle('Variable Interaction Analysis', y=1.02)
    plt.savefig(f"{export_dir}/Perspective5_5_VariableInteraction_Pairplot.png")
    plt.show()

    # ==================================================================
    # Perspective 6: Tail Risk Quantification (VaR / CVaR)
    # ==================================================================
    print("\n--- Perspective 6: Tail Risk (VaR/CVaR) ---")
    risk_metrics = []
    for i, strat in enumerate(strategies):
        data = dist_data[i]['total_costs']
        var_95 = np.percentile(data, 95)
        cvar_95 = data[data >= var_95].mean()
        risk_metrics.append({
            'Strategy': labels[strat],
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95
        })

    df_risk = pd.DataFrame(risk_metrics)
    df_risk.to_csv(f"{export_dir}/Perspective6_RiskMetrics.csv", index=False)

    df_risk.set_index('Strategy')[['VaR (95%)', 'CVaR (95%)']].plot(kind='bar', figsize=(10, 6), color=['#e74c3c', '#8e44ad'])
    plt.title('Perspective 6: Extreme Tail Risk Metrics (VaR & CVaR @ 95%)')
    plt.ylabel('Cost Value (Tail Loss)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective6_TailRisk.png")
    plt.show()

    # ==================================================================
    # Final Step: Create ZIP Archive and Download
    # ==================================================================
    print("\n--- Creating ZIP Archive ---")
    zip_filename = "LCC_Simulation_Results.zip"

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(export_dir):
            for file in files_in_dir:
                file_path = os.path.join(root, file)
                # Add file to ZIP with relative path
                zipf.write(file_path, os.path.relpath(file_path, start=os.path.dirname(export_dir)))

    print(f"Successfully created: {zip_filename} containing all CSVs and PNGs.")

    if IN_COLAB:
        print("Triggering download...")
        files.download(zip_filename)
    else:
        print(f"Execution complete. Please find '{zip_filename}' in your local directory.")

if __name__ == "__main__":
    run_comprehensive_analysis()
