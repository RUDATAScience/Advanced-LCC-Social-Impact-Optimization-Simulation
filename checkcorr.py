# ======================================================================
# Advanced LCC & Social Impact Optimization Simulation
# (Stochastic Costs Included for Realistic Correlation)
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

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

warnings.filterwarnings('ignore')

def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

class LCCSimulator:
    def __init__(self, seed=42):
        self.rng = Generator(PCG64(seed))
        self.states = ['A', 'B', 'C', 'D']
        self.default_costs = {'inspection': 1, 'repair_B': 15, 'repair_C': 60, 'replace_D': 250, 'social_loss_D': 500}

    def _get_next_state(self, current_state_idx, probs):
        if self.states[current_state_idx] == 'D': return 3
        p = probs.get(f"{self.states[current_state_idx]}->{self.states[current_state_idx + 1]}", 0)
        return current_state_idx + 1 if self.rng.random() < p else current_state_idx

    # ★ここが修正ポイント：コストに確率分布（ばらつき）を導入
    def _get_stochastic_cost(self, cost_type, base_val):
        if cost_type == 'replace':
            # 直接費は正規分布（±15%程度のばらつき）
            return max(base_val * 0.5, self.rng.normal(loc=base_val, scale=base_val * 0.15))
        elif cost_type == 'social':
            # 社会的損失は対数正規分布（極端な大事故＝ファットテールを再現）
            mu = np.log(base_val) - (0.8**2) / 2
            return self.rng.lognormal(mean=mu, sigma=0.8)
        return base_val

    def run_cohort_simulation(self, strategy, transition_probs, cost_params=None, duration_years=50, num_units=1000):
        if cost_params is None: cost_params = self.default_costs
        months = duration_years * 12
        unit_direct_costs, unit_social_costs, unit_failures = np.zeros(num_units), np.zeros(num_units), np.zeros(num_units)
        
        for i in range(num_units):
            curr_state, d_cost, s_cost, fails = 0, 0, 0, 0
            for m in range(months):
                d_cost += cost_params['inspection']
                action_taken = False
                
                if strategy == 'preventive_B' and curr_state >= 1:
                    if curr_state == 1: cost = cost_params['repair_B']
                    elif curr_state == 2: cost = cost_params['repair_C']
                    else: cost = self._get_stochastic_cost('replace', cost_params['replace_D'])
                    d_cost += cost
                    if curr_state == 3:
                        s_cost += self._get_stochastic_cost('social', cost_params['social_loss_D'])
                        fails += 1
                    curr_state, action_taken = 0, True

                elif strategy == 'preventive_C' and curr_state >= 2:
                    if curr_state == 2: cost = cost_params['repair_C']
                    else: cost = self._get_stochastic_cost('replace', cost_params['replace_D'])
                    d_cost += cost
                    if curr_state == 3:
                        s_cost += self._get_stochastic_cost('social', cost_params['social_loss_D'])
                        fails += 1
                    curr_state, action_taken = 0, True
                
                elif strategy == 'corrective' and curr_state == 3:
                    d_cost += self._get_stochastic_cost('replace', cost_params['replace_D'])
                    s_cost += self._get_stochastic_cost('social', cost_params['social_loss_D'])
                    fails += 1
                    curr_state, action_taken = 0, True
                
                curr_state = self._get_next_state(0, transition_probs) if action_taken else self._get_next_state(curr_state, transition_probs)

            unit_direct_costs[i], unit_social_costs[i], unit_failures[i] = d_cost, s_cost, fails

        return {'strategy': strategy, 'direct_costs': unit_direct_costs, 'social_costs': unit_social_costs, 'total_costs': unit_direct_costs + unit_social_costs, 'failures': unit_failures}

def run_comprehensive_analysis():
    set_publication_style()
    sim = LCCSimulator()
    base_probs = {'A->B': 0.010, 'B->C': 0.020, 'C->D': 0.030}
    strategies = ['corrective', 'preventive_C', 'preventive_B']
    labels = {'corrective': 'Corrective (State D)', 'preventive_C': 'Preventive (State C)', 'preventive_B': 'Pre-emptive (State B)'}
    
    export_dir = "LCC_Simulation_Results_Stochastic"
    if os.path.exists(export_dir): shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    # Perspective 5.5: Correlation Analysis (Only executing this part for speed, but keeping structure)
    print("\n--- Perspective 5.5: Correlation Analysis (Stochastic Costs) ---")
    res_corr = sim.run_cohort_simulation('corrective', base_probs, num_units=100000)
    df_analysis = pd.DataFrame({
        'Direct_Cost': res_corr['direct_costs'], 'Social_Cost': res_corr['social_costs'],
        'Total_Cost': res_corr['total_costs'], 'Failure_Count': res_corr['failures']
    })

    corr_matrix = df_analysis.corr(method='spearman')
    corr_matrix.to_csv(f"{export_dir}/Perspective5_5_CorrelationMatrix.csv")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
    plt.title('Perspective 5.5: Correlation Matrix (Corrective Strategy)\n[Stochastic Costs vs. Impacts]')
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective5_5_CorrelationHeatmap.png")
    plt.show()

    sns.pairplot(df_analysis.sample(min(1000, len(df_analysis))), kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.suptitle('Variable Interaction Analysis (Stochastic Variables)', y=1.02)
    plt.savefig(f"{export_dir}/Perspective5_5_VariableInteraction_Pairplot.jpg")
    plt.show()

    # Create ZIP
    zip_filename = "LCC_Simulation_Results_Stochastic.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(export_dir):
            for file in files_in_dir:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), start=os.path.dirname(export_dir)))
    if IN_COLAB: files.download(zip_filename)

if __name__ == "__main__":
    run_comprehensive_analysis()
