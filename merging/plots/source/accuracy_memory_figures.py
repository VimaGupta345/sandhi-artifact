import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

from sandhi_colors import systems

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "../data"
FIGURES_DIR = "../figures"

# System order for plots
SYSTEM_ORDER_FULL = ["vllm", "sandhi", "fullmerge", "lora"]
SYSTEM_ORDER_REDUCED = ["vllm", "sandhi"]

# Configurations that include LoRA and Full-merge baselines
CONFIGS_WITH_BASELINES = ['2-DS', '2-Qwen-7B', '3-Qwen-32B', '5-llama']

# Configuration that shows the legend (None = no legend on any figure)
CONFIG_WITH_LEGEND = None

# Domain to benchmark name mappings
DOMAIN_MAPS = {
    'llama': {
        'medical': 'Medical', 'financial': 'Finance', 'legal': 'Legal',
        'toxicity': 'Safety', 'truthfulness': 'Truth'
    },
    'qwen25': {'math': 'Math', 'coding': 'Coder'},
    'qwen3': {'medical': 'MedGo', 'IF': 'Light-IF 32B', 'Russian': 'T-pro-it 2.0'},
    'deepseek': {'math': 'DS-Math', 'coding': 'DS-Coder'},
}

# LoRA model name mappings
LORA_MODEL_MAP = {
    'finance': 'Finance', 'legal': 'Legal', 'medical': 'Medical',
    'toxicity': 'Safety', 'truthfulness': 'Truth',
    'qihoo360/Light-IF-32B': 'Light-IF 32B',
    'OpenMedZoo/MedGo': 'MedGo',
    't-tech/T-pro-it-2.0': 'T-pro-it 2.0',
}

# Plot styling constants
PLOT_STYLE = {
    'font_size': 14,
    'bar_edge_color': '#000000',
    'bar_linewidth': 0.5,
    'missing_marker_color': '#636363',
    'grid_color': 'lightgrey',
    'memory_line_alpha': 0.8,
    'memory_line_width': 2,
    'group_gap': 2,
    'figure_height': 4,
    'base_width_per_benchmark': 2.5,
}

# Figure sizes (width, height) in inches
FIGSIZE_CONFIG = {
    '2-DS': (5.5, 2.5),
    '2-Qwen-7B': (5.5, 2.5),
    '3-Qwen-32B': (5.5, 2.5),
    '5-llama': (6.5, 2.5),
    '7-5-llama-2-qwen': (6.5, 2.5),
    '7-5-llama-2-DS': (6.5, 2.5),
    '9-5-llama-2-DS-2-Qwen-7B': (10, 2.5),
    '12-model': (10, 2.5),
}

# Font sizes per configuration
FONT_SIZE_CONFIG = {
    '2-DS': 18,
    '2-Qwen-7B': 18,
    '3-Qwen-32B': 18,
    '5-llama': 16,
    '7-5-llama-2-qwen': 16,
    '7-5-llama-2-DS': 16,
    '9-5-llama-2-DS-2-Qwen-7B': 14,
    '12-model': 14,
}

# Delta label rotation per configuration (degrees)
DELTA_ROTATION_CONFIG = {
    '5-llama': 90,
    '7-5-llama-2-qwen': 90,
    '7-5-llama-2-DS': 90,
}

# Configurations that get +4 font boost for y-axis labels, y-ticks, and data labels
FONT_BOOST_CONFIGS = ['2-DS', '2-Qwen-7B']
FONT_BOOST = 4


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_baseline_data():
    """Load vLLM-no-merge baseline data."""
    df = pd.read_csv(f"{DATA_DIR}/vllm-no-merge.csv")
    baselines = {}
    for _, row in df.iterrows():
        is_qwen = 'qwen' in row['base_model'].lower()
        domain_map = {
            'coding': 'Coder' if is_qwen else 'DS-Coder',
            'math': 'Math' if is_qwen else 'DS-Math',
            **DOMAIN_MAPS['llama'],
            **DOMAIN_MAPS['qwen3'],
            'tinyMMLU': 'tinyMMLU',
        }
        key = domain_map.get(row['domain'], row['domain'])
        baselines[key] = row['base_score']
    return baselines


def load_fullmerge_data():
    """Load full-merge baseline data (multi-slerp values)."""
    fullmerge = {}

    # Load from CSV files (multi_slerp column)
    csv_configs = [
        ('llama3.1_domain_results.csv', DOMAIN_MAPS['llama'], 'multi_slerp'),
        ('qwen2_5_domain_results.csv', DOMAIN_MAPS['qwen25'], 'multi_slerp'),
        ('qwen3_domain_results.csv', DOMAIN_MAPS['qwen3'], 'multi_slerp'),
    ]
    for filename, domain_map, col in csv_configs:
        df = pd.read_csv(f"{DATA_DIR}/full_merge/{filename}")
        for _, row in df.iterrows():
            key = domain_map.get(row['domain'], row['domain'])
            fullmerge[key] = row[col]

    # Load DeepSeek from Excel (slerp column)
    df = pd.read_csv(f"{DATA_DIR}/full_merge/deepseek_domain_results.csv")
    for _, row in df.iterrows():
        key = DOMAIN_MAPS['deepseek'].get(row['domain'], row['domain'])
        fullmerge[key] = row['slerp']

    return fullmerge


def load_lora_data():
    """Load LoRA adapter baseline data."""
    lora = {}

    # Llama LoRA
    df = pd.read_csv(f"{DATA_DIR}/lora/llama_adapter_acc.csv")
    for _, row in df.iterrows():
        key = LORA_MODEL_MAP.get(row['model'])
        if key:
            lora[key] = row['lora_acc'] * 100  # Convert to percentage

    # Qwen32b LoRA
    df = pd.read_csv(f"{DATA_DIR}/lora/qwen32b_adapter_acc.csv")
    for _, row in df.iterrows():
        key = LORA_MODEL_MAP.get(row['model'])
        if key:
            lora[key] = row['lora_acc'] * 100

    return lora


def load_sandhi_data():
    """Load Sandhi data from all_models_final.csv."""
    df = pd.read_csv(f"{DATA_DIR}/all_models_final.csv")
    sandhi_data = {}

    for set_name in df['Set'].unique():
        set_df = df[df['Set'] == set_name]
        sandhi_data[set_name] = {
            'benchmarks': {row['Model']: row['Accuracy_Delta_Percent'] for _, row in set_df.iterrows()},
            'memory_saved_gb': set_df['Memory_Saved_GB'].iloc[0]
        }

    return sandhi_data


def load_memory_savings():
    """Load memory savings data for all systems."""
    df = pd.read_csv(f"{DATA_DIR}/memory_savings.csv")
    memory_data = {}

    for _, row in df.iterrows():
        memory_data[row['Configuration']] = {
            'num_models': int(row['Num Models']),
            'total': float(row['Total Memory (GB)']),
            'vllm': float(row['vLLM-no-merge (GB)']),
            'lora': None if row['LoRA (GB)'] == 'x' else float(row['LoRA (GB)']),
            'fullmerge': None if row['Full-merge (GB)'] == 'x' else float(row['Full-merge (GB)']),
            'sandhi': float(row['Sandhi (GB)']),
            'vllm_pct': float(row['vLLM-no-merge (%)']),
            'lora_pct': None if row['LoRA (%)'] == 'x' else float(row['LoRA (%)']),
            'fullmerge_pct': None if row['Full-merge (%)'] == 'x' else float(row['Full-merge (%)']),
            'sandhi_pct': float(row['Sandhi (%)']),
        }

    return memory_data


def load_ft_baselines():
    """Load fine-tuned model baseline accuracies."""
    ft_baselines = {}

    # Load from CSV files
    csv_configs = [
        ('llama3.1_domain_results.csv', DOMAIN_MAPS['llama']),
        ('qwen2_5_domain_results.csv', DOMAIN_MAPS['qwen25']),
        ('qwen3_domain_results.csv', DOMAIN_MAPS['qwen3']),
    ]
    for filename, domain_map in csv_configs:
        df = pd.read_csv(f"{DATA_DIR}/full_merge/{filename}")
        for _, row in df.iterrows():
            key = domain_map.get(row['domain'], row['domain'])
            if pd.notna(row['ft_score']) and row['ft_score'] != 'N/A':
                ft_baselines[key] = float(row['ft_score'])

    # Load DeepSeek FT baselines from Excel
    df = pd.read_csv(f"{DATA_DIR}/full_merge/deepseek_domain_results.csv")
    for _, row in df.iterrows():
        key = DOMAIN_MAPS['deepseek'].get(row['domain'], row['domain'])
        ft_baselines[key] = row['ft_score']

    return ft_baselines


# =============================================================================
# Plotting Functions
# =============================================================================

def get_accuracy_for_system(sys_key, bench, ft_baseline, benchmarks,
                            fullmerge_data, lora_data, memory_savings):
    """Get accuracy value for a given system and benchmark."""
    if sys_key == "vllm":
        return ft_baseline
    elif sys_key == "fullmerge":
        if memory_savings.get('fullmerge') is None:
            return None
        return fullmerge_data.get(bench)
    elif sys_key == "lora":
        if memory_savings.get('lora') is None:
            return None
        return lora_data.get(bench)
    else:  # sandhi
        delta = benchmarks.get(bench)
        if delta is not None and ft_baseline is not None:
            return ft_baseline + delta
        return None


def create_accuracy_memory_figure(set_name, benchmarks, memory_saved_gb,
                                   ft_baselines, fullmerge_data, lora_data,
                                   memory_savings, figname, figsize):
    """Create a two-panel figure with accuracy bars (left) and memory saving percentage (right)."""
    style = PLOT_STYLE.copy()
    # Override font size based on configuration
    style['font_size'] = FONT_SIZE_CONFIG.get(set_name, style['font_size'])

    # Font boost for specific configs (y-axis labels, y-ticks, data labels)
    font_boost = FONT_BOOST if set_name in FONT_BOOST_CONFIGS else 0

    benchmark_list = list(benchmarks.keys())
    n_benchmarks = len(benchmark_list)

    # Use full system order for configs with baselines, reduced for others
    system_order = SYSTEM_ORDER_FULL if set_name in CONFIGS_WITH_BASELINES else SYSTEM_ORDER_REDUCED
    n_systems = len(system_order)

    # Show legend only for specific config
    show_legend = (set_name == CONFIG_WITH_LEGEND)

    # Build x positions for accuracy plot
    x = []
    bar_idx = 1
    for _ in range(n_benchmarks):
        for _ in range(n_systems):
            x.append(bar_idx)
            bar_idx += 1
        bar_idx += style['group_gap']

    # Build xticks and labels
    xticks = []
    xlbls = []
    bar_idx = 1
    for bench in benchmark_list:
        for sys_idx in range(n_systems):
            xticks.append(bar_idx)
            xlbls.append(bench if sys_idx == n_systems // 2 else '')
            bar_idx += 1
        bar_idx += style['group_gap']

    # Adjust figsize for two panels
    two_panel_figsize = (figsize[0] + 3, figsize[1])

    with plt.style.context('paper.mplstyle'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=two_panel_figsize,
                                        gridspec_kw={'width_ratios': [figsize[0], 3]})

        # =====================================================================
        # Left Panel: Accuracy
        # =====================================================================
        data_idx = 0
        for bench in benchmark_list:
            ft_baseline = ft_baselines.get(bench)
            for sys_key in system_order:
                acc = get_accuracy_for_system(
                    sys_key, bench, ft_baseline, benchmarks,
                    fullmerge_data, lora_data, memory_savings
                )
                if acc is None:
                    ax1.text(x[data_idx], 2, 'x', ha='center', va='bottom',
                            fontsize=style['font_size'], fontweight='bold',
                            color=style['missing_marker_color'])
                else:
                    ax1.bar(x[data_idx], acc,
                           color=systems[sys_key]['color'],
                           edgecolor=style['bar_edge_color'],
                           linewidth=style['bar_linewidth'],
                           hatch=systems[sys_key]['hatch'])
                    # Add accuracy delta annotation (skip vllm as it's the baseline)
                    if sys_key != "vllm" and ft_baseline is not None:
                        delta = acc - ft_baseline
                        delta_str = f"{delta:+.1f}" if delta != 0 else "0"
                        rotation = DELTA_ROTATION_CONFIG.get(set_name, 45)
                        ha = 'center' if rotation == 90 else 'left'
                        ax1.text(x[data_idx], acc + 1, delta_str,
                                ha=ha, va='bottom',
                                fontsize=style['font_size'] - 2 + font_boost,
                                rotation=rotation)
                data_idx += 1

        # Configure accuracy axes
        ax1.set_ylabel(r'Accuracy (%)', fontsize=style['font_size'] + font_boost)
        ax1.set_ylim([0, 100])
        ax1.yaxis.set_ticks(np.arange(0, 101, 20))
        ax1.set_xlim([0, max(x) + 2])
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xlbls, fontsize=style['font_size'])
        ax1.tick_params(axis='y', labelsize=style['font_size'] + font_boost)

        # Tick params for accuracy plot
        ax1.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)
        ax1.tick_params(axis='x', which='major', bottom=False, top=False, labelbottom=True)
        ax1.tick_params(axis='y', which='minor', left=True, right=False)
        ax1.tick_params(axis='y', which='major', left=True, right=False)

        # Add dashed vertical separators between groups
        separator_positions = []
        bar_idx = 1
        for bench_idx in range(n_benchmarks):
            bar_idx += n_systems
            if bench_idx < n_benchmarks - 1:
                separator_positions.append(bar_idx + style['group_gap'] / 2 - 0.5)
            bar_idx += style['group_gap']

        for sep_x in separator_positions:
            ax1.axvline(x=sep_x, color=style['grid_color'], linestyle='--', linewidth=1)

        # Grid for accuracy plot
        ax1.set_axisbelow(True)
        ax1.grid(color=style['grid_color'], linestyle='dashed', axis='y', alpha=0.7)

        # Legend on accuracy plot (only for specific config)
        if show_legend:
            legend_elements = [
                Patch(facecolor=systems[s]['color'], edgecolor=style['bar_edge_color'],
                      hatch=systems[s]['hatch'], label=systems[s]['name'])
                for s in system_order
            ]
            ax1.legend(handles=legend_elements, loc='upper left', ncols=len(system_order),
                       fontsize=style['font_size'])

        # =====================================================================
        # Right Panel: Memory Saving Percentage (excluding vllm/no-merge)
        # =====================================================================
        # Also exclude lora for 2-DS and 2-Qwen-7B
        if set_name in ['2-DS', '2-Qwen-7B']:
            mem_system_order = [s for s in system_order if s not in ['vllm', 'lora']]
        else:
            mem_system_order = [s for s in system_order if s != 'vllm']
        n_mem_systems = len(mem_system_order)

        mem_x = []
        mem_bar_idx = 1
        for sys_idx in range(n_mem_systems):
            mem_x.append(mem_bar_idx)
            mem_bar_idx += 1

        for sys_idx, sys_key in enumerate(mem_system_order):
            mem_pct = memory_savings.get(f'{sys_key}_pct')
            if mem_pct is None:
                ax2.text(mem_x[sys_idx], 2, 'x', ha='center', va='bottom',
                        fontsize=style['font_size'], fontweight='bold',
                        color=style['missing_marker_color'])
            else:
                ax2.bar(mem_x[sys_idx], mem_pct,
                       color=systems[sys_key]['color'],
                       edgecolor=style['bar_edge_color'],
                       linewidth=style['bar_linewidth'],
                       hatch=systems[sys_key]['hatch'])
                # Add percentage label on bar
                ax2.text(mem_x[sys_idx], mem_pct + 1, f'{mem_pct:.1f}%',
                        ha='center', va='bottom',
                        fontsize=style['font_size'] - 2 + font_boost)

        # Configure memory percentage axes (y-axis on right)
        ax2.set_ylim([0, 100])
        ax2.yaxis.set_ticks(np.arange(0, 101, 20))
        ax2.set_xlim([0, max(mem_x) + 1])
        ax2.set_xticks(mem_x)
        ax2.set_xticklabels([''] * len(mem_x))  # No x labels, legend shows systems
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel(r'Memory Saving (%)', fontsize=style['font_size'] + font_boost)
        ax2.tick_params(axis='y', labelsize=style['font_size'] + font_boost)

        # Tick params for memory plot
        ax2.tick_params(axis='x', which='major', bottom=False, top=False, labelbottom=False)
        ax2.tick_params(axis='y', which='minor', left=False, right=True)
        ax2.tick_params(axis='y', which='major', left=False, right=True)

        # Grid for memory plot
        ax2.set_axisbelow(True)
        ax2.grid(color=style['grid_color'], linestyle='dashed', axis='y', alpha=0.7)

        plt.tight_layout()
        fig.savefig(f'{FIGURES_DIR}/{figname}.pdf')
        print(f"Saved {FIGURES_DIR}/{figname}.pdf")
        plt.show(block=False)
        plt.pause(0.5)


def create_legend_figure():
    """Create a standalone legend figure."""
    style = PLOT_STYLE

    with plt.style.context('paper.mplstyle'):
        fig, ax = plt.subplots(figsize=(12.5, 1))
        ax.axis('off')

        legend_elements = [
            Patch(facecolor=systems[s]['color'], edgecolor=style['bar_edge_color'],
                  hatch=systems[s]['hatch'], label=systems[s]['name'])
            for s in SYSTEM_ORDER_FULL
        ]
        ax.legend(handles=legend_elements, loc='center',
                  ncols=len(SYSTEM_ORDER_FULL), fontsize=style['font_size'], frameon=False)

        fig.savefig(f'{FIGURES_DIR}/legend.pdf', bbox_inches='tight')
        print(f"Saved {FIGURES_DIR}/legend.pdf")
        plt.show(block=False)
        plt.pause(0.5)


def get_figsize(set_name):
    """Get figure size based on configuration."""
    return FIGSIZE_CONFIG.get(set_name, (7, 2.5))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Loading baseline data...")
    ft_baselines = load_ft_baselines()
    fullmerge_data = load_fullmerge_data()
    lora_data = load_lora_data()
    sandhi_data = load_sandhi_data()
    memory_savings_data = load_memory_savings()

    print(f"Found {len(sandhi_data)} model sets: {list(sandhi_data.keys())}")

    for set_name, data in sandhi_data.items():
        print(f"\nGenerating figure for: {set_name}")
        figname = set_name.replace(' ', '_').replace('/', '-')
        mem_savings = memory_savings_data.get(set_name, {
            'vllm': 0, 'lora': None, 'fullmerge': None, 'sandhi': data['memory_saved_gb']
        })
        create_accuracy_memory_figure(
            set_name, data['benchmarks'], data['memory_saved_gb'],
            ft_baselines, fullmerge_data, lora_data, mem_savings,
            figname, get_figsize(set_name)
        )

    create_legend_figure()
    print("\nAll figures generated!")
