import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .config import (
    DATASETS, ANALYSIS_BANDS, COND_COLORS, MONKEY_MARKERS, COND_TO_DRUG,
)
from .statistics import bootstrap_ci_cohens_d

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def _make_legend_handles():
    handles = [Line2D([0], [0], color=COND_COLORS[c], lw=2, label=c) for c in COND_COLORS]
    handles += [Line2D([0], [0], color='gray', marker=m, ls='', label=n)
                for n, m in MONKEY_MARKERS.items()]
    return handles


# Figure 1: Kuramoto Synchrony
def plot_kuramoto_sync(kuramoto_results):
    handles = _make_legend_handles()
    fig, axes = plt.subplots(1, len(ANALYSIS_BANDS), figsize=(22, 5), sharey=False)
    for idx, band in enumerate(ANALYSIS_BANDS):
        ax = axes[idx]
        for ds_name, ds in DATASETS.items():
            aw = kuramoto_results[ds_name].get('Awake', {}).get(band, {}).get('sync', 0)
            un = kuramoto_results[ds_name].get('Unconscious', {}).get(band, {}).get('sync', 0)
            ax.plot([0, 1], [aw, un], color=COND_COLORS[ds['condition']],
                    marker=MONKEY_MARKERS[ds['monkey']], ls='-', lw=1.5, markersize=8, alpha=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Awake', 'Unconscious'])
        ax.set_title(band, fontsize=13)
        if idx == 0:
            ax.set_ylabel('Kuramoto Synchrony <R(t)>')
    fig.legend(handles=handles, loc='upper right', ncol=3, fontsize=10)
    fig.suptitle('Kuramoto Synchrony (CAR) - Awake vs Unconscious', fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig1_kuramoto_sync.png', dpi=150, bbox_inches='tight')
    plt.show()


# Figure 2: wPLI
def plot_wpli(wpli_results):
    handles = _make_legend_handles()
    fig, axes = plt.subplots(1, len(ANALYSIS_BANDS), figsize=(22, 5), sharey=False)
    for idx, band in enumerate(ANALYSIS_BANDS):
        ax = axes[idx]
        for ds_name, ds in DATASETS.items():
            aw = wpli_results[ds_name].get('Awake', {}).get(band, 0)
            un = wpli_results[ds_name].get('Unconscious', {}).get(band, 0)
            ax.plot([0, 1], [aw, un], color=COND_COLORS[ds['condition']],
                    marker=MONKEY_MARKERS[ds['monkey']], ls='-', lw=1.5, markersize=8, alpha=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Awake', 'Unconscious'])
        ax.set_title(band, fontsize=13)
        if idx == 0:
            ax.set_ylabel('wPLI')
    fig.legend(handles=handles, loc='upper right', ncol=3, fontsize=10)
    fig.suptitle('wPLI (CAR) - Awake vs Unconscious', fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig2_wpli.png', dpi=150, bbox_inches='tight')
    plt.show()


# Figure 3: FCD & Coalition Entropy
def plot_fcd_coalition(fcd_results, coal_results):
    handles = _make_legend_handles()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ds_name, ds in DATASETS.items():
        color = COND_COLORS[ds['condition']]
        marker = MONKEY_MARKERS[ds['monkey']]
        axes[0].plot([0, 1],
                     [fcd_results[ds_name].get('Awake', 0), fcd_results[ds_name].get('Unconscious', 0)],
                     color=color, marker=marker, ls='-', lw=1.5, markersize=8, alpha=0.8)
        axes[1].plot([0, 1],
                     [coal_results[ds_name].get('Awake', 0), coal_results[ds_name].get('Unconscious', 0)],
                     color=color, marker=marker, ls='-', lw=1.5, markersize=8, alpha=0.8)
    axes[0].set_title('FCD Variance (delta, CAR)'); axes[0].set_ylabel('FCD Var')
    axes[1].set_title('Coalition Entropy (delta, CAR)'); axes[1].set_ylabel('H_norm')
    for ax in axes:
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Awake', 'Unconscious'])
    fig.legend(handles=handles, loc='upper right', ncol=3, fontsize=10)
    fig.suptitle('FCD & Coalition Entropy (CAR, delta)', fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig3_fcd_coalition.png', dpi=150, bbox_inches='tight')
    plt.show()


# Figure 4: Effect Sizes
def plot_effect_sizes(all_tests, bf_cache):
    fig, ax = plt.subplots(figsize=(14, 8))
    labels = [f'{name} {band}' for name, band, *_ in all_tests]
    d_vals = [d for _, _, _, _, d, _ in all_tests]
    colors_bar = []
    for bf in bf_cache:
        if bf > 10: colors_bar.append('#1B5E20')
        elif bf > 3: colors_bar.append('#E53935')
        else: colors_bar.append('#90A4AE')
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, d_vals, color=colors_bar, edgecolor='white', height=0.65, alpha=0.85)
    for i, (name, band, aw, un, d, p) in enumerate(all_tests):
        ci_lo, ci_hi = bootstrap_ci_cohens_d(aw, un)
        ax.plot([ci_lo, ci_hi], [i, i], 'k-', lw=1.5, alpha=0.6)
        ax.plot([ci_lo, ci_hi], [i, i], 'k|', markersize=6)
        bf = bf_cache[i]
        label_text = f'p={p:.3f} BF={bf:.1f}' if bf > 1 else f'p={p:.3f}'
        offset = 0.05 if d >= 0 else -0.05
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + offset, i, label_text, va='center', ha=ha, fontsize=8)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='k', lw=1)
    ax.axvline(-0.8, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0.8, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel("Cohen's d (with 95% Bootstrap CI)")
    ax.set_title('Effect Sizes: Awake -> Unconscious (n=8 paired datasets)', fontsize=14)
    legend_el = [
        mpatches.Patch(color='#1B5E20', label='BF10 > 10 (strong)'),
        mpatches.Patch(color='#E53935', label='BF10 > 3 (moderate)'),
        mpatches.Patch(color='#90A4AE', label='n.s.'),
    ]
    ax.legend(handles=legend_el, loc='lower right')
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig4_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.show()


# Figure 5: JR Dose-Response
def plot_jr_dose_response(results_jr):
    drug_colors = {'Ketamine': '#FF9800', 'Medetomidine': '#9C27B0',
                   'Propofol': '#009688', 'Sleep': '#607D8B'}
    levels_x = {'light': 1, 'medium': 2, 'deep': 3}
    bl_jr = results_jr['Baseline']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, bl_val, title in zip(
        axes, ['sync', 'meta'], [bl_jr['sync'], bl_jr['meta']],
        ['Synchrony <R(t)>', 'Metastability Var[R(t)]']
    ):
        for drug, color in drug_colors.items():
            xs = [0] + [levels_x[l] for l in results_jr[drug]]
            ys = [bl_val] + [results_jr[drug][l][metric] for l in results_jr[drug]]
            ax.plot(xs, ys, 'o-', color=color, linewidth=2.5, markersize=9, label=drug)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Awake', 'Light', 'Medium', 'Deep'])
        ax.set_title(title, fontweight='bold'); ax.legend(fontsize=10); ax.set_ylabel(metric)
    plt.suptitle('Jansen-Rit: Dose-Response Curves', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig5_jr_dose_response.png', dpi=150, bbox_inches='tight')
    plt.show()


# Figure 6: Model vs Empirical
def plot_model_vs_empirical(kuramoto_results, results_jr, results_wc):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for col, (cond_short, drug) in enumerate(COND_TO_DRUG.items()):
        for row, metric in enumerate(['sync', 'meta']):
            ax = axes[row, col]
            for ds_name, ds in DATASETS.items():
                if ds['condition'] != cond_short:
                    continue
                aw = kuramoto_results[ds_name].get('Awake', {}).get('broadband', {}).get(metric, 0)
                un = kuramoto_results[ds_name].get('Unconscious', {}).get('broadband', {}).get(metric, 0)
                ax.plot([0, 1], [aw, un], marker=MONKEY_MARKERS[ds['monkey']],
                        color='#333333', ls='-', lw=1.5, markersize=8, alpha=0.7)
            bl_val = results_jr['Baseline'][metric]
            jr_val = results_jr[drug]['medium'][metric]
            ax.plot([0, 1], [bl_val, jr_val], 'D-', color='#2196F3', lw=2.5, markersize=10, zorder=5)
            bl_wc = results_wc['Baseline'][metric]
            wc_val = results_wc[drug][metric]
            ax.plot([0, 1], [bl_wc, wc_val], '^--', color='#F44336', lw=2, markersize=9, alpha=0.7, zorder=4)
            ax.set_xticks([0, 1]); ax.set_xticklabels(['Awake', 'Uncon.'])
            if col == 0:
                ax.set_ylabel('Synchrony' if metric == 'sync' else 'Metastability', fontsize=11)
            if row == 0:
                ax.set_title(f'{cond_short}\n({drug})', fontsize=12, fontweight='bold')

    legend_el = [
        Line2D([0], [0], color='#333333', marker='o', ls='-', label='George (empirical)'),
        Line2D([0], [0], color='#333333', marker='s', ls='-', label='Chibi (empirical)'),
        Line2D([0], [0], color='#2196F3', marker='D', ls='-', lw=2.5, label='JR model'),
        Line2D([0], [0], color='#F44336', marker='^', ls='--', lw=2, label='WC model'),
    ]
    fig.legend(handles=legend_el, loc='upper center', ncol=4, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Model Predictions vs Empirical Data (Broadband Kuramoto, CAR)',
                 fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig6_model_vs_empirical.png', dpi=150, bbox_inches='tight')
    plt.show()


# Figure 7: Combined Summary
def plot_summary(all_tests, bf_cache, kuramoto_results, model_dirs, results_jr, results_wc):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.35)

    # Left panel: effect sizes
    ax1 = fig.add_subplot(gs[0])
    labels_full = [f'{name} {band}' for name, band, *_ in all_tests]
    d_vals_full = [d for _, _, _, _, d, _ in all_tests]
    ci_data = [bootstrap_ci_cohens_d(aw, un) for _, _, aw, un, _, _ in all_tests]
    y_pos = np.arange(len(labels_full))
    colors_bar = []
    for bf in bf_cache:
        if bf > 10: colors_bar.append('#1B5E20')
        elif bf > 3: colors_bar.append('#E53935')
        else: colors_bar.append('#90A4AE')
    ax1.barh(y_pos, d_vals_full, color=colors_bar, edgecolor='white', height=0.65, alpha=0.85)
    for i in range(len(d_vals_full)):
        ax1.plot([ci_data[i][0], ci_data[i][1]], [i, i], 'k-', lw=1.5, alpha=0.6)
        ax1.plot([ci_data[i][0], ci_data[i][1]], [i, i], 'k|', markersize=6)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(labels_full, fontsize=9)
    ax1.axvline(0, color='k', lw=1)
    ax1.axvline(-0.8, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax1.axvline(0.8, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax1.set_xlabel("Cohen's d (with 95% Bootstrap CI)")
    ax1.set_title('Empirical Effect Sizes\n(Awake -> Unconscious, n=8)', fontsize=12, fontweight='bold')
    legend_el1 = [
        mpatches.Patch(color='#1B5E20', label='BF10 > 10 (strong)'),
        mpatches.Patch(color='#E53935', label='BF10 > 3 (moderate)'),
        mpatches.Patch(color='#90A4AE', label='n.s.'),
    ]
    ax1.legend(handles=legend_el1, loc='lower right', fontsize=9)

    # Right panel: model match
    ax2 = fig.add_subplot(gs[1])
    conditions = ['KT', 'MD', 'PF', 'SLP']

    def _get_dir(v1, v2):
        return '+' if v2 > v1 else '-' if v2 < v1 else '='

    jr_matches = []
    wc_matches = []
    for cond_short in conditions:
        drug = COND_TO_DRUG[cond_short]
        ds_cond = [n for n, d in DATASETS.items() if d['condition'] == cond_short]
        emp_s = [_get_dir(kuramoto_results[n]['Awake']['broadband']['sync'],
                          kuramoto_results[n]['Unconscious']['broadband']['sync']) for n in ds_cond]
        emp_m = [_get_dir(kuramoto_results[n]['Awake']['broadband']['meta'],
                          kuramoto_results[n]['Unconscious']['broadband']['meta']) for n in ds_cond]
        s_maj = max(set(emp_s), key=emp_s.count)
        m_maj = max(set(emp_m), key=emp_m.count)
        jr_s = model_dirs['JR'][drug]['sync']; jr_m = model_dirs['JR'][drug]['meta']
        wc_s = model_dirs['WC'][drug]['sync']; wc_m = model_dirs['WC'][drug]['meta']
        jr_matches.append((jr_s == s_maj) + (jr_m == m_maj))
        wc_matches.append((wc_s == s_maj) + (wc_m == m_maj))

    x = np.arange(len(conditions)); w = 0.35
    ax2.bar(x - w/2, jr_matches, w, color='#2196F3', alpha=0.85, label='JR', edgecolor='white')
    ax2.bar(x + w/2, wc_matches, w, color='#F44336', alpha=0.7, label='WC', edgecolor='white')
    ax2.set_xticks(x); ax2.set_xticklabels(conditions, fontsize=11)
    ax2.set_ylabel('Direction Matches (out of 2)')
    ax2.set_ylim(0, 2.5)
    ax2.set_title('Model Validation\n(Direction Match Score)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    for i, (jr, wc) in enumerate(zip(jr_matches, wc_matches)):
        ax2.text(i - w/2, jr + 0.08, f'{jr}/2', ha='center', fontsize=10, fontweight='bold')
        ax2.text(i + w/2, wc + 0.08, f'{wc}/2', ha='center', fontsize=10)

    fig.suptitle('Publication Summary: Empirical Effects + Model Validation', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig7_summary.png', dpi=150, bbox_inches='tight')
    plt.show()


# Figure 8: JR R(t) Dynamics
def plot_jr_dynamics(results_jr):
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    sim_labels = [('Baseline', 'Awake (Baseline)'),
                  ('Ketamine', 'Ketamine (Medium)'),
                  ('Medetomidine', 'Medetomidine (Medium)'),
                  ('Propofol', 'Propofol (Medium)'),
                  ('Sleep', 'Sleep (Medium)')]
    sim_colors = ['#333333', '#FF9800', '#9C27B0', '#009688', '#607D8B']

    for idx, ((drug_name, label), color) in enumerate(zip(sim_labels, sim_colors)):
        ax = axes[idx]
        R_ts = results_jr['Baseline']['R'] if drug_name == 'Baseline' else results_jr[drug_name]['medium']['R']
        t = np.arange(len(R_ts)) / 250
        ax.plot(t, R_ts, color=color, alpha=0.3, lw=0.5)
        win = 500
        if len(R_ts) > win:
            R_smooth = np.convolve(R_ts, np.ones(win) / win, mode='same')
            ax.plot(t, R_smooth, color=color, lw=2)
        sync_val = np.mean(R_ts); meta_val = np.var(R_ts)
        ax.axhline(sync_val, color='k', ls='--', alpha=0.3)
        ax.set_ylabel('R(t)', fontsize=10)
        ax.set_ylim([0, 0.8])
        ax.set_title(f'{label} -- Sync={sync_val:.3f}, Meta={meta_val:.5f}',
                     fontsize=11, fontweight='bold', color=color)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('JR Model: R(t) Dynamics under Different Conditions', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{FIGURES_DIR}/fig8_jr_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
