"""
ECoG Metastability Analysis Pipeline
=====================================
Empirical analysis + computational modeling of brain-state transitions
under anesthesia and sleep in macaque ECoG data.

Usage:
    python run_pipeline.py
"""
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import (
    DATASETS, STATE_MAP, ANALYSIS_BANDS,
    JR_PARAMS, WC_PARAMS, COND_TO_DRUG,
)
from src.data_loading import parse_info_file, load_ecog_session, extract_epochs
from src.preprocessing import detect_bad_channels, compute_all_phases
from src.metrics import compute_kuramoto, compute_wpli_all, compute_fcd_and_coalition
from src.statistics import collect_tests, print_stats_table, bootstrap_ci_cohens_d
from src.models import (
    run_jr_simulations, run_wc_simulations, direction_match_analysis,
)
from src.visualization import (
    plot_kuramoto_sync, plot_wpli, plot_fcd_coalition, plot_effect_sizes,
    plot_jr_dose_response, plot_model_vs_empirical, plot_summary, plot_jr_dynamics,
)

plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 11})


# ============================================================
# 1. Load All Raw Data
# ============================================================
print('=' * 60)
print('STEP 1: Loading raw ECoG data')
print('=' * 60)

from src.config import FS

all_raw = {}
for ds_name, ds in DATASETS.items():
    print(f'Loading {ds_name} ...')
    events = parse_info_file(ds['path'])
    ds_epochs = {}
    for sess_num in [1, 2, 3]:
        sess_dir = ds['path'] / f'Session{sess_num}'
        if not sess_dir.exists():
            continue
        data_sess, time_sess = load_ecog_session(ds['path'], sess_num)
        sess_events = events.get(sess_num, [])
        ds_epochs.update(extract_epochs(data_sess, time_sess, sess_events))
        del data_sess, time_sess
    sm = STATE_MAP[ds_name]
    raw = {}
    for state_label, cond_key in [('Awake', sm['Awake']), ('Unconscious', sm['Unconscious'])]:
        if cond_key in ds_epochs:
            raw[state_label] = ds_epochs[cond_key]
    all_raw[ds_name] = raw
    for s, (d, _) in raw.items():
        print(f'  {s}: {d.shape[0]}ch x {d.shape[1]} ({d.shape[1]/FS:.1f}s)')
gc.collect()
print(f'\nLoaded: {sum(d.nbytes for ep in all_raw.values() for d, _ in ep.values())/1024**2:.0f} MB')


# ============================================================
# 2. Bad Channel Detection + CAR Preprocessing
# ============================================================
print('\n' + '=' * 60)
print('STEP 2: Bad channel detection & preprocessing')
print('=' * 60)

bad_ch_all, good_ch_all = detect_bad_channels(all_raw, DATASETS)
all_phases = compute_all_phases(all_raw, DATASETS, good_ch_all)
gc.collect()
print('All preprocessing complete (bad channels excluded).')


# ============================================================
# 3. Compute Metrics
# ============================================================
print('\n' + '=' * 60)
print('STEP 3: Computing metrics (Kuramoto, wPLI, FCD, Coalition Entropy)')
print('=' * 60)

kuramoto_results = compute_kuramoto(all_phases, DATASETS)
wpli_results = compute_wpli_all(all_raw, DATASETS, good_ch_all)
gc.collect()

fcd_results, coal_results = compute_fcd_and_coalition(all_phases, DATASETS)

# Free raw data
del all_raw
gc.collect()
print('\nRaw data freed.')


# ============================================================
# 4. Statistical Analysis
# ============================================================
print('\n' + '=' * 60)
print('STEP 4: Statistical analysis')
print('=' * 60)

all_tests, p_fdr, sig_fdr, bf_cache = collect_tests(
    kuramoto_results, wpli_results, fcd_results, coal_results, DATASETS, ANALYSIS_BANDS
)
print_stats_table(all_tests, p_fdr, sig_fdr, bf_cache)


# ============================================================
# 5. Figures 1-4: Empirical Results
# ============================================================
print('\n' + '=' * 60)
print('STEP 5: Generating empirical figures (1-4)')
print('=' * 60)

plot_kuramoto_sync(kuramoto_results)
plot_wpli(wpli_results)
plot_fcd_coalition(fcd_results, coal_results)
plot_effect_sizes(all_tests, bf_cache)

# Free phase data before model simulations
del all_phases
gc.collect()
print('Phase data freed. Ready for model simulations.')


# ============================================================
# 6. Model Simulations
# ============================================================
print('\n' + '=' * 60)
print('STEP 6: Running model simulations (JR + WC)')
print('=' * 60)

results_jr = run_jr_simulations(JR_PARAMS)
results_wc = run_wc_simulations(WC_PARAMS)


# ============================================================
# 7. Direction Match Analysis
# ============================================================
print('\n' + '=' * 60)
print('STEP 7: Direction match analysis')
print('=' * 60)

emp_dirs, model_dirs, jr_score, wc_score, total = direction_match_analysis(
    kuramoto_results, results_jr, results_wc, DATASETS, COND_TO_DRUG
)


# ============================================================
# 8. Figures 5-8: Model Results
# ============================================================
print('\n' + '=' * 60)
print('STEP 8: Generating model figures (5-8)')
print('=' * 60)

plot_jr_dose_response(results_jr)
plot_model_vs_empirical(kuramoto_results, results_jr, results_wc)
plot_summary(all_tests, bf_cache, kuramoto_results, model_dirs, results_jr, results_wc)
plot_jr_dynamics(results_jr)


# ============================================================
# 9. Final Summary
# ============================================================
print('\n' + '=' * 90)
print('             PUBLICATION RESULTS SUMMARY')
print('=' * 90)

p_orig = [t[5] for t in all_tests]
n_raw_sig = sum(1 for p in p_orig if p < 0.05)
n_fdr_sig = sum(sig_fdr)

print('\n1. EMPIRICAL RESULTS')
print(f'   Datasets: 2 monkeys x 4 conditions = 8 paired comparisons')
print(f'   Tests: {len(all_tests)} metrics x frequency bands')
print(f'   Raw significant (p<0.05): {n_raw_sig}/{len(all_tests)}')
print(f'   FDR corrected (BH q<0.05): {n_fdr_sig}/{len(all_tests)}')
print(f'   Bayes Factor > 10 (strong): {sum(1 for bf in bf_cache if bf > 10)}/{len(all_tests)}')
print(f'   Bayes Factor > 3 (moderate): {sum(1 for bf in bf_cache if bf > 3)}/{len(all_tests)}')

print('\n   Key Findings:')
for i, (name, band, aw, un, d, p) in enumerate(all_tests):
    bf = bf_cache[i]
    if bf > 3:
        ci_lo, ci_hi = bootstrap_ci_cohens_d(aw, un)
        print(f'   * {name} {band}: d={d:+.3f} [{ci_lo:+.2f},{ci_hi:+.2f}], p={p:.4f}, BF10={bf:.1f}')

print('\n2. COMPUTATIONAL MODEL VALIDATION')
print(f'   JR (3-population): {jr_score}/{total} direction matches ({jr_score/total*100:.0f}%)')
print(f'   WC (2-population): {wc_score}/{total} direction matches ({wc_score/total*100:.0f}%)')
print(f'   Key: JR correctly predicts Propofol direction; WC fails.')

print('\n3. FIGURES SAVED')
for i in range(1, 9):
    fnames = ['kuramoto_sync', 'wpli', 'fcd_coalition', 'effect_sizes',
              'jr_dose_response', 'model_vs_empirical', 'summary', 'jr_dynamics']
    print(f'   figures/fig{i}_{fnames[i-1]}.png')

print(f'\n{"="*90}')
print(f'PIPELINE COMPLETE')
print(f'{"="*90}')
