import numpy as np
from scipy.stats import t as t_dist
from scipy.integrate import quad


def permutation_test(aw_vals, un_vals, n_perm=10000):
    diffs = np.array(un_vals) - np.array(aw_vals)
    obs_diff = np.mean(diffs)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        if abs(np.mean(diffs * signs)) >= abs(obs_diff):
            count += 1
    return count / n_perm, obs_diff


def cohens_d(aw_vals, un_vals):
    diffs = np.array(un_vals) - np.array(aw_vals)
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))


def benjamini_hochberg(p_values, alpha=0.05):
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adj_p = np.empty(n)
    adj_p[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adj_p[sorted_idx[i]] = min(adj_p[sorted_idx[i + 1]], sorted_p[i] * n / (i + 1))
    return adj_p.tolist(), [ap < alpha for ap in adj_p]


def bootstrap_ci_cohens_d(aw_vals, un_vals, n_boot=5000, ci=0.95):
    diffs = np.array(un_vals) - np.array(aw_vals)
    n = len(diffs)
    rng = np.random.default_rng(42)
    boot_d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bd = diffs[idx]
        std = np.std(bd, ddof=1)
        boot_d.append(np.mean(bd) / (std + 1e-12))
    lo = np.percentile(boot_d, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_d, (1 + ci) / 2 * 100)
    return lo, hi


def bayes_factor_paired(aw_vals, un_vals, r=0.707):
    """JZS Bayes Factor for paired samples (Rouder et al., 2009)."""
    diffs = np.array(un_vals) - np.array(aw_vals)
    n = len(diffs)
    t_stat = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(n) + 1e-12)
    df = n - 1

    def integrand(g):
        se_adj = np.sqrt(1 + n * g)
        return (t_dist.pdf(t_stat / se_adj, df) / se_adj
                * (2 / np.pi / r**2) * np.exp(-g / (2 * r**2)) * (g > 0))

    marginal_h1, _ = quad(integrand, 0, np.inf, limit=100)
    marginal_h0 = t_dist.pdf(t_stat, df)
    if marginal_h1 < 1e-20:
        return 0.01
    return marginal_h1 / marginal_h0


def collect_tests(kuramoto_results, wpli_results, fcd_results, coal_results, datasets, analysis_bands):
    """Collect all test data and run statistical analysis."""
    all_tests = []

    for band in analysis_bands:
        aw_sync, un_sync, aw_meta, un_meta = [], [], [], []
        for ds_name in datasets:
            ka = kuramoto_results[ds_name].get('Awake', {}).get(band, {})
            ku = kuramoto_results[ds_name].get('Unconscious', {}).get(band, {})
            if ka and ku:
                aw_sync.append(ka['sync']); un_sync.append(ku['sync'])
                aw_meta.append(ka['meta']); un_meta.append(ku['meta'])
        if aw_sync:
            p_s, _ = permutation_test(aw_sync, un_sync)
            d_s = cohens_d(aw_sync, un_sync)
            all_tests.append(('Kur_Sync', band, aw_sync, un_sync, d_s, p_s))
            p_m, _ = permutation_test(aw_meta, un_meta)
            d_m = cohens_d(aw_meta, un_meta)
            all_tests.append(('Kur_Meta', band, aw_meta, un_meta, d_m, p_m))

    for band in analysis_bands:
        aw_w, un_w = [], []
        for ds_name in datasets:
            aw_w.append(wpli_results[ds_name].get('Awake', {}).get(band, 0))
            un_w.append(wpli_results[ds_name].get('Unconscious', {}).get(band, 0))
        p_w, _ = permutation_test(aw_w, un_w)
        d_w = cohens_d(aw_w, un_w)
        all_tests.append(('wPLI', band, aw_w, un_w, d_w, p_w))

    aw_f, un_f, aw_h, un_h = [], [], [], []
    for ds_name in datasets:
        aw_f.append(fcd_results[ds_name].get('Awake', 0))
        un_f.append(fcd_results[ds_name].get('Unconscious', 0))
        aw_h.append(coal_results[ds_name].get('Awake', 0))
        un_h.append(coal_results[ds_name].get('Unconscious', 0))
    p_f, _ = permutation_test(aw_f, un_f); d_f = cohens_d(aw_f, un_f)
    all_tests.append(('FCD_Var', 'delta', aw_f, un_f, d_f, p_f))
    p_h, _ = permutation_test(aw_h, un_h); d_h = cohens_d(aw_h, un_h)
    all_tests.append(('H_norm', 'delta', aw_h, un_h, d_h, p_h))

    # FDR correction
    p_orig = [t[5] for t in all_tests]
    p_fdr, sig_fdr = benjamini_hochberg(p_orig)

    # Bayes factors
    bf_cache = []
    for name, band, aw, un, d, p in all_tests:
        bf_cache.append(bayes_factor_paired(aw, un))

    return all_tests, p_fdr, sig_fdr, bf_cache


def print_stats_table(all_tests, p_fdr, sig_fdr, bf_cache):
    """Print full results table."""
    print(f'{"Metric":<12s} {"Band":<11s} | {"d":>7s} {"95% CI":>16s} | '
          f'{"p_raw":>7s} {"p_FDR":>7s} {"FDR":>4s} | {"BF10":>7s} {"Evidence":>12s}')
    print('-' * 100)
    for i, (name, band, aw, un, d, p) in enumerate(all_tests):
        ci_lo, ci_hi = bootstrap_ci_cohens_d(aw, un)
        bf = bf_cache[i]
        fdr_sig = '*' if sig_fdr[i] else ''
        if bf > 10: ev = 'Strong H1'
        elif bf > 3: ev = 'Moderate H1'
        elif bf > 1: ev = 'Anecdotal H1'
        elif bf > 1/3: ev = 'Anecdotal H0'
        elif bf > 1/10: ev = 'Moderate H0'
        else: ev = 'Strong H0'
        raw_sig = '*' if p < 0.05 else '(+)' if p < 0.1 else ''
        print(f'{name:<12s} {band:<11s} | {d:>+7.3f} [{ci_lo:>+6.2f},{ci_hi:>+6.2f}] | '
              f'{p:>7.4f}{raw_sig:<3s} {p_fdr[i]:>7.4f} {fdr_sig:>4s} | {bf:>7.3f} {ev:>12s}')

    p_orig = [t[5] for t in all_tests]
    n_raw_sig = sum(1 for p in p_orig if p < 0.05)
    n_fdr_sig = sum(sig_fdr)
    n_bf_mod = sum(1 for bf in bf_cache if bf > 3)
    print(f'\nSummary: {len(all_tests)} tests | Raw p<0.05: {n_raw_sig} | '
          f'FDR p<0.05: {n_fdr_sig} | BF>3: {n_bf_mod}')
