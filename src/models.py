import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from .config import DTYPE


class JansenRitNetwork:
    def __init__(self, n_nodes=64, dt=1e-3):
        self.n = n_nodes
        self.dt = dt
        self.A = 3.25; self.B = 22.0
        self.a = 100.0; self.b = 50.0
        self.C1 = 135.0; self.C2 = 108.0; self.C3 = 33.75; self.C4 = 33.75
        self.e0 = 2.5; self.v0 = 6.0; self.r = 0.56
        self.p_mean = 220.0; self.p_sigma = 22.0
        self.G = 2.0
        self.W = self._build_connectivity()
        self.p_heterogeneity = 15.0
        rng = np.random.default_rng(456)
        self.p_offsets = (rng.standard_normal(n_nodes) * self.p_heterogeneity).astype(DTYPE)

    def _build_connectivity(self):
        n = self.n
        W = np.zeros((n, n))
        k = max(4, n // 8)
        for i in range(n):
            for j in range(1, k + 1):
                W[i, (i + j) % n] = 1.0
                W[i, (i - j) % n] = 1.0
        rng = np.random.default_rng(42)
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0 and rng.random() < 0.15:
                    new_j = rng.integers(0, n)
                    if new_j != i and W[i, new_j] == 0:
                        W[i, j] = 0
                        W[i, new_j] = 1.0
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return W / row_sums

    def sigmoid(self, v):
        return 2.0 * self.e0 / (1.0 + np.exp(self.r * (self.v0 - v)))

    def simulate(self, T=60.0, transient=5.0, fs_out=250):
        dt, n = self.dt, self.n
        n_steps = int(T / dt)
        ds_factor = max(1, int(1.0 / (fs_out * dt)))
        rng = np.random.default_rng(0)
        y0 = rng.uniform(0, 0.5, n)
        y1 = rng.uniform(0, 0.5, n)
        y2 = rng.uniform(0, 0.5, n)
        dy0, dy1, dy2 = np.zeros(n), np.zeros(n), np.zeros(n)
        p_mean_nodes = self.p_mean + self.p_offsets
        trans_steps = int(transient / dt)
        n_out = (n_steps - trans_steps) // ds_factor
        eeg_out = np.empty((n, n_out), dtype=DTYPE)
        out_idx = 0
        for step in range(n_steps):
            eeg = y1 - y2
            S_eeg = self.sigmoid(eeg)
            coupling = self.G * (self.W @ S_eeg)
            p = p_mean_nodes + self.p_sigma * rng.standard_normal(n)
            ddy0 = self.A * self.a * S_eeg - 2 * self.a * dy0 - self.a**2 * y0
            ddy1 = (self.A * self.a * (p + coupling + self.C2 * self.sigmoid(self.C1 * y0))
                    - 2 * self.a * dy1 - self.a**2 * y1)
            ddy2 = (self.B * self.b * self.C4 * self.sigmoid(self.C3 * y0)
                    - 2 * self.b * dy2 - self.b**2 * y2)
            dy0 += ddy0 * dt; dy1 += ddy1 * dt; dy2 += ddy2 * dt
            y0 += dy0 * dt; y1 += dy1 * dt; y2 += dy2 * dt
            if step >= trans_steps and (step - trans_steps) % ds_factor == 0 and out_idx < n_out:
                eeg_out[:, out_idx] = eeg.astype(DTYPE)
                out_idx += 1
        return eeg_out


class WilsonCowanNetwork:
    def __init__(self, n_nodes=64, dt=0.5e-3):
        self.n = n_nodes
        self.dt = dt
        self.tau_e, self.tau_i = 8e-3, 16e-3
        self.c_ee, self.c_ei, self.c_ie, self.c_ii = 16.0, 12.0, 15.0, 3.0
        self.G = 0.08
        self.W = self._build_connectivity()
        self.P, self.Q = 1.25, 0.0
        self.sigma = 0.6
        rng = np.random.default_rng(123)
        self.P_offsets = (rng.standard_normal(n_nodes) * 0.15).astype(DTYPE)
        self.a_e, self.theta_e = 1.3, 4.0
        self.a_i, self.theta_i = 2.0, 3.7

    def _build_connectivity(self):
        n = self.n
        W = np.zeros((n, n))
        k = max(4, n // 8)
        for i in range(n):
            for j in range(1, k + 1):
                W[i, (i + j) % n] = 1.0; W[i, (i - j) % n] = 1.0
        rng = np.random.default_rng(42)
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0 and rng.random() < 0.15:
                    new_j = rng.integers(0, n)
                    if new_j != i and W[i, new_j] == 0:
                        W[i, j] = 0; W[i, new_j] = 1.0
        rs = W.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        return W / rs

    def simulate(self, T=60.0, transient=5.0, fs_out=250):
        dt, n = self.dt, self.n
        n_steps = int(T / dt)
        ds = int(1.0 / (fs_out * dt))
        rng = np.random.default_rng(0)
        E = rng.uniform(0, 0.5, n); I = rng.uniform(0, 0.5, n)
        P_nodes = self.P + self.P_offsets
        trans = int(transient / dt)
        n_out = (n_steps - trans) // ds
        E_out = np.empty((n, n_out), dtype=DTYPE); oi = 0
        sqrt_dt = np.sqrt(dt)
        sig_e = lambda x: 1.0 / (1.0 + np.exp(-self.a_e * (x - self.theta_e)))
        sig_i = lambda x: 1.0 / (1.0 + np.exp(-self.a_i * (x - self.theta_i)))
        for step in range(n_steps):
            coupling = self.G * (self.W @ E)
            noise_e = self.sigma * sqrt_dt * rng.standard_normal(n)
            dE = (-E + sig_e(self.c_ee * E - self.c_ei * I + coupling + P_nodes + noise_e)) / self.tau_e
            dI = (-I + sig_i(self.c_ie * E - self.c_ii * I + self.Q)) / self.tau_i
            E = E + dE * dt; I = I + dI * dt
            if step >= trans and (step - trans) % ds == 0 and oi < n_out:
                E_out[:, oi] = E.astype(DTYPE); oi += 1
        return E_out


def model_kuramoto(data, fs=250, band=(1, 50)):
    nyq = fs / 2
    sos = butter(4, [band[0] / nyq, band[1] / nyq], btype='band', output='sos')
    data_filt = sosfiltfilt(sos, data, axis=1).astype(DTYPE)
    n_ch, n_t = data_filt.shape
    phases = np.empty((n_ch, n_t), dtype=DTYPE)
    for i in range(n_ch):
        phases[i] = np.angle(hilbert(data_filt[i])).astype(DTYPE)
    R = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return {'sync': float(np.mean(R)), 'meta': float(np.var(R)), 'R': R}


def simulate_jr_condition(param_overrides, T=60.0, label=''):
    jr_sim = JansenRitNetwork(n_nodes=64)
    for key, val in param_overrides.items():
        setattr(jr_sim, key, val)
    eeg = jr_sim.simulate(T=T, transient=5.0, fs_out=250)
    m = model_kuramoto(eeg, fs=250, band=(1, 50))
    del eeg
    print(f"  {label:35s} | Sync={m['sync']:.4f}, Meta={m['meta']:.6f}")
    return m


def simulate_wc_condition(param_overrides, T=60.0, label=''):
    wc_sim = WilsonCowanNetwork(n_nodes=64)
    for key, val in param_overrides.items():
        setattr(wc_sim, key, val)
    E = wc_sim.simulate(T=T, transient=5.0, fs_out=250)
    m = model_kuramoto(E, fs=250, band=(1, 50))
    del E
    print(f"  {label:35s} | Sync={m['sync']:.4f}, Meta={m['meta']:.6f}")
    return m


def run_jr_simulations(jr_params):
    """Run all JR simulations: baseline + 4 conditions x 3 depths."""
    import gc
    results_jr = {}
    print('=== Jansen-Rit Network Simulations ===\n--- Baseline (Awake) ---')
    results_jr['Baseline'] = simulate_jr_condition({}, T=60.0, label='Awake baseline')

    for drug, levels in jr_params.items():
        print(f'\n--- {drug} ---')
        results_jr[drug] = {}
        for level, params in levels.items():
            results_jr[drug][level] = simulate_jr_condition(params, T=60.0, label=f'{drug} [{level}]')

    gc.collect()
    return results_jr


def run_wc_simulations(wc_params):
    """Run all WC simulations: baseline + 4 conditions (medium only)."""
    import gc
    results_wc = {}
    print('=== Wilson-Cowan (medium depth, for comparison) ===\n--- Baseline ---')
    results_wc['Baseline'] = simulate_wc_condition({}, T=60.0, label='Awake baseline')
    for drug, levels in wc_params.items():
        results_wc[drug] = simulate_wc_condition(levels['medium'], T=60.0, label=f'{drug} [medium]')
    gc.collect()
    return results_wc


def direction_match_analysis(kuramoto_results, results_jr, results_wc, datasets, cond_to_drug):
    """Compare model-predicted directions vs empirical directions."""
    def get_dir(val_aw, val_un):
        return '+' if val_un > val_aw else '-' if val_un < val_aw else '='

    emp_dirs = {}
    for ds_name, ds in datasets.items():
        aw_s = kuramoto_results[ds_name].get('Awake', {}).get('broadband', {}).get('sync', 0)
        un_s = kuramoto_results[ds_name].get('Unconscious', {}).get('broadband', {}).get('sync', 0)
        aw_m = kuramoto_results[ds_name].get('Awake', {}).get('broadband', {}).get('meta', 0)
        un_m = kuramoto_results[ds_name].get('Unconscious', {}).get('broadband', {}).get('meta', 0)
        emp_dirs[ds_name] = {'sync': get_dir(aw_s, un_s), 'meta': get_dir(aw_m, un_m)}

    model_dirs = {'JR': {}, 'WC': {}}
    bl_jr_s, bl_jr_m = results_jr['Baseline']['sync'], results_jr['Baseline']['meta']
    bl_wc_s, bl_wc_m = results_wc['Baseline']['sync'], results_wc['Baseline']['meta']

    for drug in ['Ketamine', 'Medetomidine', 'Propofol', 'Sleep']:
        jr_med = results_jr[drug]['medium']
        model_dirs['JR'][drug] = {'sync': get_dir(bl_jr_s, jr_med['sync']),
                                   'meta': get_dir(bl_jr_m, jr_med['meta'])}
        wc_med = results_wc[drug]
        model_dirs['WC'][drug] = {'sync': get_dir(bl_wc_s, wc_med['sync']),
                                   'meta': get_dir(bl_wc_m, wc_med['meta'])}

    # Print table and compute scores
    print(f"{'='*90}")
    print(f"{'MODEL vs EMPIRICAL DIRECTION MATCH (Broadband Kuramoto)':^90}")
    print(f"{'='*90}")
    print(f"{'Condition':<15} | {'Empirical':^25} | {'JR':^12} {'Match':^7} | {'WC':^12} {'Match':^7}")
    print(f"{'-'*90}")

    jr_score, wc_score, total = 0, 0, 0
    for cond_short, drug in cond_to_drug.items():
        ds_cond = [n for n, d in datasets.items() if d['condition'] == cond_short]
        emp_s = [emp_dirs[n]['sync'] for n in ds_cond]
        emp_m = [emp_dirs[n]['meta'] for n in ds_cond]
        s_maj = max(set(emp_s), key=emp_s.count)
        m_maj = max(set(emp_m), key=emp_m.count)
        jr_s = model_dirs['JR'][drug]['sync']; jr_m = model_dirs['JR'][drug]['meta']
        wc_s = model_dirs['WC'][drug]['sync']; wc_m = model_dirs['WC'][drug]['meta']
        jr_match_s = 'Y' if jr_s == s_maj else 'N'
        jr_match_m = 'Y' if jr_m == m_maj else 'N'
        wc_match_s = 'Y' if wc_s == s_maj else 'N'
        wc_match_m = 'Y' if wc_m == m_maj else 'N'
        jr_score += (jr_s == s_maj) + (jr_m == m_maj)
        wc_score += (wc_s == s_maj) + (wc_m == m_maj)
        total += 2
        emp_str = f"S:{s_maj} M:{m_maj}"
        print(f"{cond_short:<15} | {emp_str:^25} | S:{jr_s} M:{jr_m}  {jr_match_s}/{jr_match_m}   | "
              f"S:{wc_s} M:{wc_m}  {wc_match_s}/{wc_match_m}")

    print(f"\nJR Score: {jr_score}/{total} ({jr_score/total*100:.0f}%)")
    print(f"WC Score: {wc_score}/{total} ({wc_score/total*100:.0f}%)")

    return emp_dirs, model_dirs, jr_score, wc_score, total
