import numpy as np
from scipy import signal

from .config import FS, FS_DOWN, DTYPE, FREQ_BANDS, ANALYSIS_BANDS


def compute_kuramoto(all_phases, datasets):
    """Compute Kuramoto order parameter: synchrony and metastability."""
    kuramoto_results = {}
    for ds_name in datasets:
        kuramoto_results[ds_name] = {}
        for state in ['Awake', 'Unconscious']:
            if state not in all_phases[ds_name]:
                continue
            kuramoto_results[ds_name][state] = {}
            for band in ANALYSIS_BANDS:
                ph = all_phases[ds_name][state][band]
                R = np.abs(np.mean(np.exp(1j * ph), axis=0))
                kuramoto_results[ds_name][state][band] = {
                    'sync': float(np.mean(R)),
                    'meta': float(np.var(R)),
                }
    return kuramoto_results


def compute_wpli_mean(data, fs, band, nperseg=512):
    lo, hi = FREQ_BANDS[band]
    n_ch = data.shape[0]
    wpli_vals = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, Pxy = signal.csd(data[i], data[j], fs=fs, nperseg=nperseg)
            mask = (f >= lo) & (f <= hi)
            imag_csd = np.imag(Pxy[mask])
            num = np.abs(np.mean(np.abs(imag_csd) * np.sign(imag_csd)))
            den = np.mean(np.abs(imag_csd))
            wpli_vals.append(num / den if den > 1e-12 else 0.0)
    return float(np.mean(wpli_vals))


def compute_wpli_all(all_raw, datasets, good_ch_all, duration=120):
    """Compute wPLI for all datasets."""
    wpli_results = {}
    for ds_name in datasets:
        good = good_ch_all[ds_name]
        wpli_results[ds_name] = {}
        for state in ['Awake', 'Unconscious']:
            if state not in all_raw[ds_name]:
                continue
            data, _ = all_raw[ds_name][state]
            n_samp = min(data.shape[1], duration * FS)
            d = signal.decimate(data[:, :n_samp], int(FS / FS_DOWN), axis=1).astype(DTYPE)
            car_mean = d[good].mean(axis=0, keepdims=True)
            d = (d - car_mean)[good]
            wpli_results[ds_name][state] = {}
            for band in ANALYSIS_BANDS:
                wpli_results[ds_name][state][band] = compute_wpli_mean(d, FS_DOWN, band)
            del d
        print(f'{ds_name}: done')
    return wpli_results


def compute_fcd(phases, window_sec=5, step_sec=3, fs=FS_DOWN):
    n_ch, n_t = phases.shape
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    starts = list(range(0, n_t - win, step))
    triu_idx = np.triu_indices(n_ch, k=1)
    fc_vecs = np.empty((len(starts), len(triu_idx[0])), dtype=DTYPE)
    exp_cache = np.exp(1j * phases.astype(np.float32))
    for w, s in enumerate(starts):
        win_data = exp_cache[:, s:s + win]
        plv = np.abs(win_data @ win_data.conj().T / win)
        fc_vecs[w] = plv[triu_idx].real
    del exp_cache
    fc_z = fc_vecs - fc_vecs.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(fc_z, axis=1, keepdims=True)
    norms[norms == 0] = 1
    fcd = (fc_z / norms @ (fc_z / norms).T).astype(DTYPE)
    return fcd


def compute_coalition_entropy(phases, window_sec=2, step_sec=1, fs=FS_DOWN):
    n_ch, n_t = phases.shape
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    starts = list(range(0, n_t - win, step))
    coalitions = np.empty((len(starts), n_ch), dtype=np.uint8)
    for w, s in enumerate(starts):
        pw = phases[:, s:s + win].astype(DTYPE)
        cos_ph = np.cos(pw)
        sin_ph = np.sin(pw)
        coh = (cos_ph @ cos_ph.T + sin_ph @ sin_ph.T) / pw.shape[1]
        _, eigvecs = np.linalg.eigh(coh)
        coalitions[w] = (eigvecs[:, -1] > 0).astype(np.uint8)
    packed = np.packbits(coalitions, axis=1)
    patterns = [row.tobytes() for row in packed]
    unique, counts = np.unique(patterns, return_counts=True)
    probs = counts / counts.sum()
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = np.log2(len(coalitions))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_fcd_and_coalition(all_phases, datasets, primary_band='delta'):
    """Compute FCD variance and coalition entropy for all datasets."""
    fcd_results = {}
    coal_results = {}
    for ds_name in datasets:
        fcd_results[ds_name] = {}
        coal_results[ds_name] = {}
        for state in ['Awake', 'Unconscious']:
            if state not in all_phases[ds_name]:
                continue
            ph = all_phases[ds_name][state][primary_band]
            fcd_mat = compute_fcd(ph)
            triu = fcd_mat[np.triu_indices_from(fcd_mat, k=1)]
            fcd_results[ds_name][state] = float(np.var(triu))
            coal_results[ds_name][state] = compute_coalition_entropy(ph)
            print(f'{ds_name}/{state}: FCD_Var={fcd_results[ds_name][state]:.6f}, '
                  f'H_norm={coal_results[ds_name][state]:.4f}')
    return fcd_results, coal_results
