import numpy as np
from pathlib import Path

DTYPE = np.float32
FS = 1000
FS_DOWN = 250
N_CH = 64
DATA_DIR = Path('data')
CHANNELS_TO_LOAD = np.linspace(1, 128, N_CH, dtype=int).tolist()

FREQ_BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'broadband': (1, 50),
}
ANALYSIS_BANDS = ['delta', 'theta', 'alpha', 'beta', 'broadband']

COND_COLORS = {'KT': '#FF9800', 'MD': '#9C27B0', 'PF': '#009688', 'SLP': '#607D8B'}
MONKEY_MARKERS = {'George': 'o', 'Chibi': 's'}
COND_TO_DRUG = {'KT': 'Ketamine', 'MD': 'Medetomidine', 'PF': 'Propofol', 'SLP': 'Sleep'}

DATASETS = {
    'George_KT':  {'path': DATA_DIR / '20120810KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'KT'},
    'George_MD':  {'path': DATA_DIR / '20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'MD'},
    'George_PF':  {'path': DATA_DIR / '20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'PF'},
    'George_SLP': {'path': DATA_DIR / '20120712SLP_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128', 'monkey': 'George', 'condition': 'SLP'},
    'Chibi_KT':   {'path': DATA_DIR / '20120813KT_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'KT'},
    'Chibi_MD':   {'path': DATA_DIR / '20120809MD_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'MD'},
    'Chibi_PF':   {'path': DATA_DIR / '20120802PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'PF'},
    'Chibi_SLP':  {'path': DATA_DIR / '20120717SLP_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128', 'monkey': 'Chibi', 'condition': 'SLP'},
}

STATE_MAP = {}
for name, ds in DATASETS.items():
    if ds['condition'] == 'SLP':
        STATE_MAP[name] = {'Awake': 'AwakeEyesOpened', 'Unconscious': 'Sleeping'}
    else:
        STATE_MAP[name] = {'Awake': 'AwakeEyesOpened', 'Unconscious': 'Anesthetized'}

JR_PARAMS = {
    'Ketamine': {
        'light':  {'G': 2.3, 'p_sigma': 25.0},
        'medium': {'G': 2.5, 'p_sigma': 28.0},
        'deep':   {'G': 3.0, 'p_sigma': 30.0},
    },
    'Medetomidine': {
        'light':  {'p_mean': 200.0, 'p_sigma': 18.0, 'G': 1.5},
        'medium': {'p_mean': 180.0, 'p_sigma': 14.0, 'G': 1.0},
        'deep':   {'p_mean': 160.0, 'p_sigma': 10.0, 'G': 0.5},
    },
    'Propofol': {
        'light':  {'B': 28.0, 'b': 40.0, 'G': 1.5},
        'medium': {'B': 35.0, 'b': 30.0, 'G': 1.0},
        'deep':   {'B': 42.0, 'b': 25.0, 'G': 0.5},
    },
    'Sleep': {
        'light':  {'p_mean': 210.0, 'G': 1.8},
        'medium': {'p_mean': 200.0, 'G': 1.6},
        'deep':   {'p_mean': 190.0, 'G': 1.4},
    },
}

WC_PARAMS = {
    'Ketamine':      {'medium': {'c_ei': 10.0, 'sigma': 0.80, 'G': 0.12}},
    'Medetomidine':  {'medium': {'c_ee': 12.0, 'c_ei': 14.0, 'P': 0.85, 'G': 0.03}},
    'Propofol':      {'medium': {'c_ei': 15.0, 'c_ii': 6.0, 'tau_i': 24e-3, 'G': 0.03}},
    'Sleep':         {'medium': {'P': 1.00, 'G': 0.06}},
}
