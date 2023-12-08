"""
Purpose : do GFRO decomposition with onebody purifications until np.sqrt(np.sum(Htbt * Htbt)) < 1e-6

CLA     : python main_onebody.py (moltag) (bonustag) 
"""

import numpy as np
import time

import utils_gfro as gfro
import utils_onebody as onebody

from utils_saveload import (
    load_hamiltonian_tensors,
    save_params,
    load_params,
    output_log_filename
)

