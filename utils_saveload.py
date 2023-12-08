import pickle

def load_hamiltonian_tensors(moltag):
    filename = f'hamiltonians/{moltag}/chem_tensors'
    with open(filename, 'rb') as f:
        obt, tbt = pickle.load(f)
    return obt, tbt

def save_params(params, norms, moltag, methodtag, bonustag):
    filename = f'results/{moltag}/params_{methodtag}_{bonustag}'
    with open(filename, 'wb') as f:
        pickle.dump([params, norms], f)
    return None

def load_params(moltag, methodtag, bonustag):
    filename = f'results/{moltag}/params_{methodtag}_{bonustag}'
    with open(filename, 'rb') as f:
        params, norms = pickle.load(f)
    return params, norms

def output_log_filename(moltag, methodtag, bonustag):
    filename = f'logs/{moltag}/output_{methodtag}_{bonustag}'
    return filename 