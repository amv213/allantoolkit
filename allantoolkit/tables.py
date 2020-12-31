import yaml
import pathlib

# load the config file with reference tables and extract tables
path_to_tables = pathlib.Path(__file__).parent / 'tables.yml'

with open(path_to_tables, 'rb') as f:
    tables = yaml.load(f, Loader=yaml.SafeLoader)

# DEV PARAMS
STOP_RATIOS = tables.get('stop_ratios')
D_ORDER = tables.get('d_order')

# NOISE
ALPHA_TO_MU = tables.get('alpha_to_mu')
ALPHA_TO_NAMES = tables.get('alpha_to_names')

# BIAS FACTORS
BIAS_TOTVAR = tables.get('bias_totvar')
BIAS_MTOTVAR = tables.get('bias_mtotvar')
BIAS_HTOTVAR = tables.get('bias_htotvar')
BIAS_THEO1 = tables.get('bias_theo1')
BIAS_THEO1_FIXED = tables.get('bias_theo1_fixed')


# CONFIDENCE INTERVALS
KN_NOISE_FACTOR = tables.get('kn_noise_factor')
GREENHALL_TABLE1 = tables.get('greenhall_table1')
GREENHALL_TABLE2 = tables.get('greenhall_table2')
GREENHALL_TABLE3 = tables.get('greenhall_table3')
TOTVAR_EDF_COEFFICIENTS = tables.get('totvar_edf_coefficients')
MTOTVAR_EDF_COEFFICIENTS = tables.get('mtotvar_edf_coefficients')
HTOTVAR_EDF_COEFFICIENTS = tables.get('htotvar_edf_coefficients')
ABRAMOWITZ_COEFFICIENTS = tables.get('abramowitz_coefficients')