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
