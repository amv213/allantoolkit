import yaml
import pathlib

# load the config file with reference tables and extract tables
path_to_tables = pathlib.Path(__file__).parent / 'tables.yml'

with open(path_to_tables, 'rb') as f:
    tables = yaml.load(f, Loader=yaml.SafeLoader)

STOP_RATIOS = tables.get('stop_ratios')
D_ORDER = tables.get('d_order')
