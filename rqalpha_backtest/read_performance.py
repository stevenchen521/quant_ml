import pandas as pd
# dict = pd.read_pickle('bolling_bond.pkl')

# print (dict)

# import click
# from rqalpha import cli
# @cli.command()
# @click.argument('result_pickle_file_path', type=click.Path(exists=True), required=True)
# @click.option('--show/--hide', 'show', default=True)
# @click.option('--plot-save', 'plot_save_file', default=None, type=click.Path(), help="save plot result to file")
# def plot(result_pickle_file_path, show, plot_save_file):
#     """
#     [sys_analyser] draw result DataFrame
#     """
import pandas as pd
from rqalpha.mod.rqalpha_mod_sys_analyser.plot import plot_result

# result_dict = pd.read_pickle('Portfolio_RSI.pkl')
pd.read_pickle('bolling_bond.pkl').plots.to_excel('C:\\Users\wilsonZhang\Desktop\\bolling_bond_plot.xlsx')
# print (result_dict)
# plot_result(result_dict)