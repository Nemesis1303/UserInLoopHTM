import pathlib
import sys
import pandas as pd
import scipy.sparse as sparse
from matplotlib import pyplot as plt

sys.path.append('../..')
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel
from src.utils.misc import mallet_corpus_to_df

path_models = "/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_val_mallet"

path_models = pathlib.Path(path_models)

def read_values_from_file(filename):
    with open(filename, 'r') as file:
        values = [line.strip() for line in file]
    return values

values = []
for entry in path_models.iterdir():
    TMfolder = entry.joinpath("TMmodel")

    ntopics = int(entry.as_posix().split("ntopics_")[1].split("_alpha")[0])
    alpha = float(entry.as_posix().split("ntopics_")[1].split("_alpha_")[1].split("_optint_")[0])
    opt_int = int(entry.as_posix().split("ntopics_")[1].split("_alpha_")[1].split("_optint_")[1].split("_fold_")[0])
    fold = int(entry.as_posix().split("ntopics_")[1].split("_alpha_")[1].split("_optint_")[1].split("_fold_")[1])


    # Read thetas
    thetas = sparse.load_npz(TMfolder.joinpath('thetas.npz'))
    disp_perc = 100 * thetas.count_nonzero() / (thetas.shape[0] * thetas.shape[1])
    
    cohr = float(read_values_from_file(TMfolder.joinpath('fold_config.txt'))[3])

    values.append([ntopics, alpha, opt_int, fold, disp_perc, cohr])

df_results = pd.DataFrame(values, columns=['ntopics', 'alpha', 'opt_int', 'fold', 'disp_perc', 'cohr'])
df_results.to_csv(pathlib.Path(path_models).joinpath("val_results.csv"))    

# Define style for figures
fig_width = 6.9  # inches
fig_height = 3.5  # inches
fig_dpi = 300

plt.rcParams.update({
    'figure.figsize': (fig_width, fig_height),
    'figure.dpi': fig_dpi,

    # Fonts
    'font.size': 10,

    # Axes
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.linewidth': 0.5,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'grid.color': 'gray',

    # Legend
    'legend.fontsize': 8,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.fancybox': False,
    'legend.edgecolor': 'gray',
    'legend.facecolor': 'white',
    'legend.borderaxespad': 0.5,
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.5,

    # Lines
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# Calculate the mean and standard deviation of cohr and disp_perc per ntopics
grouped_df = df_results.groupby('ntopics').agg({'cohr': ['mean', 'std'], 'disp_perc': ['mean', 'std']}).reset_index()

# Flatten the column names for easier access
grouped_df.columns = ['ntopics', 'cohr_mean', 'cohr_std', 'disp_perc_mean', 'disp_perc_std']

# Plotting
plt.figure(figsize=(10, 6))

# Error bars represent one standard deviation above and below the mean
plt.errorbar(grouped_df['ntopics'], grouped_df['cohr_mean'], yerr=grouped_df['cohr_std'], label='Cohr')
plt.errorbar(grouped_df['ntopics'], grouped_df['disp_perc_mean'], yerr=grouped_df['disp_perc_std'], label='Disp_Perc')

plt.xlabel('ntopics')
plt.ylabel('Mean Value')
plt.title('Mean and Standard Deviation of Cohr and Disp_Perc per ntopics')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(pathlib.Path(path_models).joinpath("plot3.png"))