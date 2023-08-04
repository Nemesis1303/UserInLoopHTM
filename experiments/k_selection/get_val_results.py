import argparse
import pathlib
import sys
import pandas as pd
import scipy.sparse as sparse
from matplotlib import pyplot as plt

sys.path.append('../..')

def read_values_from_file(filename):
    with open(filename, 'r') as file:
        values = [line.strip() for line in file]
    return values

def get_results(path_models:str, corpus_name:str):
    
    path_models = pathlib.Path(path_models)
    
    values = []
    for entry in path_models.iterdir():
        if not entry.as_posix().endswith("old") and entry.is_dir() and entry.joinpath("TMmodel/fold_config.txt").is_file():
            TMfolder = entry.joinpath("TMmodel")
            
            values_file = read_values_from_file(TMfolder.joinpath('fold_config.txt'))
                                                
            ntopics = int(values_file[0])
            alpha = float(values_file[1])
            opt_int = int(values_file[2])
            fold = int(entry.as_posix().split("ntopics_")[1].split("_alpha_")[1].split("_optint_")[1].split("_fold_")[1])


            # Read thetas
            thetas = sparse.load_npz(TMfolder.joinpath('thetas.npz'))
            disp_perc = 100 * ((thetas.shape[0] * thetas.shape[1]) - len(thetas.data)) / (thetas.shape[0] * thetas.shape[1])
            #disp_perc = 100 * thetas.count_nonzero() / (thetas.shape[0] * thetas.shape[1])
            
            cohr = float(read_values_from_file(TMfolder.joinpath('fold_config.txt'))[3])

            values.append([ntopics, alpha, opt_int, fold, disp_perc, cohr])

    df_results = pd.DataFrame(values, columns=['ntopics', 'alpha', 'opt_int', 'fold', 'disp_perc', 'cohr'])
    df_results.to_csv(pathlib.Path(path_models).joinpath("val_results.csv"))    

    # Define style for figures
    fig_width = 6.9  # inches
    fig_height = 3.5  # inches
    fig_dpi = 600

    plt.rcParams.update({
        'figure.figsize': (fig_width, fig_height),
        'figure.dpi': fig_dpi,

        # Fonts
        'font.size': 12,

        # Axes
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.linewidth': 1,
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.linewidth': 1,
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
        'lines.linewidth': 4.0,
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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting 'cohr' on the left y-axis
    ax1.errorbar(
        grouped_df['ntopics'],
        grouped_df['cohr_mean'],
        yerr=grouped_df['cohr_std'],
        fmt='x-',
        ecolor='gray',
        capsize=2,
        lw=1,
        color='#36AE7C',
        label='Cohr')

    ax1.set_xlabel('Nr topics')
    ax1.set_ylabel('NPMI coherence', color='#36AE7C')
    ax1.grid(True)

    # Creating a twin axis on the right side for 'disp_perc'
    ax2 = ax1.twinx()
    ax2.errorbar(grouped_df['ntopics'],
                grouped_df['disp_perc_mean'],
                yerr=grouped_df['disp_perc_std'], 
                fmt='x-',
                ecolor='gray',
                capsize=2,
                lw=1,
                color='#187498',
                label='Disp')
    ax2.set_ylabel('Thetas dispersion percentage', color='#187498')

    # Title and legend
    plt.title(f'NPMI coherence and thetas dispersion percentage per nr topics in {corpus_name} dataset')
    #plt.legend()
    plt.savefig(pathlib.Path(path_models).joinpath("plot3.png"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_val_mallet",
                        help="Path to the trained models.")
    parser.add_argument('--corpus', type=str,
                        default="Cordis",
                        help="Name the corpus used for the training.")
    args = parser.parse_args()
    
    get_results(args.path_models, args.corpus)
    
if __name__ == '__main__':
    main()