import gzip
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import src.utils as utils

sns.set_theme(style="white")

sns.set_context("paper", rc={"font.size":16, "font_scale":1.25, "axes.titlesize":16,"axes.labelsize":16,
                "xtick.labelsize":16, "ytick.labelsize":16, "legend.fontsize":16, "legend.title_fontsize": 16, 
                            "legend.loc": 'upper center', "alpha":0.2, "figure.dpi":600, 'savefig.dpi':600})

data_path = "/home/franciscoperez/Documents/GitHub/CNN-PELSVAE2/cnn-pels-vae/data/time_series/real/OGLE3_lcs_I_meta_snr5_augmented_folded_trim600_GAIA3_LOG_6PP.npy.gz"
print('Loading from:\n', data_path)
with gzip.open(data_path, 'rb') as f:
    np_data = np.load(f, allow_pickle=True)

print(np_data.item()['meta'])

df_pp = np_data.item()['meta']

id_test, id_train, values_count  = utils.get_ids(n=400000) 

id_test['ID'] = id_test['ID'].str.replace('.dat', '')
id_train['ID'] = id_train['ID'].str.replace('.dat', '')

print(id_test.ID.to_list())

df_pp_test = df_pp[df_pp.OGLE_id.isin(id_test.ID.to_list())]
df_pp_train = df_pp[df_pp.OGLE_id.isin(id_train.ID.to_list())]

star_type = 'DSCT'
pp = ['[Fe/H]_J95','teff_val','Period','abs_Gmag','radius_val','logg']
df_plot = df_pp_test[df_pp_test.Type == star_type][pp]
df_plot.columns = ['[Fe/H]_J95','log_teff_val','log_period','abs_Gmag','log_radius_val','logg']
g = sns.PairGrid(df_plot, diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.5)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
plt.savefig(star_type+'_test.svg', format='svg', bbox_inches='tight')
plt.savefig(star_type+'_test.png', format='png', bbox_inches='tight')
plt.show()


df_plot = df_pp_train[df_pp_train.Type == star_type][pp]
df_plot.columns = ['[Fe/H]_J95','log_teff_val','log_period','abs_Gmag','log_radius_val','logg']
g = sns.PairGrid(df_plot, diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.5)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
plt.savefig(star_type+'_train.svg', format='svg', bbox_inches='tight')
plt.savefig(star_type+'_train.png', format='png', bbox_inches='tight')
plt.show()