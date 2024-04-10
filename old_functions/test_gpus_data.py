import torch
import gzip 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_gpus():
    print(torch.cuda.is_available())
    print(torch.__version__)

def load_data_to_train(): 
    data_path = ('data/time_series/real/OGLE3_lcs_I_meta_snr5_augmented_folded_trim600.npy.gz')
    print('Loading from:\n', data_path)
    with gzip.open(data_path, 'rb') as f:
        np_data = np.load(f, allow_pickle=True)
    return np_data.item()['meta'], np_data.item()['lcs'], np_data

def save_data(meta, lcs, data):
    data_path = 'data/time_series/real/OGLE3_lcs_I_meta_snr5_augmented_folded_trim600_GAIA3_6PP.npy.gz'
    print('Saving to:\n', data_path)
    data.item()['meta'] = meta
    with gzip.open(data_path, 'wb') as f:
        np.save(f, data)
    
    meta.to_csv('/home/franciscoperez/Documents/GitHub/CNN-PELSVAE2/cnn-pels-vae/data/metadata_updated_0823.csv')
    
    print('Data saved successfully.')

def load_new_validated_pp():
    pp_path = 'data/inter/Validated_OGLExGaiaDR3.csv'
    df = pd.read_csv(pp_path)
    return df

def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]

def compare_frequency(s1, s2, s3, feature = "Period", clean = True): 
    
    if clean:
        s1 = remove_outliers(s1)
        s2 = remove_outliers(s2)
        s3 = remove_outliers(s3)

    plt.figure(figsize=(10,6))

    # Plot histograms
    plt.hist(s1, bins=100, alpha=0.5, label=feature +' 1')
    plt.hist(s2, bins=100, alpha=0.5, label= feature +' 2')
    plt.hist(s3, bins=100, alpha=0.5, label='Final ' + feature)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Comparison of Histograms')
    plt.show()

def basic_histogram(s1, title='Delta of Teff'): 
    plt.figure(figsize=(10,6))

    # Plot histograms
    plt.hist(s1, bins=50, alpha=0.5)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()

def update_values(df, feature="teff_val"):
    condition_orig_is_nan = df[feature+'_orig'].isna()
    condition_new_is_nan = df[feature+'_new'].isna()

    df[feature] = np.where(
        condition_orig_is_nan, 
        df[feature+'_new'],
        np.where(condition_new_is_nan, df[feature+'_orig'], df[feature+'_new'])
    )

    return df

meta1, lcs, data = load_data_to_train()

print(meta1[meta1.abs_Gmag>20].drop_duplicates('OGLE_id'))


meta2 = load_new_validated_pp()
print(meta2[meta2['OGLE-ID']=='OGLE-BLG-LPV-225466'][['OGLE-ID','Dist']])


meta2['[Fe/H]'].hist()
plt.show()
meta1['[Fe/H]_J95'].hist()
plt.show()
## Teff

df1 = meta1[['OGLE_id', 'teff_val']]
df2 = meta2[["OGLE-ID", "Teff"]]
df1.columns = ['OGLE_id', 'teff_val']
df2.columns = ['OGLE_id', 'teff_val']

new_data = df1.merge(df2, on="OGLE_id", how="outer", suffixes=('_orig', '_new'))


new_data['delta_Teff'] = new_data["teff_val_orig"] - new_data["teff_val_new"]

new_data = update_values(new_data)

print('missing origin: ', new_data.drop_duplicates('OGLE_id').teff_val_orig.isna().sum())
print('missing new: ', new_data.drop_duplicates('OGLE_id').teff_val_new.isna().sum())
print('missing final: ', new_data.drop_duplicates('OGLE_id').teff_val.isna().sum())

basic_histogram(new_data.delta_Teff, title='Delta of Teff')

compare_frequency(meta1.drop_duplicates('OGLE_id').teff_val, 
                    meta2.drop_duplicates('OGLE-ID').Teff, 
                    new_data.drop_duplicates('OGLE_id').teff_val, 
                    feature="teff_val", clean=False)


meta1 = meta1.merge(new_data.drop_duplicates('OGLE_id'), on="OGLE_id", how="left", suffixes=('_orig', ''))

print('missing final: ', meta1.drop_duplicates('OGLE_id').teff_val.isna().sum())

## Period

df1 = meta1[['OGLE_id', 'Period']]
df2 = meta2[["OGLE-ID", "Period"]]
df1.columns = ['OGLE_id', 'Period']
df2.columns = ['OGLE_id', 'Period']

#del new_data

new_data = df1.merge(df2, on="OGLE_id", how="outer", suffixes=('_orig', '_new'))

new_data['delta_Period'] = new_data["Period_orig"] - new_data["Period_new"]

new_data = update_values(new_data, feature="Period")

print('missing origin: ', new_data.drop_duplicates('OGLE_id').Period_orig.isna().sum())
print('missing new: ', new_data.drop_duplicates('OGLE_id').Period_new.isna().sum())
print('missing final: ', new_data.drop_duplicates('OGLE_id').Period.isna().sum())

basic_histogram(new_data.delta_Period, title='Delta of Period')

basic_histogram(meta1.Period, title='Period 1')
basic_histogram(meta2.Period, title='Period 2')
basic_histogram(new_data.Period, title='Period 3')

compare_frequency(meta1.drop_duplicates('OGLE_id').Period, meta2.drop_duplicates('OGLE-ID').Period, new_data.drop_duplicates('OGLE_id').Period, feature="Period")



meta1 = meta1.merge(new_data.drop_duplicates('OGLE_id'), on="OGLE_id", how="left", suffixes=('_orig', ''))
print('missing final: ', meta1.drop_duplicates('OGLE_id').Period.isna().sum())


## Abs Mag G
df1 = meta1[['OGLE_id', 'abs_Gmag']]
df2 = meta2[["OGLE-ID", "GMAG_x"]]
df1.columns = ['OGLE_id', 'abs_Gmag']
df2.columns = ['OGLE_id', 'abs_Gmag']

#del new_data

new_data = df1.merge(df2, on="OGLE_id", how="outer", suffixes=('_orig', '_new'))

new_data['delta_abs_Gmag'] = new_data["abs_Gmag_orig"] - new_data["abs_Gmag_new"]

new_data = update_values(new_data, feature="abs_Gmag")

print('missing origin: ', new_data.drop_duplicates('OGLE_id').abs_Gmag_orig.isna().sum())
print('missing new: ', new_data.drop_duplicates('OGLE_id').abs_Gmag_new.isna().sum())
print('missing final: ', new_data.drop_duplicates('OGLE_id').abs_Gmag.isna().sum())

basic_histogram(new_data.abs_Gmag, title='Delta of abs_Gmag')

basic_histogram(meta1.abs_Gmag, title='abs_Gmag 1')
basic_histogram(meta2.GMAG_x, title='abs_Gmag 2')
basic_histogram(new_data.abs_Gmag, title='abs_Gmag 3')

print(meta1.abs_Gmag.min(), meta1.abs_Gmag.max())
print(meta2.GMAG_x.min(), meta2.GMAG_x.max())



compare_frequency(meta1.drop_duplicates('OGLE_id').abs_Gmag, 
                 meta2.drop_duplicates('OGLE-ID').GMAG_x, 
                 new_data.drop_duplicates('OGLE_id').abs_Gmag, feature="abs_Gmag", clean=False)


meta1 = meta1.merge(new_data.drop_duplicates('OGLE_id'), on="OGLE_id", how="left", suffixes=('_orig', ''))
print('missing final: ', meta1.drop_duplicates('OGLE_id').abs_Gmag.isna().sum())

# Metallicity
df1 = meta1[['OGLE_id', '[Fe/H]_J95']]
df2 = meta2[["OGLE-ID", "[Fe/H]"]]
df1.columns = ['OGLE_id', '[Fe/H]_J95']
df2.columns = ['OGLE_id', '[Fe/H]_J95']

new_data = df1.merge(df2, on="OGLE_id", how="outer", suffixes=('_orig', '_new'))


new_data['delta_Teff'] = new_data["[Fe/H]_J95_orig"] - new_data["[Fe/H]_J95_new"]

new_data = update_values(new_data, feature="[Fe/H]_J95")

print('missing origin: ', new_data.drop_duplicates('OGLE_id')['[Fe/H]_J95_orig'].isna().sum())
print('missing new: ', new_data.drop_duplicates('OGLE_id')['[Fe/H]_J95_new'].isna().sum())
print('missing final: ', new_data.drop_duplicates('OGLE_id')['[Fe/H]_J95'].isna().sum())

basic_histogram(new_data.delta_Teff, title='Delta of [Fe/H]_J95')

compare_frequency(meta1.drop_duplicates('OGLE_id')['[Fe/H]_J95'], 
                    meta2.drop_duplicates('OGLE-ID')['[Fe/H]'], 
                    new_data.drop_duplicates('OGLE_id')['[Fe/H]_J95'], 
                    feature="[Fe/H]_J95", clean=False)

meta1 = meta1.merge(new_data.drop_duplicates('OGLE_id'), on="OGLE_id", how="left", suffixes=('_orig', ''))
print(meta1.shape)

print('missing final: ', meta1.drop_duplicates('OGLE_id')['[Fe/H]_J95'].isna().sum())


# radius_val
df1 = meta1[['OGLE_id', 'radius_val']]
df2 = meta2[["OGLE-ID", "Rad"]]
df1.columns = ['OGLE_id', 'radius_val']
df2.columns = ['OGLE_id', 'radius_val']

new_data = df1.merge(df2, on="OGLE_id", how="outer", suffixes=('_orig', '_new'))

new_data['delta_radius_val'] = new_data["radius_val_orig"] - new_data["radius_val_new"]

new_data = update_values(new_data, feature="radius_val")

print('missing origin: ', new_data.drop_duplicates('OGLE_id')['radius_val_orig'].isna().sum())
print('missing new: ', new_data.drop_duplicates('OGLE_id')['radius_val_new'].isna().sum())
print('missing final: ', new_data.drop_duplicates('OGLE_id')['radius_val'].isna().sum())

basic_histogram(new_data.delta_radius_val, title='Delta of radius_val')

compare_frequency(meta1.drop_duplicates('OGLE_id')['radius_val'], 
                    meta2.drop_duplicates('OGLE-ID')['Rad'], 
                    new_data.drop_duplicates('OGLE_id')['radius_val'], 
                    feature="radius_val", clean=False)


meta1 = meta1.merge(new_data.drop_duplicates('OGLE_id'), on="OGLE_id", how="left", suffixes=('_orig', ''))

print('missing final: ', meta1.drop_duplicates('OGLE_id')['radius_val'].isna().sum())



# Logg
df2 = meta2[["OGLE-ID", "logg"]]
df2.columns = ["OGLE_id", "logg"]

basic_histogram(df2.logg, title='logg')
meta1 = meta1.merge(df2.drop_duplicates('OGLE_id'), on="OGLE_id", how="left", suffixes=('_orig', ''))

print(meta1[['OGLE_id','logg', '[Fe/H]_J95', 'abs_Gmag', 'teff_val', 'Period', 'abs_Gmag', 'radius_val']].head(50))

save_data(meta1, lcs, data)
