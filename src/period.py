import pandas as pd
import scipy.signal as signal
import numpy as np

PATH_FEATURES_TRAIN = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Train_rrlyr-1.csv'
PATH_FEATURES_TEST = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Test_rrlyr-1.csv'
PATH_LIGHT_CURVES_OGLE = '/home/franciscoperez/Desktop/Code/FATS/LCsOGLE/data/'


def read_light_curve_ogle(n=1):
    path_train = PATH_FEATURES_TRAIN
    path_test = PATH_FEATURES_TEST
    lc_test = pd.read_table(path_test, sep= ',')
    #lc_test = lc_test[lc_test.label=='ClassA']
    lc_train = pd.read_table(path_train, sep= ',')
    #lc_train = lc_train[lc_train.label=='ClassA']
    example_test  = lc_test['ID'].sample(n)
    example_train = lc_train['ID'].sample(n)
    lcs = {}
    for lc in example_test.unique():
        new_test = lc_test[lc_test.ID==lc].ID.str.split("-", n = 3, expand = True)
        period = lc_test[lc_test.ID==lc].PeriodLS.values[0]
        print('period from FATS', period)
        field = new_test[new_test.columns[1]].values[0].lower()
        lcu = pd.read_table(PATH_LIGHT_CURVES_OGLE+field+
                            '/'+lc.split('-')[2].lower()+'/phot/I/'+
                            lc, sep=" ", names=['time', 'magnitude', 'error'])
        #print(lcu['time'])
        lcu['time'] = lcu['time'] - lcu['time'].min()
        lcu.dropna(axis=0, inplace=True)
        #print(lcu['time'])
        #print(pd.to_datetime(lcu['time']))

        lcs[lc]=lcu
    return lcs, period

def period_optimization(lcs, period, k=2):
    "k : number of periods required"
    period_dict = {}
   
    for key in lcs:
        period_dict[key] = [period,period+np.random.uniform(0,0.1)]
    '''
        nout = 100000
        w = np.linspace(0.01, 1000, nout)
        print(lcs[key]['time'])
        pgram = signal.lombscargle(lcs[key]['time'], lcs[key]['magnitude'], w, normalize=True)
        periods = pgram.argsort()[-k:][::-1]
        T = 1.0 / pgram[periods]
        new_time = np.mod(pgram[periods], 2 * T) / (2 * T)

        period_dict[key]=new_time
    '''
    return period_dict

def phased_curves(light_curves, periods, k = 2): 
    print(light_curves)
    phased_light_curves = {}
    for key in light_curves.keys():
        lcu_dict_nested = {}
        for p in range(0,k):
            print(light_curves[key])
            m0 = light_curves[key]['magnitude'].min()
            df = pd.DataFrame(light_curves[key])
            t0 = df[df.magnitude==m0].time.min()
            t = light_curves[key]['time']
            lcu = ((t-t0)/periods[key][p])%1 
            lcu_dict_nested[p] = lcu
        phased_light_curves[key] = lcu_dict_nested
    return phased_light_curves 

def main(): 
    light_curves_dict, period = read_light_curve_ogle(n=2)
    #print(light_curves_dict)
    period_dict = period_optimization(light_curves_dict, period, k=2)

    phased_light_curves = phased_curves(light_curves_dict, period_dict)

    print(phased_light_curves)

if __name__ == "__main__":
    main()