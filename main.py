import src.cnn as cnn
import src.bgmm as bgmm 
import src.fit_regressor as reg
#import src.create_lc as creator
import src.modifiedgmm as mgmm

PATH_MODELS = 'models/'
CLASSES = ['CEP']

mean_prior_dict = {'RRLYR':{'components':3,'mean_priors':[[1.0, 3.0, 4.9],[1.0, 3.0, 4.9]]}, 
                   'CEP':{'components':3,'mean_priors':[[1.0, 3.0, 4.9],[1.0, 3.0, 4.9]]}}
                #TODO: complete with expert knowledge this dictionary

def main():
    sampler = mgmm.ModifiedGaussianSampler(b=0.5, components = mean_prior_dict[CLASSES[0]]['components'])
    model_name = PATH_MODELS+'bgm_model_'+str(CLASSES[0])+'.pkl'
    samples = sampler.modify_and_sample(model_name)

    print(samples)
    
    reg.main()
    bgmm.train_and_save()
    cnn.run_cnn()

if __name__ == "__main__":
    main()
