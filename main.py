import src.cnn as cnn
import src.gmm.bgmm as bgmm 
import src.sampler.fit_regressor as reg
import src.sampler.create_lc as creator
import src.gmm.modifiedgmm as mgmm
from src.utils import load_yaml_priors

PATH_MODELS = 'models/'
PATH_PRIOS = 'src/gmm/priors.yaml'
CLASSES = ['CEP']

mean_prior_dict = load_yaml_priors(PATH_PRIOS)

def main(train_gmm = True, create_samples=False):

    if train_gmm: 
        bgmm.train_and_save()

    if create_samples:
        sampler = mgmm.ModifiedGaussianSampler(b=0.5, components = mean_prior_dict[CLASSES[0]]['components'])
        model_name = PATH_MODELS+'bgm_model_'+str(CLASSES[0])+'.pkl'
        samples = sampler.modify_and_sample(model_name)
        z_hat = reg.main(samples)
        creator.main(samples, z_hat)

    cnn.run_cnn(create_samples, mode_running='create')
    
if __name__ == "__main__":
    main()
