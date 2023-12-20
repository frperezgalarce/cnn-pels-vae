from typing import Union, Tuple, Optional, Any, Dict, List
import numpy as np
import pickle
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
import src.utils as utils
import src.gmm.modifiedgmm as mgmm
import src.sampler.fit_regressor as reg
import matplotlib.pyplot as plt
from src.sampler.LightCurveRandomSampler import LightCurveRandomSampler

gpu: bool = True 
with open('src/nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

    
class SyntheticDataBatcher:
    def __init__(self, config_file_path: str = 'src/regressor.yaml', 
                 nn_config_path: str = 'src/nn_config.yaml', paths: str = 'src/paths.yaml', PP=[], 
                 vae_model=None, 
                 n_samples=16, seq_length = 100, batch_size=128, prior=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config_file = self.load_yaml(config_file_path)
        self.nn_config = self.load_yaml(nn_config_path)
        self.path = self.load_yaml(paths)['paths']
        self.mean_prior_dict = self.load_yaml(self.path['PATH_PRIOS'])  
        self.priors = prior
        self.PP = PP
        self.vae_model = vae_model
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.delta_max = 100
        self.batch_size = batch_size

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def construct_model_name(self, star_class: str, base_path: str = 'PATH_MODELS'):
        """Construct a model name given parameters."""
        file_name = f"{base_path}bgm_model_{str(star_class)}_priors_{self.priors}_PP_{len(self.PP)}.pkl"
        return file_name

    @staticmethod
    def count_subclasses(star_type_data: Dict[str, Any]) -> int:
        excluded_keys = ['CompleteName', 'min_period', 'max_period']
        return len([key for key in star_type_data.keys() if key not in excluded_keys])
    
    @staticmethod
    def process_in_batches(model, mu_, times, onehot, phy, batch_size):

        total_samples = mu_.size(0)
        n_batches = (total_samples + batch_size - 1) // batch_size

        results = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)

            mu_batch = mu_[start_idx:end_idx]
            times_batch = times[start_idx:end_idx]
            onehot_batch = onehot[start_idx:end_idx]
            phy_batch = phy[start_idx:end_idx]

            xhat_mu_batch = model.decoder(mu_batch, times_batch, label=onehot_batch, phy=phy_batch)
            results.append(xhat_mu_batch)
            del xhat_mu_batch
            torch.cuda.empty_cache()

        xhat_mu = torch.cat(results, dim=0)
        return xhat_mu
    
    @staticmethod
    def plot_light_curve(array, label_y = 'Magnitude', label_x='MJD'):
        fig, axis = plt.subplots(nrows=8, ncols=3, 
                             figsize=(16,14),
                             sharex=True, sharey=True)
        axs = axis.flatten()
        for i in range(len(array)):
            axs[i].scatter(array[i][0], array[i][1], color='royalblue', s=4)
        axis[-1,1].set_xlabel(label_x, fontsize=20)
        axis[4,0].set_ylabel(label_y, fontsize=20)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def attempt_sample_load(model_name: str, sampler: 'YourSamplerType', n_samples=nn_config['training']['sinthetic_samples_by_class']) -> Tuple[Union[np.ndarray, None], bool]:
        try:
            samples = sampler.modify_and_sample(model_name, n_samples=n_samples, 
                                                mode= nn_config['sampling']['mode'])
            return samples, True
        except Exception as e:
            raise Exception(f"Failed to load samples from model {model_name}. Error: {str(e)}")
    
    @staticmethod
    def load_encoder_vae(PATH_MODELS):
        with open(PATH_MODELS+'label_encoder_vae.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder
    
    def get_label_encodings(self, label_encoder, lb): 
        integer_encoded = label_encoder.transform(lb)
        n_values = len(label_encoder.classes_)
        onehot = np.eye(n_values)[integer_encoded]

        encoded_labels, _ = utils.transform_to_consecutive(integer_encoded, label_encoder)
        n_values = len(np.unique(encoded_labels))
        onehot_to_train = np.eye(n_values)[encoded_labels]
        onehot = torch.tensor(onehot, device=self.device)
        return onehot, onehot_to_train

    def create_time_sequences(self, lb, period):
        
        np.set_printoptions(suppress=True)
        times, original_sequences =  utils.get_only_time_sequence(n=1, star_class=lb, 
                                                                 period = period, factor1=0.8, 
                                                                 factor2= 1.2)
        times = np.array(times) 
        original_sequences = np.array(original_sequences) 
        times = torch.from_numpy(times).to(self.device)
        times = times.to(dtype=torch.float32)
        
        return times, original_sequences
    
    def batch_preprocessing(self, array):
        
        array = np.diff(array, axis=-1)
        mean_value = np.nanmean(array)
        array[(array)> self.delta_max] = mean_value 
        
        return array

    def save_batch(self, lc_reverted, onehot_to_train , PATH_DATA):
        
        utils.save_arrays_to_folder(lc_reverted, onehot_to_train , PATH_DATA)

        numpy_array_x = np.load(PATH_DATA+'/x_batch_pelsvae.npy', allow_pickle=True)
        numpy_array_y = np.load(PATH_DATA+'/y_batch_pelsvae.npy', allow_pickle=True)

        synth_data = utils.move_data_to_device((numpy_array_x, numpy_array_y), self.device)
        synthetic_dataset = TensorDataset(*synth_data)
        synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=self.batch_size, shuffle=True)

        return synthetic_dataloader
    
    def create_labels(self, star_class, lb, samples_dict):
        if samples_dict==None:
            n_samples = self.n_samples
            lb += [star_class] * self.n_samples
        else: 
            n_samples = int(samples_dict[star_class])
            lb += [star_class] * n_samples

        return lb
    
    def get_all_labels(self, samples_dict):
        lb = []
        for star_class in list(self.nn_config['data']['classes']):
            torch.cuda.empty_cache()            
            lb = self.create_labels(star_class, lb, samples_dict)
        return lb
    
    def get_latent_space(self, all_classes_samples, index_period):
        z = reg.process_regressors(self.config_file, phys2=self.PP, samples= all_classes_samples, 
                                            from_vae=False, train_rf=False) 
        z = torch.tensor(z, device=self.device)
        return z

    def check_nan(self, lc_reverted):
        if np.sum(np.isnan(lc_reverted)) > 0:
            print(f"Number of NaN values detected: {np.sum(np.isnan(lc_reverted))}")
            raise ValueError("NaN values detected in lc_reverted array")
    
    #TODO: review oversampling and undersampling methods
    def set_lc_length(self, oversampling, lc_reverted, n_oversampling, onehot_to_train):
        if oversampling: 
            sampler = LightCurveRandomSampler(lc_reverted, onehot_to_train, self.seq_length, n_oversampling)
            lc_reverted, onehot_to_train = sampler.sample()
        else:
            obs = lc_reverted.shape[2]
            print(obs)
            random_indexes = np.sort(np.random.choice(600, self.seq_length, replace=False))
            lc_reverted = lc_reverted[:, :, random_indexes]
        return lc_reverted, onehot_to_train
        
    def get_samples(self, samples_dict, PATH_MODELS, b):
        for star_class in list(self.nn_config['data']['classes']):

            if samples_dict==None:
                n_samples = self.n_samples
            else: 
                n_samples = int(samples_dict[star_class])

            print('------- sampling ' +star_class+'---------')
            components = self.count_subclasses(self.mean_prior_dict['StarTypes'][star_class])

            print(star_class +' includes '+ str(components) +' components ')

            sampler: mgmm.ModifiedGaussianSampler = mgmm.ModifiedGaussianSampler(b=b, 
                                                                                components=components, 
                                                                                features=self.PP)
            model_name = self.construct_model_name(star_class, PATH_MODELS)
            samples, error = self.attempt_sample_load(model_name, sampler, n_samples=n_samples)

            if samples is None:
                raise ValueError("The model can't be loaded." + str(error))

            if 'all_classes_samples' in locals() and all_classes_samples is not None: 
                all_classes_samples = np.vstack((all_classes_samples, samples))
            else: 
                all_classes_samples = samples

        return all_classes_samples
        
    
    def create_synthetic_batch(self, plot_example=False, b=1.0, 
                               wandb_active=False, samples_dict = None, 
                               oversampling = True, n_oversampling=12):
        
        PATH_MODELS = self.path['PATH_MODELS']
        PATH_DATA = self.path['PATH_DATA_FOLDER']

        label_encoder = self.load_encoder_vae(PATH_MODELS)
        
        lb = self.get_all_labels(samples_dict)
        
        onehot, onehot_to_train = self.get_label_encodings(label_encoder, lb)
        
        all_classes_samples = self.get_samples(samples_dict, PATH_MODELS, b)
        
        index_period = self.PP.index('Period')

        z = self.get_latent_space(all_classes_samples, index_period)
        
        lb = np.array(lb)  
        pp = torch.tensor(all_classes_samples, device=self.device)

        vae, _ = utils.load_model_list(ID=self.vae_model, device=self.device)
        
        #TODO: review method
        times, original_sequences = self.create_time_sequences(lb, all_classes_samples[:,index_period])  

        xhat_mu = self.process_in_batches(vae, z, times, onehot, pp, 1)
        xhat_mu = torch.cat([times.unsqueeze(-1), xhat_mu], dim=-1).cpu().detach().numpy()

        indices = np.random.choice(xhat_mu.shape[0], 24, replace=False)
        sampled_arrays = xhat_mu[indices, :, :]
        
        utils.plot_wall_lcs_sampling(sampled_arrays, sampled_arrays,  cls=lb[indices],  column_to_sensivity=index_period,
                                to_title = pp[indices], sensivity = 'Period', all_columns=self.PP, save=False, 
                                wandb_active=wandb_active) 

        #TODO: review method
        lc_reverted = utils.revert_light_curve(pp[:,index_period], xhat_mu, original_sequences, classes = lb) 

        if plot_example:
            self.plot_light_curve(lc_reverted[indices], label_y = 'Magnitude', label_x='MJD')

        mean_value = np.nanmean(lc_reverted)
        lc_reverted[np.isnan(lc_reverted)] = mean_value

        lc_reverted, onehot_to_train = self.set_lc_length(oversampling, lc_reverted, n_oversampling, onehot_to_train)
        
        lc_reverted = self.batch_preprocessing(lc_reverted)

        if plot_example:
            self.plot_light_curve(lc_reverted[indices], label_y = 'Delta magnitude', label_x='delta time')

        self.check_nan(lc_reverted)

        synthetic_dataloader = self.save_batch(lc_reverted, onehot_to_train , PATH_DATA)

        return synthetic_dataloader