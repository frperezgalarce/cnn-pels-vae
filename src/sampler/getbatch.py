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

gpu: bool = True # fail when true is selected

class SyntheticDataBatcher:
    def __init__(self, config_file_path: str = 'src/regressor.yaml', 
                 nn_config_path: str = 'src/nn_config.yaml', paths: str = 'src/paths.yaml', PP=[], vae_model=None, 
                 n_samples=16, seq_length = 100, batch_size=128):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config_file = self.load_yaml(config_file_path)
        self.nn_config = self.load_yaml(nn_config_path)
        self.path = self.load_yaml(paths)['paths']
        self.mean_prior_dict = self.load_yaml(self.path['PATH_PRIOS'])  # to be filled in later
        self.priors = False
        self.PP = PP
        self.vae_model = vae_model
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.CLASSES = ['ACEP','CEP', 'DSCT', 'ECL',  'ELL', 'LPV',  'RRLYR', 'T2CEP']
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
        return len([key for key in star_type_data.keys() if key != 'CompleteName'])



    def process_in_batches(self, model, mu_, times, onehot, phy, batch_size):
        # Split tensors into smaller batches
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

        # Concatenate results from all batches
        xhat_mu = torch.cat(results, dim=0)
        return xhat_mu


    def attempt_sample_load(self, model_name: str, sampler: 'YourSamplerType') -> Tuple[Union[np.ndarray, None], bool]:
        try:
            samples = sampler.modify_and_sample(model_name, n_samples=self.n_samples)
            return samples, True
        except Exception as e:
            raise Exception(f"Failed to load samples from model {model_name}. Error: {str(e)}")

    def create_synthetic_batch(self, plot_example=True, b=1.0):
        print(self.path)
        PATH_MODELS = self.path['PATH_MODELS']
        PATH_DATA = self.path['PATH_DATA_FOLDER']
        lb = []

        print(self.n_samples)
        print(len(list(self.nn_config['data']['classes'])))
        print(len(list(self.nn_config['data']['classes']))*self.n_samples)
        with open(PATH_MODELS+'label_encoder_vae.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        for star_class in list(self.nn_config['data']['classes']):
            torch.cuda.empty_cache()

            print('------- sampling ' +star_class+'---------')
            lb += [star_class] * self.n_samples

            integer_encoded = label_encoder.transform(lb)
            n_values = len(label_encoder.classes_)
            onehot = np.eye(n_values)[integer_encoded]

            encoded_labels, _ = utils.transform_to_consecutive(integer_encoded, label_encoder)
            n_values = len(np.unique(encoded_labels))
            onehot_to_train = np.eye(n_values)[encoded_labels]


            components = self.count_subclasses(self.mean_prior_dict['StarTypes'][star_class])
            print(star_class +' includes '+ str(components) +' components ')
            sampler: mgmm.ModifiedGaussianSampler = mgmm.ModifiedGaussianSampler(b=b, components=components, features=self.PP)
            model_name = self.construct_model_name(star_class, PATH_MODELS)
            samples, error = self.attempt_sample_load(model_name, sampler)
            print(samples.shape)
            # If we have priors and failed to load the model, try with priors=False
            if self.priors and samples is None:
                model_name = self.construct_model_name(star_class, PATH_MODELS)
                print(self.n_samples)
                samples, error = self.attempt_sample_load(model_name, sampler, n_samples=self.n_samples)
            
            # If still not loaded, raise an error
            if samples is None:
                raise ValueError("The model can't be loaded." + str(error))

            if 'all_classes_samples' in locals() and all_classes_samples is not None: 
                all_classes_samples = np.vstack((samples, all_classes_samples))
            else: 
                all_classes_samples = samples
                print(all_classes_samples.shape)

        print('cuda: ', torch.cuda.is_available())
        print('model: ', self.vae_model)


        columns = ['Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg']
        index_period = columns.index('Period')

        print(all_classes_samples.shape)
        print(len(lb))
        mu_ = reg.process_regressors(self.config_file, phys2=columns, samples= all_classes_samples, 
                                            from_vae=False, train_rf=False)
        
        times = [i/600 for i in range(600)]
        times = np.tile(times, (self.n_samples*len(list(self.nn_config['data']['classes'])), 1))
        times = np.array(times)  
        times = torch.from_numpy(times).to(self.device)
        # Load model first
        vae, _ = utils.load_model_list(ID=self.vae_model, device=self.device)

        # Convert times to float32
        times = times.to(dtype=torch.float32)

        # Directly convert to tensors and move to the GPU
        mu_ = torch.tensor(mu_, device=self.device)
        onehot = torch.tensor(onehot, device=self.device)
        lb = np.array(lb)  
        pp = torch.tensor(all_classes_samples, device=self.device)

        # Clear GPU cache
        torch.cuda.empty_cache()
        xhat_mu = self.process_in_batches(vae, mu_, times, onehot, pp, 8)

        #xhat_mu = vae.decoder(mu_, times, label=onehot, phy=pp)

        xhat_mu = torch.cat([times.unsqueeze(-1), xhat_mu], dim=-1).cpu().detach().numpy()
        indices = np.random.choice(xhat_mu.shape[0], 24, replace=False)
        sampled_arrays = xhat_mu[indices, :, :]

        utils.plot_wall_lcs_sampling(sampled_arrays, sampled_arrays,  cls=lb[indices],  column_to_sensivity=index_period,
                                to_title = pp[indices], sensivity = 'Period', all_columns=columns, save=True) 

        lc_reverted = utils.revert_light_curve(pp[:,index_period], xhat_mu, classes = lb)

        print(np.min(lc_reverted[0][0]),np.max(lc_reverted[0][0]))
        print(np.min(lc_reverted[0][1]),np.max(lc_reverted[0][1])) 
        
        if plot_example:
            plt.figure()
            plt.scatter(lc_reverted[0][1], lc_reverted[0][0])
            plt.show()

        #TODO: manage nan
        mean_value = np.nanmean(lc_reverted)
        lc_reverted[np.isnan(lc_reverted)] = mean_value

        print('before diff: ')
        print(lc_reverted[0])
        lc_reverted = np.diff(lc_reverted, axis=-1)

        print('after diff: ')
        print(lc_reverted[0])
        if plot_example:
            plt.figure()
            plt.scatter(lc_reverted[0][1], lc_reverted[0][0])
            plt.show()

        oversampling = True
        if np.sum(np.isnan(lc_reverted)) > 0:
            print(f"Number of NaN values detected: {np.sum(np.isnan(lc_reverted))}")
            raise ValueError("NaN values detected in lc_reverted array")
        if oversampling: 
            sampler = LightCurveRandomSampler(lc_reverted, onehot_to_train, self.seq_length, 8)
            lc_reverted, onehot_to_train = sampler.sample()
            print(lc_reverted.shape)
            print(onehot_to_train.shape)
        else:
            lc_reverted = lc_reverted[:, :, :self.seq_length]
        utils.save_arrays_to_folder(lc_reverted, onehot_to_train , PATH_DATA)

        numpy_array_x = np.load(PATH_DATA+'/x_batch_pelsvae.npy', allow_pickle=True)
        numpy_array_y = np.load(PATH_DATA+'/y_batch_pelsvae.npy', allow_pickle=True)

        if plot_example:
            plt.figure()
            plt.scatter(numpy_array_x[0][0], numpy_array_x[0][1])
            plt.show()

        synth_data = utils.move_data_to_device((numpy_array_x, numpy_array_y), self.device)
        synthetic_dataset = TensorDataset(*synth_data)
        synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=self.batch_size, shuffle=True)

        return synthetic_dataloader