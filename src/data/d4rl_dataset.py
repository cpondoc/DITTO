import glob
import os
import sys

import cv2
import d4rl_atari
import gym
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class D4RLDataset(Dataset):
    def __init__(self, dataset_config):
        self.action_type = dataset_config.action_type
        self.seq_length = dataset_config.seq_length
        self.device = dataset_config.load_device
        self.num_transitions = 0
        self.data_keys = dataset_config.data_keys

        self.pixel_mean = dataset_config.pixel_mean
        self.pixel_std = dataset_config.pixel_std
        # self.pixel_mean = 33
        # self.pixel_std = 55

        if "dataset_path" in dataset_config:
            self.data = self.load_files(dataset_config.dataset_path)
        else:
            self.data = self.load_d4rl_data(dataset_config.dataset_names)

        print("total transitions:", self.num_transitions)
        print("num batch elements:", self.num_transitions//self.seq_length)

    def fix_obs(self, img, new_hw=64, resize=True, sb3=False):
        """
        Normalizes image: FIX TO 128 for Atari!
        """
        #img = np.transpose(img, (1, 2, 0))

        # this only works for single channel images
        # need to handle channels dim correctly if color/multichannel
        img = np.squeeze(img)
        if resize:
            img = np.array(cv2.resize(img, dsize=(
                new_hw, new_hw), interpolation=cv2.INTER_AREA))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = (img - self.pixel_mean) / self.pixel_std

        return img

    def fix_terminals(self, terminals):
        resets = np.zeros(len(terminals), dtype=np.float32).astype(bool)
        idxs = (np.where(terminals)[0]+1)%len(terminals)
        resets[idxs] = True
        return resets

    def fix_actions(self, actions, reset, cats=18):
        """takes array of scalar actions and converts to one-hot.
        also offsets actions b/c dreamer uses pre-actions instead of post-actions"""
        ridxs = np.where(reset)[0]
        # actions = actions.reshape(-1)
        targets = np.roll(actions, 1, axis=0)
        if self.action_type == "discrete":
            one_hot = np.eye(cats)[targets]
            one_hot[ridxs] = np.zeros_like(one_hot[0])
            return one_hot
        else:
            return targets

    def get_data(self, dataset_name):
        sys.stdout = open(os.devnull, 'w')
        env = gym.make(dataset_name)
        data = env.get_dataset()
        sys.stdout = sys.__stdout__
        return data
    
    def other_load_files(self, path):
        data = {k: [] for k in self.data_keys}
        print("loading dataset from", path)

        filenames = glob.glob(path+'/*.npz')
        filenames = filenames[:1000]
        print('got', len(filenames), 'files...')
        rewards=[]

        # Where we are loading in the data
        for filename in tqdm(filenames, desc="loading files.."):
            # Load in dictionary and observation key
            npz_dict = np.load(filename,allow_pickle=True)
            obs_key = "images" # or "observations" or "vecobs"
            
            # Determine number of transitions and episode length
            episode_length = len(npz_dict["actions"])
            self.num_transitions += episode_length

            # [CHECK] Crafted custom terminals array, since no terminal state
            terminals = np.array([False] * episode_length)
            resets = self.fix_terminals(terminals)

            # Fix actions, set specific parameter for action space
            data["reset"].extend(resets)
            data["action"].extend(self.fix_actions(
                npz_dict["actions"], resets, cats=294))

            # Extend, gather observations
            data["obs"].extend([self.fix_obs(x, resize=True, sb3=True) #should resize be here?
                                for x in npz_dict[obs_key]])
            #rewards.append(np.sum(npz_dict["rewards"]))
        print("finished loading")
        
        # No need to print rewards here
        #print("Reward min, max, mean, std", np.min(rewards), 
        #    np.max(rewards), np.mean(rewards), np.std(rewards))

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        print(data["action"].shape, "here")
        data["reset"] = data["reset"].bool()
        return data

    def load_files(self, path):
        data = {k: [] for k in self.data_keys}
        print("loading dataset from", path)

        filenames = glob.glob(path+'/*.npz')
        filenames = filenames[:1000]
        print('got', len(filenames), 'files...')

        rewards=[]

        for filename in tqdm(filenames, desc="loading files.."):
            npz_dict = np.load(filename,allow_pickle=True)

            obs_key = "images" # or "observations" or "vecobs"
            
            episode_length = len(npz_dict["actions"])
            self.num_transitions += episode_length
            resets = self.fix_terminals(npz_dict["terminals"])
            data["reset"].extend(resets)
            data["action"].extend(self.fix_actions(
                npz_dict["actions"], resets))
            data["obs"].extend([self.fix_obs(x, resize=True, sb3=True) #should resize be here?
                                for x in npz_dict[obs_key]])
            rewards.append(np.sum(npz_dict["rewards"]))
        print("finished loading")
        
        print("Reward min, max, mean, std", np.min(rewards), 
            np.max(rewards), np.mean(rewards), np.std(rewards))
        

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        print(data["action"].shape, "here")
        data["reset"] = data["reset"].bool()
        return data

    def load_d4rl_data(self, dataset_names):
        data = {k: [] for k in self.data_keys}
        print("loading the following datasets:", dataset_names)

        for dataset_name in dataset_names:
            npz_dict = self.get_data(dataset_name)
            episode_length = len(npz_dict["actions"])
            self.num_transitions += episode_length
            data["reset"].extend(self.fix_terminals(npz_dict["terminals"]))
            data["action"].extend(self.fix_actions(
                npz_dict["actions"], data["reset"]))
            data["obs"].extend([self.fix_obs(x)
                                for x in npz_dict["observations"]])
        print("finished loading")

        data = {k: torch.tensor(np.array(v, dtype=np.float32)).to(self.device)
                for k, v in data.items()}
        data["reset"] = data["reset"].bool()
        return data

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        end_idx = idx+self.seq_length
        action, obs, reset = \
            [self.data[key][idx:end_idx] for key in self.data_keys]

        pad_size = end_idx-self.num_transitions
        if pad_size > 0:
            action = torch.cat((action, self.data['action'][:pad_size]), dim=0)
            obs = torch.cat(
                (obs, self.data['obs'][:pad_size]), dim=0)
            reset = torch.cat((reset, self.data['reset'][:pad_size]), dim=0)

        ret = {"action": action, "obs": obs, "reset": reset}
        return ret

    def get_trans(self, idx):
        action, obs, reset = \
            [self.data[key][idx].unsqueeze(0).unsqueeze(0)
             for key in self.data_keys]
        return action, obs, reset          
        
        