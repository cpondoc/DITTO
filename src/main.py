#!/usr/bin/env python3
import os
import sys

import yaml

import wandb
from config.config import build_config
from trainers.ac_trainer import ACTrainer
from trainers.bc_trainer import BCTrainer
from trainers.rssm_trainer import RSSMTrainer


def load_conf(config_file, env_name):
    with open(config_file, "r") as f:
        raw_conf = yaml.safe_load(f)
    base_conf = raw_conf["base"]
    data_conf = raw_conf["data"]
    env_conf = raw_conf[env_name]
    dream_conf = raw_conf["dreamer"] if "dreamer" in raw_conf else None
    conf = {**base_conf, **env_conf}
    conf["data"] = data_conf
    conf["dreamer"] = dream_conf
    return conf


def wm_train(conf):
    # Set up Weights and Biases
    os.environ['WANDB__EXECUTABLE'] = "/home/cdpg/anaconda3/bin/python"
    wandb.login()
    wandb.init(project="world-model", config=conf.to_dict(),
               settings=wandb.Settings(code_dir="."))
    #wandb.run.log_code(".")

    # Create the trainer, watch the model, train
    trainer = RSSMTrainer(conf)
    wandb.watch(trainer.model)
    trainer.train()


def bc_train(conf):
    os.environ['WANDB__EXECUTABLE'] = "/home/cdpg/anaconda3/bin/python"
    wandb.login()
    wandb.init(project="bc", config=conf.to_dict(),
               settings=wandb.Settings(code_dir="."))
    conf = wandb.config
    trainer = BCTrainer(conf)
    wandb.watch(trainer.model)
    trainer.train()


def ac_train(conf):
    # Note -- have to set WANDB__EXECUTABLE here
    os.environ['WANDB__EXECUTABLE'] = "/home/cdpg/anaconda3/bin/python"
    wandb.login()
    wandb.init(project="dreamer", config=conf.to_dict(),
               settings=wandb.Settings(code_dir="."))
    
    # Might comment out just to check?
    # wandb.run.log_code(".")
    #conf = wandb.config
    # print('conf',conf)

    # Initialize trainer?
    trainer = ACTrainer(conf)
    wandb.watch(trainer.policy)
    # wandb.watch(trainer.discrim)
    trainer.train()
    print("Finished training")


def main(conf_path=None):
    config_file = "config/test_config.yaml" if conf_path is None else conf_path # Change to nocturne_config.yaml
    conf = build_config(config_file)
    # ac_train(conf)
    
    # Now just doing behavior cloning training
    bc_train(conf)

    # Currently just doing world model training
    # wm_train(conf)


if __name__ == "__main__":
    conf_path = None
    if len(sys.argv) > 1:
        conf_path = sys.argv[1]
        print('got conf', conf_path)

    main(conf_path=conf_path)
