import os
import pickle
from typing import Dict
from argparse import Namespace

import yaml
import optuna
import numpy as np
from optuna.trial import Trial

from src.data import prepare_realgrab_data
from src.simulator import Environment
from src.utils import (
    make_plot, get_timestamp, 
    calc_diff_score, calc_fft_diff_score,
    process_simulated_signal, normalize_score
)

cfg_file = "config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

class TemporalResultSaver:

    def __init__(self, save_loc: str):
        """
        Args:
            save_loc (str): "results/v001/(date)/(settings)/hps/"
        Returns:
            None
        """
        self.save_loc = save_loc
        
    def save_temporal_result(self, study, frozen_trial):
        """
        Arguments are required by optuna.
        Args:
            study: 
            frozen_trial: <REQUIRED BY OPTUNA>
        Returns:
            None
        """
        filename = f"/tmp_result_hps.csv"
        csv_name = self.save_loc + filename
        df_hps = study.trials_dataframe()
        df_hps.to_csv(csv_name)
        
class Objective:

    def __init__(
        self, 
        base_args: Namespace, 
        search_space: Dict, 
    ):
        """
        Args:
            base_args (Namespace): Hyper parameters fixed during training.
            search_space (Dict)  : Dictionary of search space for hyper paramter optimization.
                                    {"arg_name": [range_type, low, high]}
        Returns:
            None
        """
        self.args = base_args
        self.search_space = search_space

        self.real_grab = prepare_realgrab_data(base_args.target_label)
        self.nstep = int(
            self.args.activation_length / config["env_param"]["sec_per_step"]
        )

    def prepare_params(self, trial: Trial):
        """
        Concatenate `base_args` and parameters sampled from `search_space`,
        return as single Namespace.
        Args:
            trial (Trial):
        Returns:
            params (Namespace): parameters in simulation
        """
        params = vars(self.args) # Namespace -> Dict
        for variable, sample_info in self.search_space.items():
            if sample_info[0] == "int":
                _param = trial.suggest_int(variable, sample_info[1], sample_info[2])
            elif sample_info[0] == "uniform":
                _param = trial.suggest_uniform(variable, sample_info[1], sample_info[2])
            elif sample_info[0] == "log_uniform":
                _param = trial.suggest_loguniform(variable, sample_info[1], sample_info[2])
            elif sample_info[0] == "discrete_uniform":
                _param = trial.suggest_discrete_uniform(
                    variable, sample_info[1], sample_info[2], sample_info[3])
            elif sample_info[0] == "int_pow":
                _param = trial.suggest_int(variable, sample_info[1], sample_info[2])
                _param = 2 ** _param
            elif sample_info[0] == "categorical":
                _param = trial.suggest_categorical(variable, sample_info[1])
            else:
                raise NotImplementedError
            
            if variable == "stim_power" and self.args.activation_ach.startswith("on-auto"):
                variable = "activation_ach"
                _param = self.args.activation_ach + f"-{_param:.3f}"

            params.update({variable: _param})
        
        params = Namespace(**params) # Dict -> Namespace
        return params

    def _calc_score(self, norm_simulated: np.ndarray):
        """
        Args:
            norm_simulated (np.ndarray): Array of shape (2, simulation_length).
                (signal order: ach, da)
        Returns:
            best_score
        """
        if self.args.score_type == "diff":
            best_score_ach = calc_diff_score(
                self.real_grab, norm_simulated, signal_idx=0)
            best_score_da = calc_diff_score(
                self.real_grab, norm_simulated, signal_idx=1)
            best_score = (best_score_da + best_score_ach) / 2
        
        elif self.args.score_type == "fft_diff":
            best_score_ach = calc_fft_diff_score(
                self.real_grab, norm_simulated, signal_idx=0)
            best_score_da = calc_fft_diff_score(
                self.real_grab, norm_simulated, signal_idx=1)
            best_score = (best_score_da + best_score_ach) / 2
        
        elif self.args.score_type == "mix":
            best_score_ach_d = calc_diff_score(
                self.real_grab, norm_simulated, signal_idx=0)
            best_score_da_d = calc_diff_score(
                self.real_grab, norm_simulated, signal_idx=1)
            best_score_ach_f = calc_fft_diff_score(
                self.real_grab, norm_simulated, signal_idx=0)
            best_score_da_f = calc_fft_diff_score(
                self.real_grab, norm_simulated, signal_idx=1)

            # Sum up ach and da score for each metric
            best_score_d = best_score_ach_d + best_score_da_d
            best_score_f = best_score_ach_f + best_score_da_f

            # Normalize and calc sum.
            best_score_d_norm = normalize_score(
                best_score_d, self.args.lambda_, "diff")
            best_score_f_norm = normalize_score(
                best_score_f, 1-self.args.lambda_, "fft_diff")

            best_score = (best_score_d_norm + best_score_f_norm) / 2
        else:
            raise NotImplementedError
        
        return best_score
        
    def __call__(self, trial: Trial):
        """
        Args:
            trial (optuna.trial.Trial): 

        Returns:
            score (float): the score (diff NTsim-NTreal)
        """
        
        params = self.prepare_params(trial)
        
        env = Environment(
            self.args.activation_da, 
            self.args.activation_ach, 
            params.diff_step,
            params.diff_rate,
            params.d1r_delay, 
            params.d2r_delay, 
            params.d1r_ach_efficacy,
            params.d2r_daaxon_efficacy,
            params.d2r_ach_efficacy,
            self.args.da_split_type, 
            self.args.use_nAChR_at_da,
            self.args.use_d2_at_da,
            self.args.use_d1_at_ach,
            self.args.use_d2_at_ach,
            config["env_param"]["max_da"],
            config["env_param"]["max_ach"],
            self.args.seed
        )
        try: 
            # Simulator will raise error 
            # if absolute value of ach or da is over threshold given in config.yaml.
            simulation = env.run(self.nstep)
        except:
            return 500

        norm_simulated_da, _ = process_simulated_signal(simulation, "da")
        norm_simulated_ach, _ = process_simulated_signal(simulation, "ach")

        norm_simulated = np.stack(
            [norm_simulated_ach, norm_simulated_da]
        )
        score = self._calc_score(norm_simulated)
        return score

def run_hps(
    args: Namespace, 
    hps_config: Dict,
    save_loc: str
): 
    """
    Args:
        args (Namespace)  : "resources/exp01.yaml" etc
        hps_config (Dict) : "hps_config" in "resources/exp01.yaml" etc
        save_loc (str)    : path to save "results/v001/(date)/(settings)/hps/result.csv(or .pkl)"
    Returns:
        best_params
    """
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    objective = Objective(args, hps_config["search_space"])
    study = optuna.create_study(
        sampler=sampler, 
        direction="minimize"
    )

    tmp_saver = TemporalResultSaver(save_loc)
    study.optimize(
        objective, 
        n_trials=hps_config["num_trials"], 
        timeout=hps_config["max_time"], 
        catch=(RuntimeError,), 
        callbacks=[tmp_saver.save_temporal_result]
    )

    # 探索後の最良値
    print(study.best_value)
    print(study.best_params)
    
    best_params = study.best_params

    # Save study record as csv.
    csv_name = save_loc + "/result.csv"
    df_hps = study.trials_dataframe()
    df_hps.to_csv(csv_name)

    # Save study as pickle.
    pkl_name = save_loc + "/result.pkl"
    with open(pkl_name, "wb") as fp:
        pickle.dump(study, fp)

    return best_params

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--config', 
        help='path to config file', 
        default="resources/exp01.yaml"
    )
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        run_config = yaml.safe_load(f)
    base_config = Namespace(**run_config["base"])
    hps_config = run_config["hps_config"]

    # Run and save.
    save_loc = os.path.join(
        config["data_path"]["save_loc"],
        get_timestamp()
    )
    os.makedirs(save_loc, exist_ok=True)

    best_params = run_hps(
        base_config, hps_config, save_loc)

    env = Environment(
        base_config.activation_da, 
        base_config.activation_ach, 
        best_params['diff_step'], 
        best_params['diff_rate'],
        best_params['d1r_delay'],
        best_params['d2r_delay'], 
        best_params['d1r_ach_efficacy'],
        best_params['d2r_daaxon_efficacy'], 
        best_params['d2r_ach_efficacy'], 
        base_config.da_split_type, 
        base_config.use_nAChR_at_da,
        base_config.use_d2_at_da,
        base_config.use_d1_at_ach,
        base_config.use_d2_at_ach,
        config["env_param"]["max_da"],
        base_config.seed
    )
    nstep = int(
        base_config.activation_length / config["env_param"]["sec_per_step"]
    )
    best_simulation = env.run(nstep)
    
    savename = save_loc + "/best_simulation.png"
    make_plot(
        best_simulation, 
        config["env_param"]["sensor_fps"], 
        config["env_param"]["sec_per_step"],
        save_loc, 
        base_config.seed
    )
    print(save_loc)
