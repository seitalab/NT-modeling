import os
from argparse import Namespace

import yaml
import pandas as pd

from run_hps import run_hps
from run_eval import load_hps_result, main as run_eval

from hps_visualization import plot_hps, main as hps_visualization

cfg_file = "config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

def settings2string(
    setting,
    activation
):
    """
    Args:
        settings (Tuple)      : (use_nAChR_at_da, use_d2_at_da, 
                                    use_d1_at_ach, use_d2_at_ach, target_label)
        activation (Tuple)    : (da_activation, ach_activation)
    Returns:
        setting_info (str)    : base string representing the target data
    """
    use_nAChR_at_da, use_d2_at_da, use_d1_at_ach, use_d2_at_ach, target_label = setting
    da_activation, ach_activation = activation

    receptor_info = f"receptor-{int(use_nAChR_at_da)}{int(use_d2_at_da)}"
    receptor_info += f"{int(use_d1_at_ach)}{int(use_d2_at_ach)}-label{target_label}"
    activation_info = f"da-{da_activation}_ach-{ach_activation}"
    setting_info = f"{receptor_info}_{activation_info}"
    return setting_info

def main(
    base_config_file, 
    setting, 
    activation, 
    saveroot, 
    seed: int=1
):
    """
    Args:
        base_config_file (str): "resources/exp01.yaml" etc
        settings (Tuple)      : (use_nAChR_at_da, use_d2_at_da, 
                                    use_d1_at_ach, use_d2_at_ach, target_label)
        activation (Tuple)    : (da_activation, ach_activation)
    Returns:
        score_str (str)       : the evaluation score (diff NTsim-NTreal)
    """
    use_nAChR_at_da, use_d2_at_da, use_d1_at_ach, use_d2_at_ach, target_label = setting
    da_activation, ach_activation = activation

    with open(base_config_file) as f:
        run_config = yaml.safe_load(f)
    base_config = Namespace(**run_config["base"])
    base_config.seed = seed
    base_config.activation_da = da_activation
    base_config.activation_ach = ach_activation
    base_config.use_nAChR_at_da = use_nAChR_at_da
    base_config.use_d2_at_da = use_d2_at_da
    base_config.use_d1_at_ach = use_d1_at_ach
    base_config.use_d2_at_ach = use_d2_at_ach
    base_config.target_label = target_label

    hps_config = run_config["hps_config"]
    # Run and save.
    save_loc_hps = os.path.join(
        saveroot, 
        settings2string(setting, activation),
        "hps"
    )
    save_loc_eval = os.path.join(
        saveroot, 
        settings2string(setting, activation),
        "eval"
    )
    os.makedirs(save_loc_hps, exist_ok=True)

    # Run hyperparameter search.  
    run_hps(base_config, hps_config, save_loc_hps)

    # Select best hyper parameter settting.
    base_config.hps = save_loc_hps
    args = load_hps_result(base_config)
    args.use_nAChR_at_da = use_nAChR_at_da
    args.use_d2_at_da = use_d2_at_da
    args.use_d1_at_ach = use_d1_at_ach
    args.use_d2_at_ach = use_d2_at_ach
    score_str = run_eval(args, save_loc_eval)
    
    hps_visualization(save_loc_hps)

    return score_str

def csv2settings(csvfile_path: str, target_label: int):
    """
    Args:
        csvfile_path (str): path to the resorces/trial.csv
        target_label (int): the target data 
                                0 = intromission only (IO), 
                                1 = intromission preceding ejaculation (IPE)
    Returns:
        settings (tuple)  : settings data about activation and the target data

    """
    df_settings = pd.read_csv(csvfile_path, index_col=0)
    settings = []
    for _, row in df_settings.iterrows():
        use_nAChR_at_da = row["use_nAChR_at_da"]
        use_d2_at_da = row["use_d2r_at_da"]
        use_d1_at_ach = row["use_d1r_at_ach"]
        use_d2_at_ach = row["use_d2r_at_ach"]
        setting = (
            use_nAChR_at_da, 
            use_d2_at_da, 
            use_d1_at_ach, 
            use_d2_at_ach, 
            target_label
        )
        da_activation = row["da_activation"]
        ach_activation = row["ach_activation"]
        activation = (da_activation, ach_activation)
        settings.append([setting, activation])
    return settings

if __name__ == "__main__":

    from argparse import ArgumentParser

    from src.utils import get_timestamp

    parser = ArgumentParser()

    parser.add_argument(
        '--config', 
        help='path to config file', 
        default="resources/exp01.yaml"
    )
    parser.add_argument(
        '--setting', 
        help='path to the receptor setting csv for hps visualization',
        default="resources/trial.csv"
    )
    parser.add_argument(
        '--label', 
        type=int,
        help='target label index 0 or 1',
        default=0
    )
  
    args = parser.parse_args()

    saveroot = os.path.join(
        config["data_path"]["save_loc"],
        get_timestamp()
    )
    

    settings = csv2settings(
        args.setting, 
        args.label
    )  
    trial_num = 0
    for setting, activation_pair in settings:
        trial_num += 1
        message = (
            f"Starting {trial_num}/{len(settings)} "
            f"({settings2string(setting, activation_pair)})"
        )

        score_str = main(
            args.config, 
            setting, 
            activation_pair, 
            saveroot,
        )
        message = f"--> Experiment done: {score_str}"
        