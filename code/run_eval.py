import os
import yaml
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn import preprocessing

from src import utils
from src.data import prepare_realgrab_data
from src.simulator import Environment

# sns.set()
sns.set(
    rc={'axes.facecolor':'white', 'figure.facecolor':'white'}
)

cfg_file = "config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
EPS = 1e-10

def load_hps_result(args, load_non_efficacy_params: bool=True):
    """
    Args:

    Returns:

    """
    result_csv = os.path.join(args.hps, "result.csv")
    df_hps = pd.read_csv(result_csv)
    scores = df_hps.loc[:, "value"].values
    best_idx = np.argmin(scores)
    best_row = df_hps.iloc[best_idx]

    # Update param.
    args.d1r_ach_efficacy = best_row["params_d1r_ach_efficacy"]
    args.d2r_daaxon_efficacy = best_row["params_d2r_daaxon_efficacy"]
    args.d2r_ach_efficacy = best_row["params_d2r_ach_efficacy"]
    if load_non_efficacy_params:
        args.d1r_delay = best_row["params_d1r_delay"]
        args.d2r_delay = best_row["params_d2r_delay"]
        args.diff_rate = best_row["params_diff_rate"]
        args.diff_step = best_row["params_diff_step"]
    return args

def make_best_match_fig(
    real_grab: np.ndarray, 
    sim_grab: np.ndarray, 
    scores: np.ndarray, 
    savename: str
):
    """
    Args:
        real_grab (np.ndarray): Array of shape (2, signal_length).
        sim_grab (np.ndarray) : Array of shape (2, simulation_length).
        scores (np.ndarray)   : Array of shape (simulation_length - signal_length,).
        savename (str)        : file name of the best_match/*.png

    Returns:
        None
    """
    sim_length = sim_grab.shape[-1]
    real_grab_length = real_grab.shape[-1]
    sensor_fps = config["env_param"]["sensor_fps"]

    data_duration = int(sim_length / sensor_fps) # seconds
    timestamps = np.arange(sim_length) / sensor_fps

    start = np.argmin(scores)
    end = int(start + real_grab_length)

    fig, (ax1, ax2) =\
        plt.subplots(2, sharex=True, figsize=(data_duration*2, 6))
    ax1.plot(timestamps, sim_grab[0], color='r', label='X')
    ax1.plot(timestamps[start:end], real_grab[0], label="real grab")
    ax2.plot(timestamps, sim_grab[1], color='c', label='DA')
    ax2.plot(timestamps[start:end], real_grab[1], label="real grab")
    plt.xlabel("Pseudo time")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.savefig(savename)
    plt.clf()
    plt.close()    

def make_best_match_fig_zoom(
    real_grab: np.ndarray, 
    sim_grab: np.ndarray, 
    scores: np.ndarray, 
    savename: str
):
    """
    Args:
        real_grab (np.ndarray): Array of shape (2, signal_length).
        sim_grab (np.ndarray) : Array of shape (2, simulation_length).
        scores (np.ndarray)   : Array of shape (simulation_length - signal_length,).
        savename (str)        : file name of the best_match_zoom/*.png

    Returns:
        None
    """
    sim_length = sim_grab.shape[-1]
    real_grab_length = real_grab.shape[-1]
    sensor_fps = config["env_param"]["sensor_fps"]

    data_duration = int(real_grab_length / sensor_fps) # seconds
    timestamps = np.arange(real_grab_length) / sensor_fps

    start = np.argmin(scores)
    end = int(start + real_grab_length)
    sim_grab_zoom = sim_grab[:, start:end]
    sim_grab_zoom = (sim_grab_zoom - sim_grab_zoom.min()) / (sim_grab_zoom.max() - sim_grab_zoom.min() + EPS)

    fig, (ax1, ax2) =\
        plt.subplots(2, sharex=True, figsize=(data_duration*4, 12))

    ax1.spines["left"].set_edgecolor("black")
    ax1.spines["bottom"].set_edgecolor("black")
    ax2.spines["left"].set_edgecolor("black")
    ax2.spines["bottom"].set_edgecolor("black")

    ax1.plot(timestamps, sim_grab_zoom[0], linestyle="solid", color='#609DAD', label='ACh (sim)')
    ax1.plot(timestamps, real_grab[0], linestyle="dashed", color="#65A1F0", label="ACh (exp)")
    ax2.plot(timestamps, sim_grab_zoom[1], linestyle="solid", color='#CC7688', label='DA (sim)')
    ax2.plot(timestamps, real_grab[1],  linestyle="dashed", color='#A9188B', label='DA (exp)')
    ax1.legend()
    ax2.legend()

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True
    ) 

    plt.savefig(savename, bbox_inches='tight')
    plt.clf()
    plt.close()    

def make_score_plot(
    scores: np.ndarray, 
    score_type: str,
    savename: str
):
    """ 
    Args:
        scores (np.ndarray) : the score (diff NTsim-NTreal)
        score_type (str)    : Score method
        savename (str)      : file name of the score_plot/*.png

    Returns:
        None
    """
    sensor_fps = config["env_param"]["sensor_fps"]

    data_duration = int(len(scores) / sensor_fps) # seconds
    timestamps = np.arange(len(scores)) / sensor_fps

    plt.figure(figsize=(data_duration*2, 4))
    plt.plot(timestamps[:len(scores)], scores, label=f"{score_type} score")
    plt.xlabel("Pseudo time")
    plt.ylabel(f"{score_type}")
    plt.legend()
    plt.savefig(savename)
    plt.clf()
    plt.close()

def make_score_histogram(
    best_scores: List,
    scoretype: str,
    savename: str
):
    """ plot histogram
    Args:
        best_scores (float) : the best score (diff NTsim-NTreal)
        score_type (str)    : Score method
        savename (str)      : file name of the histogram.png

    Returns:
        None
    """
    plt.hist(best_scores, color="c")
    plt.title(f"{scoretype}")
    plt.savefig(savename)
    plt.clf()
    plt.close()

def calc_score_and_save_fig(
    real_grabs: List, 
    sim_grab: np.ndarray,
    score_type: str,
    save_loc: str,
    lambda_: float=0.5
):
    """
    Args:
        real_grabs (list)    : NTreal
        sim_grab (np.ndarray): NTsim
        score_type (str)     : Score method (difference or FFT difference)
        save_loc (str)       : path to save 
        lambda_ (float)      : ratio to mix the score methods, difference and FFT difference

    Returns:
        best_scores (float)  : the best score (diff NTsim-NTreal)
    """
    # Calculate scores.
    if score_type == "diff":
        scores_ach = utils.calc_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=0)
        scores_da = utils.calc_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=1)
    elif score_type == "fft_diff":
        scores_ach = utils.calc_fft_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=0)
        scores_da = utils.calc_fft_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=1)
    elif score_type == "mix":
        scores_ach_d = utils.calc_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=0)
        scores_da_d = utils.calc_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=1)
        scores_ach_f = utils.calc_fft_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=0)
        scores_da_f = utils.calc_fft_diff_score(
            real_grabs, sim_grab, calc_mean=False, signal_idx=1)
    else:
        raise NotImplementedError

    best_scores = []
    os.makedirs(os.path.join(save_loc, "bestmatch"), exist_ok=True)
    os.makedirs(os.path.join(save_loc, "bestmatch_zoom"), exist_ok=True)
    os.makedirs(os.path.join(save_loc, "score_plot"), exist_ok=True)
    for idx in tqdm(range(len(real_grabs))):
        # Temporal solution (221107)
        if score_type != "mix":
            score = (scores_ach[idx] + scores_da[idx]) / 2
        else:
            # Sum up Nx and Da score for each metric
            scores_d = scores_ach_d[idx] + scores_da_d[idx]
            scores_f = scores_ach_f[idx] + scores_da_f[idx]

            # Normalize and calc sum.
            scores_d_norm = utils.normalize_score(scores_d, lambda_, "diff")
            scores_f_norm = utils.normalize_score(scores_f, 1-lambda_, "fft_diff")

            score = (scores_d_norm + scores_f_norm) / 2

        filename = f"id{idx+1:02d}_{score_type}.png"
        savename = os.path.join(save_loc, "bestmatch", filename)
        make_best_match_fig(
            real_grabs[idx], sim_grab, score, savename)
        
        filename = f"id{idx+1:02d}_{score_type}.png"
        savename = os.path.join(save_loc, "bestmatch_zoom", filename)
        make_best_match_fig_zoom(
            real_grabs[idx], sim_grab, score, savename)
        
        filename = f"id{idx+1:02d}_{score_type}.png"
        savename = os.path.join(save_loc, "score_plot", filename)
        make_score_plot(score, score_type, savename)

        best_scores.append(score.min())

    # Plot best score per real grab.
    filename = f"hist_{score_type}.png"
    savename = os.path.join(save_loc, filename)  
    make_score_histogram(best_scores, score_type, savename)

    return best_scores

def main(args, save_loc):
    """
    Args:
        args (Namespace)       : the target hps
        save_loc (str)         : path to save "scores.csv"
    Returns:
        mean_score_string (str): the evaluation score (diff NTsim-NTreal)
    """
    # Prepare environment and run.
    env = Environment(
        args.activation_da, 
        args.activation_ach, 
        args.diff_step,
        args.diff_rate,
        args.d1r_delay, 
        args.d2r_delay, 
        args.d1r_ach_efficacy,
        args.d2r_daaxon_efficacy,
        args.d2r_ach_efficacy,
        args.da_split_type, 
        args.use_nAChR_at_da,
        args.use_d2_at_da,
        args.use_d1_at_ach,
        args.use_d2_at_ach,          
        config["env_param"]["max_da"],
        config["env_param"]["max_ach"],
        args.seed
    )
    nstep = int(args.activation_length / config["env_param"]["sec_per_step"])
    print("Running simulation ...")
    observations = env.run(nstep)
    sim_signal_norm_da, _ =\
        utils.process_simulated_signal(observations, "da")
    sim_signal_norm_ach, _ =\
        utils.process_simulated_signal(observations, "ach")
    sim_signal_norm = np.stack(
        [sim_signal_norm_ach, sim_signal_norm_da]
    )

    # Prepare real grab signals.
    print("Loading real grab data ...")
    real_grabs = prepare_realgrab_data(args.target_label)

    # Calculate scores.
    print("Calculating score `diff` ...")
    score_diff = calc_score_and_save_fig(
        real_grabs, sim_signal_norm, "diff", save_loc)
    print("Calculating score `fft_diff` ...")
    score_fft = calc_score_and_save_fig(
        real_grabs, sim_signal_norm, "fft_diff", save_loc)
    print("Calculating score mix diff ...")
    score_mix = calc_score_and_save_fig(
        real_grabs, sim_signal_norm, "mix", save_loc, lambda_=args.lambda_) 

    # Summarize.
    df_scores = pd.DataFrame(
        np.array([score_diff, score_fft, score_mix]).T, 
        columns=["diff", "fft_diff", "mix"],
        index=np.arange(len(real_grabs))+1
    ).T
    df_scores["mean"] = df_scores.mean(axis=1)
    df_scores = df_scores.round(4).T
    filename = f"scores.csv"
    savename = os.path.join(save_loc, filename)
    df_scores.to_csv(savename)
    mean_score_string = f"diff: {df_scores.loc['mean', 'diff']:.3f}\n"
    mean_score_string += f"fft_diff: {df_scores.loc['mean', 'fft_diff']:.3f}\n"
    mean_score_string += f"mix: {df_scores.loc['mean', 'mix']:.3f}"
    return mean_score_string

if __name__ == "__main__":
    import argparse
    
    # Prepare save loc.
    SAVEROOT = "./dump"
    pkl_save_loc = "./workspace"

    timestamp = utils.get_timestamp()
    save_loc = os.path.join(SAVEROOT, timestamp)

    # Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--hps", type=str, default=None)
    parser.add_argument("--activation_length", type=int, default=120)
    parser.add_argument("--activation_da", type=str, default="none")
    parser.add_argument("--activation_ach", type=str, default="const")
    parser.add_argument("--diff_rate", type=float, default=0.05) # Rate for per step diffusion.
    parser.add_argument("--diff_step", type=int, default=50) 
    parser.add_argument("--d1r_delay", type=int, default=50)
    parser.add_argument("--d2r_delay", type=int, default=0)
    parser.add_argument("--d1r_ach_efficacy", type=float, default=0)
    parser.add_argument("--d2r_daaxon_efficacy", type=float, default=1)
    parser.add_argument("--d2r_ach_efficacy", type=float, default=1)
    parser.add_argument("--lambda_", type=float, default=0.5)
    parser.add_argument("--da_split_type", type=str, default="random")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    args.use_nAChR_at_da = True
    args.use_d2_at_da = True
    args.use_d1_at_ach = True
    args.use_d2_at_ach = True

    if args.hps is not None:
        args = load_hps_result(args)
    main(args, save_loc)
