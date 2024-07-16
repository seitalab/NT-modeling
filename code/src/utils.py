from datetime import datetime
from typing import Dict, List, Tuple, Union

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from sklearn import preprocessing

sns.set()

cfg_file = "./config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
EPS = 1e-10

def get_timestamp():
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

def relu(val: float):
    """
    Args:
        val (float): 
    Returns: 
        val
    """
    return max(0, val)

def split_da(
    da_val: float, 
    split_type: str
):
    """
    Split the released DA value.

    Args:
        da_val (float): 
        split_type (str): 
    Returns:
        da_values
            to_da_axon (float): 
            to_d2r_ach_neuron (float): 
            to_d1r_ach_neuron (float): 
            da_washout (float):
        da_ratio (float): 
    """
    if split_type == "random":
        ratio = np.random.dirichlet([1, 1, 1, 1])
        to_da_axon = da_val * ratio[0]
        to_d2r_ach_neuron = da_val * ratio[1]
        to_d1r_ach_neuron = da_val * ratio[2]
        da_wash = da_val * ratio[3]
    else:
        raise NotImplementedError

    return to_da_axon, to_d2r_ach_neuron, to_d1r_ach_neuron, da_wash

def make_plot(
    observations: Dict, 
    fps: int, 
    sec_per_step: float,
    save_dir: str, 
    seed: int
):
    """
    Args:
        observations (Dict) : Dictionary of observations.
            Each observation is a numpy array with length of `num_steps`.
        fps (int)           : Observation frequency of sensor [Count/sec].
        sec_per_step (float): Number of steps per seconds [sec/step].
        save_dir (str)      : the target directory to save the figures
        seed (int)          : the target seed
    Returns:
        None
    """
    width = int(len(observations["da"]) / int(1/sec_per_step))
    fig, (ax1, ax2, ax2a, ax3, ax4, ax5, ax6, ax7) =\
         plt.subplots(8, sharex=True, figsize=(width, 8))
    
    fig.suptitle(save_dir)
    da_observed = observations["da"]
    step_per_obs = int(1 / (fps * sec_per_step)) # [step / Count]

    b = np.ones(step_per_obs) / step_per_obs
    da_mv = np.convolve(da_observed, b, mode="full")
    
    da_sample = da_observed[::step_per_obs]
    xloc = np.arange(len(da_observed))[::step_per_obs]
    
    # ACh
    ax1.plot(observations["ach"], color='r', label='ACh')
    ax1.legend()
    # DA moving averaged
    ax2.plot(da_mv, color='b', label="DA (moving average)")
    ax2.legend()
    # DA sampled
    ax2a.plot(xloc, da_sample, color='b', label="DA (sampled)")
    ax2a.legend()
    # DA (raw)
    ax3.plot(da_observed, color='b', label='DA (raw)')
    ax3.legend()
    # D2R@DAT
    ax4.plot(observations["da_to_da"], color='c', label='DA binding to D2R@DAT')
    ax4.legend()
    # Stimulation
    ax7.plot(observations["activation"])
    x = np.arange(len(da_observed)+1)[::int(1/sec_per_step)]
    label = np.arange(len(x))
    plt.xticks(x, label)
    plt.xlabel("[sec]")
    plt.savefig(save_dir + f"/seed{seed:02d}.png")

def make_plot_simple(
    observations: Dict, 
    sec_per_step: float,
    save_dir: str, 
    seed: int
):
    """
    Args:
        observations (Dict) : Dictionary of observations.
            Each observation is a numpy array with length of `num_steps`.
        sec_per_step (float): Number of steps per seconds [sec/step].
        save_dir (str)      : the target directory to save the figures
        seed (int)          : the target seed
    Returns:
        None
    """
    width = int(len(observations["da"]) / int(1/sec_per_step))
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(width, 3))
    
    fig.suptitle(save_dir)
    da_observed = observations["da"]
    
    # ACh
    ax1.plot(observations["ach"], color='r', label='ACh')
    ax1.legend()

    # DA (raw)
    ax2.plot(da_observed, color='b', label='DA (raw)')
    ax2.legend()

    x = np.arange(len(da_observed)+1)[::int(1/sec_per_step)]
    label = np.arange(len(x))
    plt.xticks(x, label)
    plt.xlabel("[sec]")
    plt.savefig(save_dir + f"/seed{seed:02d}.png")


def process_simulated_signal(
    simulated_signals: Dict, 
    target_signal: str,
): 
    """
    Args:
        simulated_signals (Dict): 
        target_signal (str)     : "da" or "ach"  
    Returns:
        norm_simulated (np.ndarray): normalized signal
        validity (bool): whether or not the NTsim is saturated
    """
    simulated_signal = simulated_signals[target_signal]
    # Process simulated signal
    sensor_fps = config["env_param"]["sensor_fps"]
    sec_per_step = config["env_param"]["sec_per_step"]
    step_per_obs = int(1 / (sensor_fps * sec_per_step))
    simulated_signal = simulated_signal[::step_per_obs]

    simulated_signal = np.clip(simulated_signal, -1e10, 1e10)
    norm_simulated = preprocessing.minmax_scale(simulated_signal)
    validity = np.abs(simulated_signal).max() > float(config["env_param"][f"max_{target_signal}"])
    return norm_simulated, validity

# Score calc functions.
def calc_diff_score(
    real_grab: List, 
    normalized_simulated_signal: np.ndarray,
    signal_idx: int,
    calc_mean: bool=True
):
    """
    Args:
        real_grab (List)                 : List of arrays (num_samples,)
        normalized_simulated (np.ndarray): Array of signals (2, len_simulation).
    Returns:
        best_scores(np.ndarray): 
    """
    best_scores = []
    for i in range(len(real_grab)):
        ref_length = real_grab[i].shape[-1]
        simulated_length = normalized_simulated_signal.shape[-1]
        score = np.array([
            np.abs(
                normalized_simulated_signal[signal_idx][j:j+ref_length] - real_grab[i][signal_idx]
            ).mean() for j in range(simulated_length - ref_length)
        ])
        if calc_mean:
            best_scores.append(score.min())
        else:
            best_scores.append(score)

    if calc_mean:
        best_scores = np.array(best_scores)
        return best_scores.mean()
    else:
        return best_scores

def calc_fft_diff_score(
    real_grab: List, 
    normalized_simulated_signal: np.ndarray,
    signal_idx: int,
    calc_mean: bool=True
):
    """
    Args:
        real_grab (List)                        : List of arrays (num_samples,)
        normalized_simulated_signal (np.ndarray): Array of signals (2, len_simulation).
        signal_idx (int)                        : signal index 
    Returns:
        feat(): 
    """
    def calc_feat(signal: np.ndarray):
        """
        Args:
            signal (np.ndarray): 
        Returns:
            feat (np.ndarray)  : 
        """
        nyquist_freq = int(len(signal)/2)+1
        
        signal_fft = np.fft.fft(signal)
        fft_abs = np.abs(signal_fft)
        fft_amp = fft_abs / len(signal) * 2
        
        fft_amp = fft_amp / 2
        feat = fft_amp[:nyquist_freq]
        feat /= (np.linalg.norm(feat) + EPS)
        return feat

    def calc_diff(sim, real):
        """
        Args:
            sim (np.ndarray) : Simulated signal (NTsim)
            real (np.ndarray): Observed signal from in vivo experiment (NTreal)
        Returns:
            diff ()          : difference between NTsim and NTreal
        """
        feat_exp = calc_feat(real)
        feat_sim = calc_feat(sim)
        diff = (np.abs(feat_exp - feat_sim)).sum()
        return diff
    
    best_scores = []
    for i in range(len(real_grab)):
        ref_length = real_grab[i].shape[-1]
        simulated_length = normalized_simulated_signal.shape[-1]
        score = np.array([
            calc_diff(
                normalized_simulated_signal[signal_idx][j:j+ref_length], 
                real_grab[i][signal_idx]
            ) for j in range(simulated_length - ref_length)
        ])
        if calc_mean:
            best_scores.append(score.min())
        else:
            best_scores.append(score)
    if calc_mean:
        best_scores = np.array(best_scores)
        return best_scores.mean()
    else:
        return best_scores

def normalize_score(scores: np.ndarray, weight: float, metric: str):
    """
    Args: 
        scores (np.ndarray): 
        weight (float)     : Weight applied to scores.
        metric (str)       : Used to select denominator when normalizing.
    Returns:
        normalized_score (np.ndarray): 
    """
    if metric == "diff":
        denominator_type = "mean_real_diff"
    elif metric == "fft_diff":
        denominator_type = "mean_real_fftdiff"
    else:
        raise NotImplementedError
    normalized_scores = weight * scores / config["env_param"][denominator_type]
    return normalized_scores

# Activation
class Activator:

    def __init__(self, activation_type: str):
        """
        Args:
            activation_type (str): "const" or "none" is given to the target neurons.
           
        Returns:

        """
        self.activation_type = activation_type

    def activation(self):
        """
        Args:
            None
        Returns:
            stimulus (float)     : "const (=1)" or "none (=0)"
        """
        if self.activation_type == "const":
            stimulus = 1
        elif self.activation_type == "none":
            stimulus = 0
        else:
            raise NotImplementedError
        return stimulus

if __name__ == "__main__":

    pass