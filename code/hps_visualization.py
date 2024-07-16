import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from argparse import Namespace
import yaml
from src.utils import (get_timestamp)
import os

plt.rc('pdf', fonttype=42)
plt.rcParams["font.size"] =30
plt.rcParams['font.family'] = "Arial"
plt.rcParams['axes.linewidth'] =5
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams["xtick.major.size"] =10    
plt.rcParams["ytick.major.size"] =10    

plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =0.
plt.rcParams['axes.linewidth'] =5
    
def plot_hps(hps,save_loc):
    """
    Args:
        hps(str)      : the target file of the searched hps in /results
        saver_loc(str): the target direcotry to save .png

    Returns:
    """
    ######################################################################################
    #calculate RS (sec)
    hps["params_d1r_delay_sec"] = hps["params_d1r_delay"]/200
    hps["params_d2r_delay_sec"] = hps["params_d2r_delay"]/200
    hps["params_diff_step_sec"] = hps["params_diff_step"]/200

    y_lim = [0,5]
    ######################################################################################
    #save diff vs RS(sec) 
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    hps.plot.scatter(y = "value",
    x = "params_d1r_delay_sec",
    label = "D1R",
    c = "goldenrod",alpha = 0.5,
    ax = ax1)

    hps.plot.scatter(y = "value",
    x = "params_d2r_delay_sec",
    c = "dodgerblue",
    label = "D2R",
    alpha = 0.5,
    ax = ax1)
    ax1.semilogx()
    ax1.set_ylabel("score")
    ax1.set_xlabel("Delay")
    ax1.set_facecolor((1,1,1,0))
    [ax1.spines[side].set_visible(False) for side in ["right", "top"]]
    plt.tight_layout()
    fig.show()
    fig.savefig(save_loc+"/log-scaled RS_sec.png")



    ######################################################################################
    #save diff vs log-scaled RS(sec) focusing on the converged RS

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    hps.plot.scatter(y = "value",
    x = "params_d1r_delay_sec",
    label = "D1R",
    c = "orange",alpha = 0.5,
    ax = ax1)

    hps.plot.scatter(y = "value",
    x = "params_d2r_delay_sec",
    c = "navy",
    label = "D2R",
    alpha = 0.5,
    ax = ax1)
    ax1.set_ylim(y_lim)
    ax1.semilogx()
    ax1.set_ylabel("Diff (NTm-NTfp)")
    ax1.set_xlabel("sec (RS)")
    ax1.set_facecolor((1,1,1,0))
    [ax1.spines[side].set_visible(False) for side in ["right", "top"]]
    plt.tight_layout()
    fig.show()
    fig.savefig(save_loc + "/log-scaled RS_sec zoom.png")



    ######################################################################################
    #save diff vs RE(original) 

    fig = plt.figure(figsize=(6, 4))
    ax2 = fig.add_subplot(1, 1, 1)

    hps.plot.scatter(y = "value",
    x = "params_d1r_ach_efficacy",
    c = "goldenrod",
    label = "D1R @ChAT",
    alpha = 0.5,
    ax = ax2)

    hps.plot.scatter(y = "value",
    x = "params_d2r_daaxon_efficacy",
    c = "mediumseagreen",
    label = "D2R @DAaxon",
    alpha = 0.5,
    ax = ax2)

    hps.plot.scatter(y = "value",
    x = "params_d2r_ach_efficacy",
    c = "dodgerblue",
    label = "D2R @ChAT",
    alpha = 0.5,
    ax = ax2)
    ax2.semilogx()
    ax2.set_ylabel("score")
    ax2.set_xlabel("Efficacy")
    ax2.set_facecolor((1,1,1,0))
    [ax2.spines[side].set_visible(False) for side in ["right", "top"]]
    plt.tight_layout()
    fig.show()
    fig.savefig(save_loc+ "/log-scaled RE.png")



    ######################################################################################
    #save diff vs log-scaled RE focusing on the converged RE

    fig = plt.figure(figsize=(8, 8))
    ax2 = fig.add_subplot(1, 1, 1)

    hps.plot.scatter(y = "value",
    x = "params_d1r_ach_efficacy",
    c = "orange",
    label = "D1R @ChAT",
    alpha = 0.5,
    ax = ax2)

    hps.plot.scatter(y = "value",
    x = "params_d2r_ach_efficacy",
    c = "dodgerblue",
    label = "D2R @ChAT",
    alpha = 0.5,
    ax = ax2)

    hps.plot.scatter(y = "value",
    x = "params_d2r_daaxon_efficacy",
    c = "darkturquoise",
    marker="D",
    label = "D2R @DAaxon",
    alpha = 0.5,
    ax = ax2)
    ax2.semilogx()
    ax2.set_ylim(y_lim)
    ax2.set_ylabel("Diff (NTm-NTfp)")
    ax2.set_xlabel("Efficacy (RE)")
    ax2.set_facecolor((1,1,1,0))
    [ax2.spines[side].set_visible(False) for side in ["right", "top"]]
    plt.tight_layout()
    fig.show()
    fig.savefig(save_loc + "/log-scaled RE zoom.png")



    ######################################################################################
    #save diff vs DA diffusion rate

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    hps.plot.scatter(y = "value",
    x = "params_diff_rate",
    label = "DA diffusion rate",
    c = "Red",alpha = 0.5,
    ax = ax1)
    ax1.set_ylabel("score")
    ax1.set_xlabel("rate")
    ax1.set_facecolor((1,1,1,0))
    [ax1.spines[side].set_visible(False) for side in ["right", "top"]]
    plt.tight_layout()
    fig.show()
    fig.savefig(save_loc + "/DA diffusiton rate.png")



    ######################################################################################
    #save diff vs DA diffusion step

    fig = plt.figure(figsize=(6, 4))
    ax2 = fig.add_subplot(1, 1, 1)

    hps.plot.scatter(y = "value",
    x = "params_diff_step_sec",
    c = "Red",
    label = "DA diffusion sec",
    alpha = 0.5,
    ax = ax2)
    ax2.set_ylabel("score")
    ax2.set_xlabel("step")
    ax2.set_facecolor((1,1,1,0))
    [ax2.spines[side].set_visible(False) for side in ["right", "top"]]
    plt.tight_layout()
    fig.show()
    fig.savefig(save_loc + "/DA diffusiton step sec.png")



    ######################################################################################
    #save diff vs DA diffusion step(sec) focusing on the converged DA diffusion step

    fig = plt.figure(figsize=(6, 4))
    ax2 = fig.add_subplot(1, 1, 1)

    hps.plot.scatter(y = "value",
    x = "params_diff_step_sec",
    c = "Red",
    label = "DA diffusion sec",
    alpha = 0.5,
    ax = ax2)
    ax2.set_ylim(y_lim)
    ax2.set_xlim([0,0.6])
    ax2.set_ylabel("score")
    ax2.set_xlabel("sec")
    ax2.set_facecolor((1,1,1,0))
    [ax2.spines[side].set_visible(False) for side in ["right", "top"]]
    plt.tight_layout()
    fig.show()
    fig.savefig(save_loc + "/DA diffusiton step sec zoom.png")

def main(save_loc_hps):
    """
    Args:
        save_loc_hps(str) : the path of the target result.csv
        saver_loc_png(str): the target direcotry to save .png

    Returns:
    """
    result_csv = os.path.join(save_loc_hps, "result.csv")
    print(result_csv)

    hps = pd.read_csv(result_csv)
    plot_hps(hps,save_loc_hps)

if __name__ == "__main__":

    main(save_loc_hps)
    print("hps visualization done")