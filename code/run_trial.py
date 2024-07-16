import yaml

from src.simulator import Environment

cfg_file = "config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    import os
    import pickle
    import argparse
    
    from src.utils import make_plot_simple, get_timestamp

    # Prepare save loc.
    SAVEROOT = "/results/trials"
    pkl_save_loc = "/results/trials"

    # Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--simlen", type=int, default=120)
    parser.add_argument("--activation_da", type=str, default="const")
    parser.add_argument("--activation_ach", type=str, default="on-60")
    parser.add_argument("--diff_rate", type=float, default=0.05) # Rate for per step diffusion.
    parser.add_argument("--diff_step", type=int, default=50) 
    parser.add_argument("--d1r_delay", type=int, default=50)
    parser.add_argument("--d2r_delay", type=int, default=0)
    parser.add_argument("--d1r_ach_efficacy", type=float, default=0)
    parser.add_argument("--d2r_daaxon_efficacy", type=float, default=1)
    parser.add_argument("--d2r_ach_efficacy", type=float, default=1)
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    args.use_xr_at_da = True
    args.use_d2_at_da = True
    args.use_d1_at_ach = True
    args.use_d2_at_ach = True
    print(args)

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
        args.split, 
        args.use_xr_at_da,
        args.use_d2_at_da,
        args.use_d1_at_ach,
        args.use_d2_at_ach,        
        config["env_param"]["max_da"],
        config["env_param"]["max_ach"],
        args.seed
    )
    nstep = int(args.simlen / config["env_param"]["sec_per_step"])
    observations = env.run(nstep)

    save_loc = get_timestamp()
    save_dir = os.path.join(SAVEROOT, save_loc)
    os.makedirs(save_dir, exist_ok=True)
    make_plot_simple(
        observations, 
        config["env_param"]["sensor_fps"], 
        config["env_param"]["sec_per_step"],
        save_dir, 
        args.seed
    )
    # Save simulated.
    with open(pkl_save_loc + "/simulated.pkl", "wb") as fp:
        pickle.dump(observations, fp)
