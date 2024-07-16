from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.cells import AChNeuron, DopamineAxon

class Environment:

    def __init__(
        self, 
        activation_da: str, 
        activation_ach: str, 
        diffusion_step: int,
        diffusion_rate: float,
        d1r_delay: int,
        d2r_delay: int,
        d1r_ach_efficacy: float,
        d2r_daaxon_efficacy: float,
        d2r_ach_efficacy: float,
        da_split_type: str,
        use_nAChR_at_da: bool,
        use_d2_at_da: bool,
        use_d1_at_ach: bool,
        use_d2_at_ach: bool,
        max_da: str,
        max_ach: str,
        seed: int=1,
    ):
        """
        Args:
            activation (str)           : inputs given to Cholinergic neurons and Dopaminergic axons.
            diffusion_step             : the latency of released DA reaches the dopamine receptors (D1R or D2R) on the receving neurons.
            diffusion_rate             : the intensity of released DA around the receving neurons.
            d1r_delay (int)            : the latency of DA activates the neuronal activity by binding to D1R.
            d2r_delay (int)            : the latency of DA inhibits the neuronal activity by binding to D2R.
            d1r_ach_efficacy (float)   : the power of single DA input activates the neuronal activity by binding to D1R on Cholinergic neurons.
            d2r_daaxon_efficacy (float): the power of single DA input inhibits neuronal activity by binding to D2R on Dopaminergic axons.
            d2r_ach_efficacy (float)   : the power of single DA input inhibits neuronal activity by binding to D2R on Cholinergic neurons.
            da_split_type (str)        : "random bindng" of released DA to "D1R on ACh neurons": "D2R on ACh neurons": "D2R on DA axons" = converge to 1:1:2 ratio.
            max_da (str): 
            max_ach (str): 
            seed (int): 
        Returns:
            None
        """
        np.random.seed(seed)

        # Validate efficacy.
        nAChR_efficacy = int(use_nAChR_at_da)
        d2r_daaxon_efficacy = d2r_daaxon_efficacy * int(use_d2_at_da)
        d1r_ach_efficacy = d1r_ach_efficacy * int(use_d1_at_ach)
        d2r_ach_efficacy = d2r_ach_efficacy * int(use_d2_at_ach)

        # 
        self.max_da = float(max_da)
        self.max_ach = float(max_ach)

        # 
        total_delay = max(d1r_delay, d2r_delay) + diffusion_step
        self.remain_rate = 1 - diffusion_rate
        self.diffusion_step = diffusion_step
        self.burn_in = total_delay * 5

        self.da_split_type = da_split_type

        self.da_observed = []
        self.da_feedback = [0 for _ in range(diffusion_step)]

        self.da_axon = DopamineAxon(
            activation_da, 
            d2r_delay,
            d2r_daaxon_efficacy,
            nAChR_efficacy
        )
        self.ach_neuron = AChNeuron(
            activation_ach, 
            d1r_delay,
            d2r_delay,
            d1r_ach_efficacy, 
            d2r_ach_efficacy
        )
        
        self.ach_observed = []
        
        self.da_to_da_observed = []
        self.da_to_ach_neuron_d1r_observed = []
        self.da_to_ach_neuron_d2r_observed = []
        self.da_ratio_observed = []

    def _da_diffusion(self, da: int):
        """
        Mimics the latency of the released DA reaching the receving neurons

        Args:
            da (float)      : the amount of DA released from the Dopaminergic axons at a specific time point.

        Returns:
            da_delay (float): the amount of DA reached to the receving neurons at a specific time point.
        """
        self.da_feedback.append(da)
        da_delay = self.da_feedback.pop(0)
        self.da_feedback = list(
            map(lambda ach: ach * self.remain_rate, self.da_feedback)
        )
        assert len(self.da_feedback) == self.diffusion_step
        return da_delay

    def _da_split(self, da: float):
        """ 
        The ratio of released DA around the receving neurons, converges to 1:1:2 for D1R on ACh neurons: D2R on ACh neurons: D2R on DA axons.
            1:1 in ACh neurons: DA axons
            1:1 in D1R on ACh neurons: D2R on ACh neurons

        Args:
            da (float): the amount of DA released from Dopaminergic axons.

        Returns:
            da_to_da_axon (float)       : the amount of DA reaches the Dopaminergic axons.
            da_to_ach_neuron_d1r (float): the amount of DA reaches the D1R on Cholinergic neurons.
            da_to_ach_neuron_d2r (float): the amount of DA reaches the D2R on Cholinergic neurons.
        """
        if self.da_split_type == "random":
            ratio_to_da_axon = np.clip(np.random.randn() + 0.5, 0, 1)
            da_to_da_axon = da * ratio_to_da_axon
            da_to_ach_neuron = da * (1 - ratio_to_da_axon)

            ratio_to_ach_neuron_d1r = np.clip(np.random.randn() + 0.5, 0, 1)
            da_to_ach_neuron_d1r = da_to_ach_neuron * ratio_to_ach_neuron_d1r
            da_to_ach_neuron_d2r = da_to_ach_neuron * (1 - ratio_to_ach_neuron_d1r)
        else:
            raise NotImplementedError
        return da_to_da_axon, da_to_ach_neuron_d1r, da_to_ach_neuron_d2r

    def step(self):
        """ 
        Execute single step.
            "step" indicates the target frame (time) (200fps).
        Args:
            None
        Returns:
            None
        """
        self.da_axon.activation()
        self.ach_neuron.activation()
        ach = self.ach_neuron.release()
        assert ach < self.max_ach

        self.ach_observed.append(ach) 
        self.da_axon.receptor_nAChR(ach) 

        da = self.da_axon.release()
        assert da < self.max_da
        self.da_observed.append(da)

        # Get delayed observation.
        da = self._da_diffusion(da)
        
        # Randomly split DA to da_axon and ach_neuron.
        da_to_da_axon, da_to_ach_neuron_d1r, da_to_ach_neuron_d2r = self._da_split(da)

        # Observations.
        self.da_to_da_observed.append(da_to_da_axon)
        self.da_to_ach_neuron_d1r_observed.append(da_to_ach_neuron_d1r)
        self.da_to_ach_neuron_d2r_observed.append(da_to_ach_neuron_d2r)

        # DA to receptors.        
        self.da_axon.receptor_d2(da_to_da_axon)
        self.ach_neuron.receptor_d1(da_to_ach_neuron_d1r)
        self.ach_neuron.receptor_d2(da_to_ach_neuron_d2r)        

    def run(self, num_steps: int):
        """
        Args:
            num_steps (int)    : the target time range
        Returns:
            observations (Dict): Dictionary of observations.
                                    Each observation is a numpy array with length of `num_steps`.
        """
        for _ in tqdm(range(num_steps + self.burn_in)):
            self.step()
        
        observations = {
            "ach": self.ach_observed[self.burn_in:],
            "da": self.da_observed[self.burn_in:],
            "activation": self.ach_neuron.get_activation_record()[self.burn_in:],
            "da_to_da": self.da_to_da_observed[self.burn_in:],
            "da_to_ach_neuron_d2r": self.da_to_ach_neuron_d2r_observed[self.burn_in:],
            "da_to_ach_neuron_d1r": self.da_to_ach_neuron_d1r_observed[self.burn_in:],
        }
        return observations

if __name__ == "__main__":
   pass