from typing import List

from src import utils

''' Dopaminergic axon terminals around the vsNAc '''
class DopamineAxon:

    def __init__(
        self, 
        activation: str, 
        delay_d2: int,
        d2_efficacy: float,
        nAChR_efficacy: float,
    ):
        """
        Args:
            activation (str)      : constant 1 or 0 input, which represents the neuronal avcitivy of Dopaminergic axons
            delay_d2 (int)        : the latency DA inhibits the neuronal activity by binding to D2R.
            d2_efficacy (float)   : the power of single DA inhibits the neuronal activity by binding to D2R.
            nAChR_efficacy (float): the power of single ACh activates neuronal activity by binding to nAChR.
        Returns:
            None
        """
        self.release_prob_idx = 0
        self.activator = utils.Activator(activation)
        self.activation_record = []

        self.da_delay_d2 = [0 for _ in range(delay_d2)]
        self.d2_efficacy = d2_efficacy
        self.nAChR_efficacy = nAChR_efficacy

    def activation(self):
        """ 
        The amount of external activation on Dopaminergic axons, 
            reflecting spontaneous activity of Dopaminergic axons or stimulation by something other than ACh and DA.

        Args:
            None
        Returns:
            None
        """
        stimulus = self.activator.activation()
        self.activation_record.append(stimulus)
        self.release_prob_idx += stimulus        

    def receptor_nAChR(self, ach: float):
        """
        the powor of nAChR functions at the Dopaminergic axons, increases DA release.

        Args:
            ach (float): the amount of ACh that binds to nAChR on DA axons.
        Returns:
            None
        """
        self.release_prob_idx += (ach * self.nAChR_efficacy)

    def receptor_d2(self, da: float):
        """
        the power of D2R functions at the Dopaminergic axons, decreases DA release.

        Args:
            da (float): the amount of DA that binds to D2R on Dopaminergic axons.
        Returns:
            None
        """
        # Delayed da effect.
        self.da_delay_d2.append(da)
        da_delay = self.da_delay_d2.pop(0)
        
        release_prob_idx = self.release_prob_idx - da_delay * self.d2_efficacy
        self.release_prob_idx = utils.relu(release_prob_idx)

    def release(self):
        """
        the amount of DA released from Dopaminergic axons drived from its neuronal activity

        Args:
            None
        Returns:
            da (float): the amount of DA released from Dopaminergic axons.
        """
        return self.release_prob_idx

''' Cholinergic neurons in the vsNAc '''
class AChNeuron:

    def __init__(
        self, 
        activation: str, 
        delay_d1: int,
        delay_d2: int,
        d1_efficacy: float,
        d2_efficacy: float
    ):
        """
        Args:
            activation (str)    : constant 1 or 0 is added to the neuronal activity of Cholinergic neurons
            delay_d1 (int)      : the latency DA activates the neuronal activity by binding to D1R.
            delay_d2 (int)      : the latency DA inhibits the neuronal activity by binding to D2R.
            d1_efficacy (float) : the power of single DA activates neuronal activity by binding to D1R.
            d2_efficacy (float) : the power of single DA inhibits the neuronal activity by binding to D2R.
        Returns:
            None
        """
        self.activator = utils.Activator(activation)
        self.release_prob_idx = 1
        self.da_delay_d1 = [0 for _ in range(delay_d1)]
        self.da_delay_d2 = [0 for _ in range(delay_d2)]

        self.d1_efficacy = d1_efficacy
        self.d2_efficacy = d2_efficacy

        self.activation_record = []

    def activation(self):
        """
        The amount of external activation on Cholinergic neurons, 
            reflecting spontaneous activity of Cholinergic neurons or stimulation by something other than ACh and DA.
        
        Args:
            None
        Returns:
            None
        """
        stimulus = self.activator.activation()
        self.activation_record.append(stimulus)
        self.release_prob_idx += stimulus

    def receptor_d2(self, da: float):
        """
        the power of D2R functions at the Cholinergic neurons, decreases ACh release.

        Args:
            da (float): the amount of DA that binds to D2R on Cholinergic neurons.
        Returns:
            None
        """
        # Delayed da effect.
        self.da_delay_d2.append(da)
        da_delay = self.da_delay_d2.pop(0)

        release_prob_idx = self.release_prob_idx - da_delay * self.d2_efficacy
        self.release_prob_idx = utils.relu(release_prob_idx)

    def receptor_d1(self, da: float):
        """
        the power of D1R functions at the Cholinergic neurons, increases ACh release.
        
        Args:
            da (float): the amount of DA that binds to D1R on Cholinergic neurons.
        Returns:
            None
        """
        # Delayed da effect.
        self.da_delay_d1.append(da)
        da_delay = self.da_delay_d1.pop(0)

        release_prob_idx = self.release_prob_idx + da_delay * self.d1_efficacy
        self.release_prob_idx = utils.relu(release_prob_idx)

    def release(self): 
        """
        the amount of ACh released from Cholinergic neurons drived from its neuronal activity.

        Args:
            None
        Returns:
            ACh (float): the amount of ACh released from Cholinergic neurons.
        """
        return self.release_prob_idx

    def get_activation_record(self):
        """
        Args:
            None
        Returns:
            activation_record (List): 
        """
        return self.activation_record