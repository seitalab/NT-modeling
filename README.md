# README

## About

This repository (repo) contains the codes for a manuscript titled ***Sequential Transitions of Male Sexual Behaviors Driven by Dual Acetylcholine-Dopamine Dynamics*** submitted to *Cell press*.
For reproducibility, the codes and data in this repo can reproduce all the reported results in our paper.

### "Elucidation of the neuronal mechanism of the release patten of neurotransmitetrs
  (NT) in the brain is crucial for understanding of the precise neural basis. Here, we reveal what kind of neurons and receptors are important for the specific NT release by computational modeling based on the real NT and neuronal activity patterns in the Nucleaus accumbuns. 
  As we observed similar dynamics in NT release and neuronal activity through GRAB and GCaMP fiber photometry imaging (GRAB<sub>DA2m</sub> and GCaMP6s of dopaminergic axons in the vsNAc, GRAB<sub>ACh3.0</sub> and GCaMP6s of cholinergic neurons in the vsNAc), we hypothesized that (1) the neuronal activity of a ensemble of neurons reflects the amount of NT released; (2) the actual GCaMP recordings can predict the release probability of the NT. 
  Based on these hypotheses, we developed a computational model to estimate the cellular environment in which neurons and receptors are involved in generating dual ACh-DA rhythms during intromission."



# Getting Started

Refering to the <span style="color: red; ">`code/run`</span>, execute python file such as <span style="color: red; ">`experiment.py`</span> in line 22.

## Folder Structure
- <span style="color: green; ">`code/resources`</span> : definition of the hps space
  - <span style="color: red; ">`exp01.yaml`</span> : To define the hps space
  - <span style="color: red; ">`trial.csv`</span> : To define $Activation$ pattern ($ChAT_{sim}^{vsNAc}$ / $DA_{sim}^{VTA→vsNAc}$ = on/on) and the receptor distribution (all 4 receptors are distributed) to visualize hps
  - <span style="color: red; ">`trial0.csv`</span> : To define $Activation$ pattern and the receptor distribution for statistics

- <span style="color: green; ">`code/src`</span> : basic codes
  - <span style="color: red; ">`cells.py`</span> :  To define each element involved in the neurotransmitter (NT) release.
  - <span style="color: red; ">`data.py`</span> :  For loading data.
  - <span style="color: red; ">`simulator.py`</span> :  For simulating $NT_{sim}$ release.
  - <span style="color: red; ">`utils.py`</span> :  To make the code easier to follow, all basic codes that are not directly relevant to the analysis logics are inside utils.py.

- <span style="color: green; ">`code`</span> : source codes for running analysis
  - <span style="color: red; ">`config.yaml`</span> : To define the data path and basic information of $NT_{real}$ and $NT_{sim}$
  - <span style="color: red; ">`experiment.py`</span> : For running the simulation experiment.

  - <span style="color: red; ">`hps_visualization.py`</span> :  For visualization of $Diff(NT_{sim}, NT_{real})$ and hps.
  - <span style="color: red; ">`run`</span> :  To perform modeling.
      - It takes 3 hours for the simulation for one setting. You can change the search time in <span style="color: red; ">`exp01.yaml`</span>.

      ```
      hps_config:
        max_time: 10800 #(sec)
      ```

      - You can define the required auguments (in <span style="color: green; ">`code/resources`</span>) in <span style="color: red; ">`experiment.py`</span> by [argparse](https://docs.python.org/3/library/argparse.html) in <span style="color: red; ">`run`</span> .
      ```
      # for hps search for statistics using trial0.csv
      python experiment.py --setting resources/trial0.csv
      ```
  - <span style="color: red; ">`run_trial.py`</span> :  For trial run to check if the simulator works.
  - <span style="color: red; ">`run_hps.py`</span> :  For executing single hyperparameter search (hps).
  - <span style="color: red; ">`run_eval.py`</span> :  For evaluating specific simulation setting.


- <span style="color: green; ">`data`</span> : typical rhythmic signals of in vivo GRAB<sub>ACh3.0</sub> and rGRAB<sub>DA2m</sub> during intromission ($NT_{real}$)

- <span style="color: green; ">`results`</span> : analysis results including hps and the simulated signal ($NT_{sim}$)
  -  <span style="color: green; ">`eval`</span> : Comparison of $NT_{sim}$ and $NT_{real}$
  -  <span style="color: green; ">`hps`</span> : $Diff(NT_{sim}, NT_{real})$ and hps.



## Prerequisites
- Required python libraries is listed under <span style="color: green; ">`code/requirements.txt`</span>.


## License
- Distributed under the MIT License. See `LICENSE` for more information.



---



## Mechanism
- In this model, we set neuronal assembly ($ChAT_{sim}^{vsNAc}$ as <span style="color: red; ">`class AChNeuron`</span> and $DA_{sim}^{VTA→vsNAc}$ as <span style="color: red; ">`class DopamineAxon`</span> in the <span style="color: green; ">`code/src/cells.py`</span>), the internal activity of neurons (release probability measurement, $RPM$ as <span style="color: red; ">`release_prob_idx`</span> in the <span style="color: green; ">`code/src/cells.py`</span>), NT release ($ACh_{sim}$ and $DA_{sim}$) based on the $RPM$ (calculated by <span style="color: red; ">`def release`</span> in each <span style="color: red; ">`class AChNeuron`</span> and <span style="color: red; ">`class DopamineAxon`</span> in the <span style="color: green; ">`code/src/cells.py`</span>), receptors expressed on neurons (D1R, D2R, and nAChR), and the extracellular environment in which the NT can be released. 

- To mimic the change of the amount of the extracellular NT over time, in this model, we assumed the three steps of cycle (as <span style="color: red; ">`def step`</span> in each <span style="color: red; ">`class Environment`</span> in the <span style="color: green; ">`code/src/simulator.py`</span>) : 
  - (i) the activation of target neurons by the external stimuli ($Activation$ as <span style="color: red; ">`def Activation`</span> in each <span style="color: red; ">`class AChNeuron`</span> and <span style="color: red; ">`class DopamineAxon`</span> in the <span style="color: green; ">`code/src/cells.py`</span> and <span style="color: red; ">`class Activator`</span> in the <span style="color: green; ">`code/src/utils.py`</span>); 
  - (ii) NT released from the target neurons (as <span style="color: red; ">`def release`</span> in each <span style="color: red; ">`class AChNeuron`</span> and <span style="color: red; ">`class DopamineAxon`</span> in the <span style="color: green; ">`code/src/cells.py`</span>); 
  - (iii) the effect of the receptor by binding NT to the receptor on the target neurons (as <span style="color: red; ">`def receptor_nAChR`</span>, <span style="color: red; ">`def receptor_d1`</span>, and <span style="color: red; ">`def receptor_d2`</span> in <span style="color: red; ">`class AChNeuron`</span> and <span style="color: red; ">`class DopamineAxon`</span> in the <span style="color: green; ">`code/src/cells.py`</span>). 


- These cycles were repeated a certain number of times and, during this period, we recorded the $RPM$ and amount of  $NT_{sim}$ released from the neurons which was determined based on the $RPM$ value. Through this model, we attempted to estimate the appropriate cellular environment, including the type of receptors or neurons involved, by minimising $Diff(NT_{sim}, NT_{real})$, the difference between the actual ACh-DA dynamics in vivo (NT recorded by fiber photometry: $NT_{real}$) and model-simulated ACh-DA dynamics (NT generated by the model: $NT_{sim}$) (**Figures 4A, 4B, and S4A**).

- Hypothetical neurons and receptors used in this simulation were identified based on ex vivo imaging (**Figure 3**), the in situ hybridisation results (**Figure S3**), and previous reports<span style="color: blue; ">$^1$</span>. As neurons and receptors, we placed D1R and D2R in $ChAT_{sim}^{vsNAc}$ neurons (as <span style="color: red; ">`def receptor_d1`</span> and <span style="color: red; ">`def receptor_d2`</span> in <span style="color: red; ">`class AChNeuron`</span> in the <span style="color: green; ">`code/src/cells.py`</span>, respectively), as well as D2R and nAChR in $DA_{sim}^{VTA→vsNAc}$ axon terminals (as <span style="color: red; ">`def receptor_d2`</span> and <span style="color: red; ">`def receptor_nAChR`</span> in <span style="color: red; ">`class DopamineAxon`</span> in the <span style="color: green; ">`code/src/cells.py`</span>, respectively).  


### (i) Activation of target neurons by the external stimuli
- To mimic the cellular environment during neurons are activated by the external stimuli, in this modeling, we assumed that the intensity of neuronal activity was changed by the external stimuli or the effect of the receptor expressed on the neurons. $Activation$ includes the spontaneous activity of the target neurons and stimulation of neurons by substances other than ACh and DA. Here, we defined that the  $Activation$ continuously adds a certain value to the target neuron (<span style="color: red; ">`class Activator`</span> in the <span style="color: green; ">`code/src/utils.py`</span>). Thus, the value added as an external stimulus to the $RPM$ of the target neurons every cycle was constant.

### (ii) Release of NT from the target neurons
- To mimic the manner of NT release, we defined that the volume of $NT_{sim}$ released to the extracellular environment from the neurons was determined by the $RPM$ value of the neurons. In this model, the volume of $NT_{sim}$ released is equivalent to the $RPM$ value if $RPM$ is positive and 0 if $RPM$ is negative (**Figure 4B**) (<span style="color: red; ">`def relu`</span> in the <span style="color: green; ">`code/src/utils.py`</span>). Additionally, we defined the diffusion function of DA to mimic the in vivo extracellular DA diffusion mechanism (<span style="color: red; ">`def da_split`</span> and <span style="color: red; ">`def da_diffusion`</span> in <span style="color: red; ">`class Environment`</span> in the <span style="color: green; ">`code/src/simulator.py`</span>)<span style="color: blue; ">$^2$</span>. The time course and intensity of the DA diffusion were investigated using this model (**Figures S4B-S4D**).

### (iii) Effect of the receptor on the target neurons by binding NT
- Next, we considered a strategy to mimic the behavior of nAChR, D1R, and D2R. As reported in previous studies, DA receptors are GPCRs that are thought to respond more slowly than nAChR, the ionotropic receptor<span style="color: blue; ">$^{3–7}$</span>. Incorporating these insights, we defined the characteristics of each receptor using two variables: $RE$ (receptor efficacy), the effect intensity which changes $RPM$; $RS$ (receptor speed), the latency of the effects occurs in $RPM$ after $NT_{sim}$ binds to the receptor. 

- In this experiment, to simplify the setting, we set the nAChR related variables as constant values ($nAChR's RE =1$ as <span style="color: red; ">`nAChR_efficacy`</span>, and the $RS =1$), and estimated the characteristics of D1R and D2R by exploring their $RE$ and $RS$. $RE$ of D1R and D2R was searched over a range of positive values, and their $RS$ was searched over a range of natural numbers.


  - For example, when $ACh_{sim}$ binds to the nAChR expressed on the $DA_{sim}^{VTA→vsNAc}$ axon terminals, there is an increase in the $RPM$ of the $DA_{sim}^{VTA→vsNAc}$ axon terminal $ACh_{sim} × 1$ ($nAChR's RE =1$) one step later ($nAChR's RS =1$). Since the D1R receptor is an excitatory GPCR, we defined its operation as follows: when D1R receives $DA_{sim}$, $D1R's RS$ (as <span style="color: red; ">`delay_d1`</span> in the <span style="color: green; ">`code/src/cells.py`</span>) step later, the $RPM$ is increased to $DA_{sim} × D1R's RE$ (as <span style="color: red; ">`d1_efficacy`</span> in the <span style="color: green; ">`code/src/cells.py`</span>). 
  - By contrast, we defined operation of the D2R receptor, an inhibitory GPCR, receiving  as $D2R's RS$ (as <span style="color: red; ">`delay_d2`</span> in the <span style="color: green; ">`code/src/cells.py`</span>) step later, $RPM$ is decreased by a factor of $DA_{sim} × D2R's RE$ (as <span style="color: red; ">`d2_efficacy`</span> in the <span style="color: green; ">`code/src/cells.py`</span>). With these receptors' settings, we conducted computational modeling to estimate the $RE$ and $RS$ values.

- To estimate receptor efficacy ($RE$) and receptor speed ($RS$) of DA receptors, we explored $RE$ and $RS$ values which minimise $Diff(NT_{sim}, NT_{real})$, the difference between $NT_{sim}$ and NTreal using Bayesian optimisation (**Figure 4B**) (<span style="color: green; ">`code/run_hps.py`</span>). 
- We used Optuna<span style="color: blue; ">$^8$</span>, an automatic hyperparameter optimisation software framework, to determine the best values for $RE$ and $RS$ using $Diff(NT_{sim}, NT_{real})$ as an indicator. We used the following values as a fixed value throughout exploration. The value added by $Activation$ is 1. The number of trials conducted to estimate $RE$ and $RS$ of D1R and D2R was approximately 1,000 to 10,000, depending on the experimental design. In each experimental design, out of these trials, the one representing the smallest $Diff(NT_{sim}, NT_{real})$, the condition where $NT_{sim}$ and NTreal are considered as suitable values, was used to define $RE$ and $RS$ of D1R and D2R. To express the effect speed of each receptor in seconds, the RS of each receptor and DA diffusion step were divided by 200, the sampling rate (**Figures 4E and S4D**).

 ---

## Reference

<span style="color: blue; ">$^1$</span>: De Mei, C., Ramos, M., Iitaka, C. & Borrelli, E. Getting specialized: presynaptic and postsynaptic dopamine D2 receptors. Current Opinion in Pharmacology 9, 53–58 (2009).  
<span style="color: blue; ">$^2$</span>: Liu, C., Goel, P. & Kaeser, P. S. Spatial and temporal scales of dopamine transmission. Nat Rev Neurosci 22, 345–358 (2021).  
<span style="color: blue; ">$^3$</span>: Neve, K. A., Seamans, J. K. & Trantham-Davidson, H. Dopamine receptor signaling. J Recept Signal Transduct Res 24, 165–205 (2004).  
<span style="color: blue; ">$^4$</span>: reif, G. J., Lin, Y. J., Liu, J. C. & Freedman, J. E. Dopamine-modulated potassium channels on rat striatal neurons: specific activation and cellular expression. J. Neurosci. 15, 4533–4544 (1995).  
<span style="color: blue; ">$^5$</span>: Cruz, H. G. et al. Bi-directional effects of GABAB receptor agonists on the mesolimbic dopamine system. Nat Neurosci 7, 153–159 (2004).  
<span style="color: blue; ">$^6$</span>: Galzi, J.-L. et al. Mutations in the channel domain of a neuronal nicotinic receptor convert ion selectivity from cationic to anionic. Nature 359, 500–505 (1992).  
<span style="color: blue; ">$^7$</span>:	Fucile, S. Ca2+ permeability of nicotinic acetylcholine receptors. Cell Calcium 35, 1–8 (2004).  
<span style="color: blue; ">$^8$</span>: Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. Optuna: A Next-generation Hyperparameter Optimization Framework. in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2623–2631 (Association for Computing Machinery, New York, NY, USA, 2019). doi:10.1145/3292500.3330701.

---
### authors
- Ai Miyasaka
  - University of Tsukuba
- Naoki Nonaka
  - RIKEN
- Takeshi Kanda
  - Nara Medical University
- Yuka Terakoshi
  - University of Tsukuba
- Yoan Cherasse
  - University of Tsukuba
- Yukiko Ishikawa
  - University of Tsukuba
- Yulong Li
  - Peking University
- Hotaka Takizawa
  - University of Tsukuba
- Jun Seita
  - RIKEN
- Masashi Yanagisawa
  - University of Tsukuba
- Katsuyasu Sakurai
  - University of Tsukuba
- Takeshi Sakurai
  - University of Tsukuba
- Qinghua Liu
  - National Institute of Biological Sciences, Beijing

**corresponding_contributors**:
  - Qinghua Liu (liuqinghua@nibs.ac.cn)
  - Takeshi Sakurai (sakurai.takeshi.gf@u.tsukuba.ac.jp)
  - Katsuyasu Sakurai (sakurai.katsuyasu.gm@u.tsukuba.ac.jp)

