Denoising diffusion models for representation learning in dynamical systems
================================================

This is the official repository to the paper: "Representation learning with unconditional denoising diffusion models for dynamical systems".
Stay tuned for the pre-print manuscript.

If you are using these scripts and the repository, please cite:

> Not yet available
--------

This repository is dedicated to proof that denoising diffusion models
(DDMs) can reproduce states on the attractor of dynamical systems.
Furthermore, the DDMs learn a representation of the dynamical system, which can be subsequently exploited for downstream tasks, different from state generation.
The experiments are performed with the Lorenz 1963 system.

The scripts and module is written in PyTorch [[1]](#1), Pytorch lightning [
[2]](#2), and 
configured with Hydra [[3]](#3).

The folder structure is the following:
```
.
|-- configs            # The hydra config scripts.
|-- data               # Storage of the data
|-- notebooks          # Notebooks that were used to visualize the results
|-- scripts            # The scripts that were used to train the models
|   |-- data           # Scripts and notebooks to generate the needed data
|   |-- predict.py     # Script to predict with a trained neural network
|   |-- train.py       # Hydra-script to train the networks
|-- src                # The Python source code
|-- environment.yml    # A possible environment configuration
|-- LICENSE            # The license file
|-- README.md          # This readme
```
In almost all scripts, only relative directories are used to reference the 
data and models.

If you have further questions, please feel free to contact us or to create a 
GitHub issue.

--------
## References
<a id="1">[1]</a> https://pytorch.org/

<a id="2">[2]</a> https://www.pytorchlightning.ai/

<a id="3">[3]</a> https://hydra.cc/


--------
<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
