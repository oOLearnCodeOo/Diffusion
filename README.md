# Diffusion
Project Description
This project is a consolidated version of MIT diffusion-related course code. It demonstrates the following core components:

Sampling from Vector Fields in an ODE:

Shows how to sample and simulate trajectories based on a given vector field defined in an ordinary differential equation (ODE).
Introducing Noise via a Score Function:

Explains how to incorporate noise into the system and utilize a score function (i.e., the gradient of the log probability) for analysis or sampling.
Flow Score Matching:

Trains a neural network to fit or approximate a target vector field in the diffusion process, highlighting how score matching is performed.
The main goal is to provide a hands-on understanding of diffusion principles and numerical methods, along with reusable code for the core algorithms and training routines.

# Overview of Flow Score Matching
Fitting a Vector Field

In Flow Score Matching, a neural network is trained to output the vector field (or its gradient/score) that aligns with a given target or reference process. This helps in modeling the trajectory or distribution within a diffusion framework.

Implementation Steps

Define the network architecture (input could be state, possibly including time; output is the corresponding gradient or vector field).
Use an appropriate loss function (e.g., KL divergence or mean squared error) to match the target field.
After training, the network can be leveraged for efficient sampling or reconstructions of the diffusion process.
For further theoretical details, please consult relevant papers or tutorials.

References
Relevant MIT diffusion course materials
Score-Based Generative Modeling research
Chen, T. Q., et al. “Neural Ordinary Differential Equations.” Advances in Neural Information Processing Systems (2018).
License
Distributed under the MIT License. You are free to use, modify, or distribute the code in this project, provided that you include the original copyright notice.