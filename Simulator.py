import sys
import os
from abc import ABC, abstractmethod
import torch   
from tqdm import tqdm
import abstract
from abstract import ODE, SDE
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.axes import Axes

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# Add the project root to sys.path
sys.path.insert(0, project_root)


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape ()
            - dt: time, shape ()
        Returns:
            - nxt: state at time t + dt
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (batch_size, dim)
            - ts: timesteps, shape (nts,)
        Returns:
            - x_final: final state at time ts[-1], shape (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (bs, dim)
            - ts: timesteps, shape (num_timesteps,)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, num_timesteps, dim)
        """
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1)):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
    

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + h * self.ode.drift_coefficient(xt, t)
    
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        # print("drift product item: ", self.sde.drift_coefficient(xt, t))
        # print("diffusion product item: ", torch.sqrt(h) * self.sde.diffusion_coefficient(xt, t) * torch.randn_like(xt))
        return xt + h * self.sde.drift_coefficient(xt, t) + torch.sqrt(h) * self.sde.diffusion_coefficient(xt, t) * torch.randn_like(xt)
    

class BrownianMotion(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma
        print("the sigma is", self.sigma)
        
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - drift: shape (bs, dim)
        """
        return torch.zeros_like(xt)
        
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - diffusion: shape (bs, dim)
        """
        coeff = torch.ones(xt.shape[1])
        coeff = coeff.expand(xt.shape[0], -1)
        return self.sigma * coeff
    

class OUProcess(SDE):
    def __init__(self, theta: float, sigma: float):
        self.theta = theta
        self.sigma = sigma
        
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - drift: shape (bs, dim)
        """
        return -self.theta * xt
        
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - diffusion: shape (bs, dim)
        """
        coeff = torch.eye(xt.shape[1])
        coeff = coeff.expand(xt.shape[0], -1)
        return self.sigma * coeff
    

def plot_trajectories_1d(x0: torch.Tensor, simulator: Simulator, timesteps: torch.Tensor, ax: Optional[Axes] = None):
        """
        Graphs the trajectories of a one-dimensional SDE with given initial values (x0) and simulation timesteps (timesteps).
        Args:
            - x0: state at time t, shape (num_trajectories, 1)
            - simulator: Simulator object used to simulate
            - t: timesteps to simulate along, shape (num_timesteps,)
            - ax: pyplot Axes object to plot on
        """
        if ax is None:
            ax = plt.gca()
        trajectories = simulator.simulate_with_trajectory(x0, timesteps) # (num_trajectories, num_timesteps, ...)
        for trajectory_idx in range(trajectories.shape[0]):
            trajectory = trajectories[trajectory_idx, :, 0] # (num_timesteps,)
            ax.plot(timesteps.cpu(), trajectory.cpu())


def plot_scaled_trajectories_1d(x0: torch.Tensor, simulator: Simulator, timesteps: torch.Tensor, time_scale: float, label: str, ax: Optional[Axes] = None):
        """
        Graphs the trajectories of a one-dimensional SDE with given initial values (x0) and simulation timesteps (timesteps).
        Args:
            - x0: state at time t, shape (num_trajectories, 1)
            - simulator: Simulator object used to simulate
            - t: timesteps to simulate along, shape (num_timesteps,)
            - time_scale: scalar by which to scale time
            - label: self-explanatory
            - ax: pyplot Axes object to plot on
        """
        if ax is None:
            ax = plt.gca()
        trajectories = simulator.simulate_with_trajectory(x0, timesteps) # (num_trajectories, num_timesteps, ...)
        for trajectory_idx in range(trajectories.shape[0]):
            trajectory = trajectories[trajectory_idx, :, 0] # (num_timesteps,)
            ax.plot(timesteps.cpu() * time_scale, trajectory.cpu(), label=label)

#### Example Usage ####
#### Main ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    sigma = 0.01
    brownian_motion = BrownianMotion(sigma)
    simulator = EulerMaruyamaSimulator(sde=brownian_motion)
    x0 = torch.zeros(5,1).to(device) # Initial values - let's start at zero
    ts = torch.linspace(0.0,5.0,500).to(device) # simulation timesteps

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_title(r'Trajectories of Brownian Motion with $\sigma=$' + str(sigma), fontsize=18)
    ax.set_xlabel(r'Time ($t$)', fontsize=18)
    ax.set_ylabel(r'$X_t$', fontsize=18)
    plot_trajectories_1d(x0, simulator, ts, ax)
    plt.show()

    sigmas = [1.0, 2.0, 10.0]
    ds = [0.25, 1.0, 4.0] # sigma**2 / 2t
    simulation_time = 10.0

    fig, axes = plt.subplots(len(ds), len(sigmas), figsize=(8 * len(sigmas), 8 * len(ds)))
    axes = axes.reshape((len(ds), len(sigmas)))
    for d_idx, d in enumerate(ds):
        for s_idx, sigma in enumerate(sigmas):
            theta = sigma**2 / 2 / d
            ou_process = OUProcess(theta, sigma)
            simulator = EulerMaruyamaSimulator(sde=ou_process)
            x0 = torch.linspace(-20.0,20.0,20).view(-1,1).to(device)
            time_scale = sigma**2
            ts = torch.linspace(0.0,simulation_time / time_scale,1000).to(device) # simulation timesteps
            ax = axes[d_idx, s_idx]
            plot_scaled_trajectories_1d(x0=x0, simulator=simulator, timesteps=ts, time_scale=time_scale, label=f'Sigma = {sigma}', ax=ax)
            ax.set_title(f'OU Trajectories with Sigma={sigma}, Theta={theta}, D={d}')
            ax.set_xlabel(f't / (sigma^2)')
            ax.set_ylabel('X_t')
    plt.show()

if __name__ == "__main__":
    main()
