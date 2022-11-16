import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdyn.nn import Augmenter


def plot_dataset(x_data, **kwargs):
    x_data = x_data.detach().cpu()
    plt.figure(figsize=(3, 3))
    plot_samples(x_data, axis=plt.gca())


def plot_for_sweep(**kwargs):
    """Plot the desired sweep plot."""
    plot_samples(**kwargs)


def plot_samples(x_sampled, axis, **kwargs):
    x_sampled = x_sampled.detach().cpu()
    if x_sampled.shape[1] > 2:
        x_sampled = x_sampled[:, 1:3]
    axis.scatter(
        x_sampled[:, 0],
        x_sampled[:, 1],
        s=0.2,
        alpha=0.8,
        linewidths=0,
        c="midnightblue",
        edgecolors=None,
    )
    axis.set_xlim([-2, 2])
    axis.set_ylim([-2, 2])


def plot_samples_density(x_data, x_sampled, **kwargs):
    x_data = x_data.detach().cpu()
    x_sampled = x_sampled.detach().cpu()
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plot_samples(x_sampled, axis=plt.gca())

    plt.subplot(122)
    plot_samples(x_data, axis=plt.gca(), color="red")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


def plot_flow(sample, trajectory, **kwargs):
    traj = trajectory.detach().cpu()
    sample = sample.detach().cpu()
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(sample[:n, 0], sample[:n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])


def plot_2D_depth_trajectory(
    s_span, trajectory, axis1=None, axis2=None, num_lines=200, **kwargs
):
    if axis1 is None or axis2 is None:
        fig = plt.figure(figsize=(8, 2))
        axis1 = fig.add_subplot(121)
        axis2 = fig.add_subplot(122)

    # trajectory has shape [len(s_span), num_data_points, dim] originally
    trajectory = trajectory.detach().permute(1, 2, 0).cpu()

    # subsample trajectories
    num_lines = min(num_lines, len(trajectory))
    trajectory = trajectory[torch.randperm(num_lines)][:num_lines]
    for traj_one in trajectory:
        axis1.plot(s_span, traj_one[0], alpha=0.2)
        axis2.plot(s_span, traj_one[1], alpha=0.2)

    axis1.set_xlabel(r"Depth")
    axis1.set_ylabel(r"Dim. 0")
    axis2.set_xlabel(r"Depth")
    axis2.set_ylabel(r"Dim. 1")


def plot_static_vector_field(model, N=100, axis=None, **kwargs):
    device = next(model.parameters()).device
    model = model[1].defunc.m.net
    x = torch.linspace(-2, 2, N)
    y = torch.linspace(-2, 2, N)

    X, Y = torch.meshgrid(x, y)
    U, V = torch.zeros(N, N), torch.zeros(N, N)

    for i in range(N):
        for j in range(N):
            p = torch.cat(
                [X[i, j].reshape(1, 1), Y[i, j].reshape(1, 1)], 1
            ).to(device)
            O = model(p).detach().cpu()
            U[i, j], V[i, j] = O[0, 0], O[0, 1]

    # convert to cpu numpy
    X, Y, U, V = [tnsr.cpu().numpy() for tnsr in (X, Y, U, V)]

    if axis is None:
        fig = plt.figure(figsize=(3, 3))
        axis = fig.add_subplot(111)

    axis.contourf(
        X,
        Y,
        np.sqrt(U ** 2 + V ** 2),
        cmap="RdYlBu",
        levels=1000,
        alpha=0.6,
    )

    axis.streamplot(
        X.T,
        Y.T,
        U.T,
        V.T,
        color="k",
        density=1.5,
        linewidth=0.7,
        arrowsize=0.7,
        arrowstyle="<|-",
    )

    axis.set_xlim([x.min(), x.max()])
    axis.set_ylim([y.min(), y.max()])
    axis.set_xlabel(r"$h_0$")
    axis.set_ylabel(r"$h_1$")
    axis.set_title("Learned Vector Field")


def prepare_data(model, loader, collect_from_loader=False, n_samp=2 ** 14):
    """Prepare model and data and return as dict."""
    device = next(model.parameters()).device

    # collect prior from model
    prior = None
    for x_b, _ in loader:
        prior = model(x_b.to(device))["prior"]
        break

    # extract ffjord model
    model = model.model

    # set s-span but keep old one around!
    s_span_backup = copy.deepcopy(model[1].s_span)
    model[1].s_span = torch.linspace(1, 0, 2).to(device)

    # s_span for trajectory
    s_span_traj = torch.linspace(1, 0, 100)

    # preparing some data and samples for plotting
    sample = prior.sample(torch.Size([n_samp])).to(device)
    x_sampled = model(sample)
    trajectory = model[1].trajectory(
        Augmenter(1, 1)(sample),
        s_span=s_span_traj.to(device),
    )
    # scrapping first dimension := jacobian trace
    trajectory = trajectory[:, :, 1:]

    # restore s-span
    model[1].s_span = s_span_backup

    data = {
        "sample": sample,
        "x_sampled": x_sampled,
        "s_span": s_span_traj,
        "trajectory": trajectory,
        "model": model,
        "device": device,
    }

    # collect data from loader
    if collect_from_loader:
        x_data = None
        for x_b, _ in loader:
            if x_data is None:
                x_data = x_b
            else:
                x_data = torch.cat((x_data, x_b))
        x_data = x_data.to(device)
        data["x_data"] = x_data

    return data


def plot_all(model, loader, plot_folder=None, all_p=False):
    plt.style.use("default")

    # retrieve plotting kwargs
    kwargs_plot = prepare_data(model, loader, True)

    def _plot_and_save(plt_handle, plt_name):
        plt_handle(**kwargs_plot)
        if plot_folder is not None:
            os.makedirs(plot_folder, exist_ok=True)
            fig = plt.gcf()
            fig.savefig(
                os.path.join(plot_folder, f"{plt_name}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    _plot_and_save(plot_samples_density, "samples_density")
    _plot_and_save(plot_static_vector_field, "static_vector_field")

    if all_p:
        _plot_and_save(plot_dataset, "dataset")
        _plot_and_save(plot_flow, "flow")
