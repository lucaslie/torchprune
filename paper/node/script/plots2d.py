import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

from torchdyn import utils as plot


def get_mesh(X_data, N=1000):
    X_data = X_data.detach()
    spacing = [torch.linspace(x_i.min(), x_i.max(), N) for x_i in X_data.T]
    return torch.stack(torch.meshgrid(*spacing), dim=-1)


def plot_for_sweep(**kwargs):
    """Plot the desired sweep plot."""
    plot_2d_boundary(**kwargs)


def plot_dataset(x_data, y_data, **kwargs):
    x_data = x_data.detach().cpu()
    y_data = y_data.detach().cpu()
    colors = ["orange", "blue"]
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    for i in range(len(x_data)):
        ax.scatter(
            x_data[i, 0], x_data[i, 1], s=1, color=colors[y_data[i].int()]
        )


def plot_2d_boundary(
    model,
    x_data,
    y_data,
    mesh,
    num_classes=2,
    axis=None,
    **kwargs,
):
    x_data = x_data.detach().cpu()
    y_data = y_data.detach().cpu()
    preds = torch.argmax(nn.Softmax(-1)(model(mesh)), dim=-1)
    preds = preds.detach().cpu().reshape(mesh.size(0), mesh.size(1))
    if axis is None:
        plt.figure(figsize=(8, 4))
        axis = plt.gca()

    contour_colors = ["navy", "tab:orange"]
    scatter_colors = ["midnightblue", "darkorange"]
    axis.contourf(
        mesh[:, :, 0].detach().cpu(),
        mesh[:, :, 1].detach().cpu(),
        preds,
        colors=contour_colors,
        alpha=0.4,
        levels=1,
    )
    for i in range(num_classes):
        axis.scatter(
            x_data[y_data == i, 0],
            x_data[y_data == i, 1],
            alpha=1.0,
            s=6.0,
            linewidths=0,
            c=scatter_colors[i],
            edgecolors=None,
        )


def plot_static_vector_field(model, x_data, t=0.0, N=100, axis=None, **kwargs):
    device = next(model.parameters()).device
    x = torch.linspace(x_data[:, 0].min(), x_data[:, 0].max(), N)
    y = torch.linspace(x_data[:, 1].min(), x_data[:, 1].max(), N)
    X, Y = torch.meshgrid(x, y)

    U, V = torch.zeros_like(X), torch.zeros_like(Y)

    for i in range(N):
        for j in range(N):
            p = torch.cat(
                [X[i, j].reshape(1, 1), Y[i, j].reshape(1, 1)], 1
            ).to(device)
            O = model.defunc(t, p).detach().cpu()
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
        arrowstyle="-|>",
    )

    axis.set_xlim([x.min(), x.max()])
    axis.set_ylim([y.min(), y.max()])
    axis.set_xlabel(r"$h_0$")
    axis.set_ylabel(r"$h_1$")
    axis.set_title("Learned Vector Field")


def plot_2D_state_space(trajectory, y_data, n_lines, **kwargs):
    plot.plot_2D_state_space(trajectory, y_data, n_lines)


def plot_2D_depth_trajectory(
    s_span, trajectory, y_data, axis1=None, axis2=None, **kwargs
):
    if axis1 is None or axis2 is None:
        fig = plt.figure(figsize=(8, 2))
        axis1 = fig.add_subplot(121)
        axis2 = fig.add_subplot(122)

    colors = ["midnightblue", "darkorange"]

    for i, label in enumerate(y_data):
        color = colors[int(label)]
        axis1.plot(s_span, trajectory[:, i, 0], color=color, alpha=0.1)
        axis2.plot(s_span, trajectory[:, i, 1], color=color, alpha=0.1)

    axis1.set_xlabel(r"Depth")
    axis1.set_ylabel(r"Dim. 0")
    axis2.set_xlabel(r"Depth")
    axis2.set_ylabel(r"Dim. 1")


def prepare_data(model, loader, compute_yhat=False, **kwargs):
    """Prepare and return the required data."""
    # setup
    model = model.model
    device = next(model.parameters()).device
    plt.style.use("default")

    # collect data from loader
    x_data, y_data = None, None
    for x_b, y_b in loader:
        if x_data is None:
            x_data = x_b
            y_data = y_b
        else:
            x_data = torch.cat((x_data, x_b))
            y_data = torch.cat((y_data, y_b))
    x_data, y_data = x_data.to(device), y_data.to(device)
    s_span = torch.linspace(0, 1, 100)
    trajectory = model.trajectory(x_data, s_span.to(device)).detach().cpu()
    mesh = get_mesh(x_data).to(device)

    data = {
        "x_data": x_data,
        "y_data": y_data,
        "n_lines": len(x_data),
        "model": model,
        "device": device,
        "s_span": s_span,
        "trajectory": trajectory,
        "mesh": mesh,
    }
    if compute_yhat:
        data["y_hat"] = model(x_data).argmax(dim=1)

    return data


def plot_all(model, loader, plot_folder=None, all_p=False):
    # default plotting style
    plt.style.use("default")

    # retrieve plotting kwargs
    kwargs_plot = prepare_data(model, loader, all_p)

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

    _plot_and_save(plot_2d_boundary, "2d_boundary")
    _plot_and_save(plot_static_vector_field, "static_vector_field")

    if not all_p:
        return

    _plot_and_save(plot_2D_state_space, "2D_state_space")
    _plot_and_save(plot_2D_depth_trajectory, "2D_depth_trajectory")
    _plot_and_save(plot_dataset, "dataset")
