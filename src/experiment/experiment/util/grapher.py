"""The grapher module containing code to plot results."""
import subprocess
from subprocess import DEVNULL, STDOUT
import os
import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import sklearn as skl


class Grapher(object):
    """The grapher class with matplotlib plotting code."""

    def __init__(
        self,
        x_values,
        y_values,
        folder,
        file_name,
        ref_idx,
        x_min,
        x_max,
        legend,
        colors,
        xlabel,
        ylabel,
        title,
        linestyles=None,
        hatches=None,
    ):
        """Initialize the grapher with some plotting properties."""
        self._save_fig_width = 20
        self._save_fig_height = 13
        self._font_size = 50
        self._leg_font_size = self._font_size
        self._labelpad = 6
        self._linewidth = 5
        self._linestyles = linestyles if linestyles else ["-"] * len(legend)
        self._hatches = hatches
        self._barwidth = 0.2
        self._markersize = 14
        self._x_values = x_values
        self._y_values = y_values
        self._folder = folder
        self._file_name = file_name
        self._ref_idx = ref_idx
        self._x_min = float(x_min)
        self._x_max = float(x_max)
        self._legend = legend
        self._colors = colors
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title
        self._open_fig = True

        self._figure = plt.figure()
        self._set_global_plt_params()

    def _set_global_plt_params(self):
        """Set global plotting parameters."""
        self._figure.clf()
        plt.figure(self._figure.number)
        plt.rcParams.update({"font.size": self._font_size})
        plt.rcParams["xtick.major.pad"] = "{}".format(self._labelpad * 5)
        plt.rcParams["ytick.major.pad"] = "{}".format(self._labelpad)
        plt.rcParams["xtick.labelsize"] = self._leg_font_size
        plt.rcParams["ytick.labelsize"] = self._leg_font_size
        sns.set_style("whitegrid")

    def _set_fig_layout(self):
        """Format the figure."""
        self._figure.set_size_inches(
            self._save_fig_width, self._save_fig_height
        )
        self._figure.tight_layout()
        self._figure.canvas.draw()

    def _set_axes_layout(
        self,
        x_data,
        y_data,
        y_offset,
        percentage_x,
        percentage_y,
        remove_outlier,
        show_delta,
        force_x_axis=False,
        logscale=False,
        x_is_int=False,
    ):
        """Format the axes of the current figure."""
        # log scale if wanted on x-axis
        if logscale:
            self._figure.gca().set_xscale("log")

        # set plot limits on x-axis ...
        x_min = self._x_min * (100 if percentage_x else 1)
        x_max = self._x_max * (100 if percentage_x else 1)

        # only dynamically update x_limits if we don't force them
        if not force_x_axis:
            x_min = max(x_min, x_data.min())
            x_max = min(x_max, x_data.max())
            if logscale:
                x_min = 10.0 ** (np.log10(x_min) - 0.02)
                x_max = 10.0 ** (np.log10(x_max) + 0.02)
            else:
                x_buffer = 0.02 * abs(x_max - x_min)
                x_min -= x_buffer
                x_max += x_buffer
        x_min = np.floor(x_min) if percentage_x and not logscale else x_min
        x_max = np.ceil(x_max) if percentage_x and not logscale else x_max
        self._figure.gca().set_xlim(x_min, x_max)

        # set plot limits on y-axis
        y_min = np.percentile(y_data - y_offset, 15 if remove_outlier else 0)
        y_min = np.floor(y_min) if percentage_y else y_min
        y_max = np.percentile(y_data + y_offset, 85 if remove_outlier else 100)
        y_max = np.ceil(y_max) if percentage_y else y_max
        if not np.isfinite(y_min):
            y_min = 0.1 * y_max if np.isfinite(y_max) else 0.0
        if not np.isfinite(y_max):
            y_max = 10 * y_min if np.isfinite(y_min) else 1.0
        y_buffer = 0.1 * abs(y_max - y_min)
        self._figure.gca().set_ylim(y_min - y_buffer, y_max + y_buffer)

        # Percentage axes
        self._figure.gca().xaxis.set_major_formatter(
            self._get_formatter(False, percentage_x, x_is_int)
        )
        self._figure.gca().yaxis.set_major_formatter(
            self._get_formatter(show_delta, percentage_y)
        )

    def _get_formatter(
        self, add_plus_sign=False, add_percentage=False, is_int=False
    ):
        @mtick.FuncFormatter
        def percentage_formatter(val, pos):
            plus_pre = "+" if add_plus_sign else ""
            suffix = "%" if add_percentage else ""
            if is_int:
                val_str = f"{int(val)}"
            elif add_percentage:
                val_str = f"{val:.1f}"
            else:
                val_str = f"{val:.2f}"
            return f"{plus_pre if val > 0.0 else ''}{val_str}{suffix}"

        return percentage_formatter

    def _open_image(self, path):
        """Open the image as a subprocess."""
        if self._open_fig:
            img_viewer_cmd = "/usr/bin/evince"
            subprocess.Popen(
                [img_viewer_cmd, path], stdout=DEVNULL, stderr=STDOUT
            )

    def store_plot(self):
        """Store the plot as desired."""
        if self._folder is not None and self._file_name is not None:
            file_name_full = os.path.join(self._folder, self._file_name)
            os.makedirs(self._folder, exist_ok=True)
            self._figure.savefig(file_name_full, bbox_inches="tight")

    def _convert_to_img(self):
        """Convert the current figure to an array-based image."""
        # Save and open the image.
        self.store_plot()

        # also get a numpy version of it and return it
        img = np.frombuffer(self._figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self._figure.canvas.get_width_height()[::-1] + (3,))

        # after conversion we can close the plot
        plt.close(self._figure)

        return img

    def _average_data(self, values, idx_ref=None):
        """Average the valus over nets and repetitions.

        values has the dimension
        [numNets, numIntervals, numRepetitions, numAlgorithms]
        average over numNets and numRepetitions for regular plot
        more general: * dim 0 and 2 are repetition, which we average over
                      * dim 1 contains the y-value for each x-value
                      * dim 3 allows for multiple lines on the same plot

        Returns data with mean and std in shape [numAlgorithms, numIntervals]
        """
        mean = values.mean(axis=(0, 2)).swapaxes(0, 1)
        std = np.std(values, axis=(0, 2)).swapaxes(0, 1)
        if idx_ref is not None:
            mean -= mean[idx_ref : idx_ref + 1]
        return mean, std

    def _flatten_data(self, values, idx_ref=None):
        """Flatten the values over nets and repetitions.

        values has the dimension
        [numNets, numIntervals, numRepetitions, numAlgorithms]

        Returns the data in shape
        [numAlgorithms, numNets * numIntervals * numRepetitions]
        """
        values_flat = values.swapaxes(0, -1).reshape(values.shape[-1], -1)
        if idx_ref is not None:
            values_flat -= values_flat[idx_ref : idx_ref + 1]
        return values_flat

    def _bootstrap_and_regress(
        self, x_train, y_train, x_test, fit_intercept, conf_int=95
    ):
        """Return regression values with bootstrapping as confidence intervals.

        Args:
        x_train (ndarray): "input", shape == [num_data_points x num_features]
        y_train (ndarray): "output", shape == [num_data_points]
        x_test (ndarray): "test input", shape [num_data_points x num_features]

        Returns predicted values for nominal model with upper/lower confidence
        based bootstrapped models.
        """

        def f_reg(_x, _y):
            """Return predicted values from sklearn's linear regressor."""
            linreg = skl.linear_model.LinearRegression(
                fit_intercept=fit_intercept
            )
            linreg.fit(_x, _y)
            return linreg.predict(x_test)

        yhat = f_reg(x_train, y_train)
        if conf_int is None:
            err_bands = None
        else:
            yhat_boots = sns.algorithms.bootstrap(x_train, y_train, func=f_reg)
            err_bands = sns.utils.ci(yhat_boots, conf_int, axis=0)
        return yhat, err_bands

    def _add_title(self, legends, is_delta=False):
        """Add title, labels, etc... to plot."""
        self._figure.gca().set_xlabel(self._xlabel, fontsize=self._font_size)
        self._figure.gca().set_ylabel(
            f"Delta {self._ylabel}" if is_delta else self._ylabel,
            fontsize=self._font_size,
        )
        self._figure.gca().set_title(
            self._title, fontsize=self._font_size, y=1.02
        )

    def graph(
        self,
        show_ref=False,
        show_delta=True,
        percentage_x=False,
        percentage_y=False,
        remove_outlier=True,
        logplot=False,
        store=True,
        kwargs_legend={},
    ):
        """Plot the actual data and store the plot as pdf."""
        # reset plot
        self._set_global_plt_params()

        # some sanity checks on parameters
        if self._ref_idx is None:
            show_ref = True
            show_delta = False

        x_mean, _ = self._average_data(self._x_values)
        y_mean, y_std = self._average_data(
            self._y_values, self._ref_idx if show_delta else None
        )

        # convert to percentage
        x_mean *= 100.0 if percentage_x else 1.0
        y_mean *= 100.0 if percentage_y else 1.0
        y_std *= 100.0 if percentage_y else 1.0

        # set axes layout
        self._set_axes_layout(
            x_mean,
            y_mean,
            y_std,
            percentage_x,
            percentage_y,
            remove_outlier,
            show_delta,
            logscale=logplot,
            x_is_int=self._x_values.dtype == int,
        )

        legends = []
        for i, (x_i, y_i, y_i_std) in enumerate(zip(x_mean, y_mean, y_std)):
            color = self._colors[i % len(self._colors)]
            if i is self._ref_idx and not show_ref:
                continue

            legends.append(self._legend[i])

            self._figure.gca().errorbar(
                x=x_i,
                y=y_i,
                yerr=y_i_std,
                linestyle=self._linestyles[i],
                linewidth=self._linewidth,
                color=color,
                marker="o",
                markersize=self._markersize,
                capsize=self._markersize / 2,
                capthick=self._linewidth / 2,
                elinewidth=self._linewidth / 2,
                lolims=False,
                uplims=False,
            )

        # add title
        self._add_title(legends, show_delta)

        # set layout
        self._set_fig_layout()

        # add legend
        l_kwargs = {"loc": "best", "fontsize": self._font_size, "ncol": 2}
        l_kwargs.update(kwargs_legend)
        self._figure.gca().legend(legends, **l_kwargs)

        # save image
        if store:
            return self._convert_to_img()

        return self._figure

    def graph_histo(
        self,
        show_ref=False,
        show_delta=True,
        normalize=True,
        store=True,
        y_err_bounds=None,
    ):
        """Plot a bar plot."""
        # reset plot
        self._set_global_plt_params()
        sns.set_style("ticks")

        # some sanity checks on parameters
        if self._ref_idx is None:
            show_ref = True
            show_delta = False

        # convert y to percentage
        y_values = 100.0 * copy.deepcopy(self._y_values)

        # normalize if wanted
        if normalize:
            y_values /= y_values.sum(axis=1, keepdims=True)

        # adjust error bounds if necessary
        if y_err_bounds is None:
            if normalize:
                y_err_bounds = np.array([-1.0, 1.0])
            else:
                y_err_bounds = np.array([0.0, 1.0])
        y_err_bounds *= 100.0

        # average values
        x_mean = self._average_data(self._x_values)[0].astype(int)
        if show_delta:
            y_mean, y_std = self._average_data(y_values, self._ref_idx)
        else:
            y_mean, y_std = self._average_data(y_values)

        # error bars from std deviation
        y_err_low = np.minimum(y_std, y_mean - y_err_bounds[0])
        y_err_up = np.minimum(y_std, y_err_bounds[1] - y_mean)

        # compute the bar width
        width_bar = 0.96 * np.diff(x_mean[0]).mean()

        # get the xlimits
        x_min = max(self._x_min, x_mean.min())
        x_max = min(self._x_max, x_mean.max())
        x_buffer = 0.53 * width_bar
        x_min -= x_buffer
        x_max += x_buffer

        # get the y limits
        y_min = (y_mean - y_err_low).min()
        y_max = (y_mean + y_err_up).max()
        y_range = abs(y_max - y_min)
        y_buffer = 0.1 * y_range
        y_min -= y_buffer
        y_max += y_buffer

        # sanity check that they are finite (avoiding matplotlib errors)
        # if limits are not finite this plot is not valid anyway
        x_min = x_min if np.isfinite(x_min) else 0
        x_max = x_max if np.isfinite(x_max) else 1.0
        y_min = y_min if np.isfinite(y_min) else 0
        y_max = y_max if np.isfinite(y_max) else 1.0

        # set grids and have index of current grid
        num_bars = len(x_mean) if show_ref else len(x_mean) - 1

        gspec = gridspec.GridSpec(num_bars, 1)
        i_gs = 0

        # have a small font size for axis here
        font_size_small = self._font_size // max(1, (num_bars / 4))

        for i, (x_i, y_i, y_i_err_low, y_i_err_up) in enumerate(
            zip(x_mean, y_mean, y_err_low, y_err_up)
        ):
            # get color
            color = self._colors[i % len(self._colors)]

            # skip ref index since it makes no sense to plot
            if i is self._ref_idx and not show_ref:
                continue

            # creating new axes object
            ax_obj = self._figure.add_subplot(gspec[i_gs : i_gs + 1, 0:])

            # plot as bar plot
            for j, (x_i_j, y_i_j, y_i_j_err_low, y_i_j_err_up) in enumerate(
                zip(x_i, y_i, y_i_err_low, y_i_err_up)
            ):
                ax_obj.bar(
                    x=x_i_j,
                    height=y_i_j,
                    width=width_bar,
                    color=color,
                    lw=3,
                    edgecolor="k",
                    hatch="//" if self._hatches is None else self._hatches[j],
                    yerr=[[y_i_j_err_low], [y_i_j_err_up]],
                    error_kw={"capsize": 7.0, "capthick": 3, "elinewidth": 3},
                )

            # setting uniform x and y lims
            ax_obj.set_xlim(x_min, x_max)
            ax_obj.set_ylim(y_min, y_max)

            # make background transparent
            ax_obj.patch.set_alpha(0)

            # only set title of top subplot
            if i_gs == 0:
                ax_obj.set_title(self._title)

            # configure x-axis (lowest plot only)
            if i_gs == num_bars - 1:
                # configure the axes
                if show_delta:
                    ax_obj.set_xlabel(f"Delta {self._xlabel}")
                else:
                    ax_obj.set_xlabel(self._xlabel)
                ax_obj.xaxis.set_major_locator(
                    mtick.MaxNLocator(integer=True, nbins="auto")
                )
                ax_obj.xaxis.set_minor_locator(
                    mtick.MaxNLocator(integer=True, nbins=len(x_i))
                )
                ax_obj.tick_params(
                    axis="x",
                    which="major",
                    length=15,
                    width=3,
                    labelsize=font_size_small,
                )
                ax_obj.tick_params(axis="x", which="minor", length=5, width=1)

            else:
                # remove borders, axis ticks, and labels for x-axis
                ax_obj.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labeltop=False,
                    labelleft=False,
                    labelright=False,
                )

            # configure y-axis (all plots)
            ax_obj.yaxis.set_minor_locator(mtick.AutoMinorLocator())
            ax_obj.tick_params(
                axis="y",
                which="major",
                length=15,
                width=3,
                labelsize=font_size_small,
            )
            ax_obj.tick_params(axis="y", which="minor", length=5, width=1)
            ax_obj.tick_params(
                axis="y", which="major", labelsize=font_size_small
            )
            ax_obj.yaxis.set_ticks_position("right")
            ax_obj.yaxis.set_major_formatter(
                self._get_formatter(show_delta, True)
            )

            # don't show borders
            for spine in ["top", "right", "left", "bottom"]:
                ax_obj.spines[spine].set_visible(False)

            # plot legend as y label
            ax_obj.set_ylabel(
                self._legend[i]
                if len(self._legend[i]) < 6
                else self._legend[i][:4],
                fontsize=font_size_small,
            )

            # increment i_gs every time we actually plotted something
            i_gs += 1

        # set layout
        # use negative hspace to make overlap if we want to in the future
        # (like a ridge plot or joy plot)
        gspec.update(hspace=0.05)
        self._set_fig_layout()

        # save image
        if store:
            return self._convert_to_img()

        return self._figure

    def graph_regression(
        self,
        fit_intercept=True,
        show_ref=False,
        show_delta=True,
        percentage_x=False,
        percentage_y=False,
        remove_outlier=True,
        store=True,
        kwargs_legend={},
    ):
        """Plot scatter with regression."""
        # reset plot
        self._set_global_plt_params()

        # some sanity checks on parameters
        if self._ref_idx is None:
            show_ref = True
            show_delta = False

        # flatten data
        x_flat = self._flatten_data(self._x_values)
        y_flat = self._flatten_data(
            self._y_values, idx_ref=self._ref_idx if show_delta else None
        )

        # convert to percentage
        x_flat *= 100.0 if percentage_x else 1.0
        y_flat *= 100.0 if percentage_y else 1.0

        # setup axes of figure (over-rule x-limits)
        self._set_axes_layout(
            x_flat,
            y_flat,
            0.0,
            percentage_x,
            percentage_y,
            remove_outlier,
            show_delta,
            force_x_axis=True,
        )

        # get "test" data from axis limits
        x_test = np.linspace(*self._figure.gca().get_xlim(), 100)

        # loop through each algorithm and visualize it.
        legends = []
        handles = []
        for i, (x_i, y_i) in enumerate(zip(x_flat, y_flat)):
            # check if we should reference
            if i is self._ref_idx and not show_ref:
                continue

            # set color and legend
            color = self._colors[i % len(self._colors)]
            legends.append(self._legend[i])

            # regress data
            y_hat, err_bands = self._bootstrap_and_regress(
                x_i.reshape(-1, 1), y_i, x_test.reshape(-1, 1), fit_intercept
            )

            # plot scatter, regressor, and confidence intervals.
            self._figure.gca().scatter(
                x_i,
                y_i,
                edgecolor="face",
                facecolor=color,
                linewidths=self._markersize / 2,
            )
            h_line = self._figure.gca().plot(
                x_test,
                y_hat,
                linestyle=self._linestyles[i],
                linewidth=self._linewidth,
                color=color,
            )[0]
            self._figure.gca().fill_between(
                x_test,
                *err_bands,
                alpha=0.15,
                facecolor=color,
            )

            # store handles as grouped tuples for the legend later ...
            handles.append(h_line)

        # add title
        self._add_title(legends, show_delta)

        # set layout
        self._set_fig_layout()

        # add legend
        l_kwargs = {"loc": "best", "fontsize": self._font_size, "ncol": 2}
        l_kwargs.update(kwargs_legend)
        self._figure.gca().legend(handles, legends, **l_kwargs)

        # save image
        if store:
            return self._convert_to_img()

        return self._figure
