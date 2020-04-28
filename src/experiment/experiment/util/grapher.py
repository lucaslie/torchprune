"""The grapher module containing code to plot results."""
import subprocess
from subprocess import DEVNULL, STDOUT
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn


class Grapher(object):
    """The grapher class with matplotlib plotting code."""

    def __init__(
        self, x_min, x_max, logplot, legend, colors, xlabel, ylabel, title
    ):
        """Initialize the grapher with some plotting properties."""
        self._save_fig_width = 20
        self._save_fig_height = 13
        self._font_size = 50
        self._leg_font_size = self._font_size
        self._labelpad = 10
        self._linewidth = 7
        self._logplot = logplot
        self._x_min = float(x_min)
        self._x_max = float(x_max)
        self._legend = legend
        self._colors = colors
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title

        self._open_fig = True

        plt.clf()
        plt.rcParams.update({"font.size": self._font_size})
        plt.rcParams["xtick.major.pad"] = "{}".format(self._labelpad * 3)
        plt.rcParams["ytick.major.pad"] = "{}".format(self._labelpad)
        plt.rcParams["xtick.labelsize"] = self._leg_font_size
        plt.rcParams["ytick.labelsize"] = self._leg_font_size

        seaborn.set_style("whitegrid")

    def _open_image(self, path):
        """Open the image as a subprocess."""
        if self._open_fig:
            img_viewer_cmd = "/usr/bin/evince"
            subprocess.Popen(
                [img_viewer_cmd, path], stdout=DEVNULL, stderr=STDOUT
            )

    def _convert_to_img(self, folder=None, file_name=None):
        """Convert the current figure to an array-based image."""
        # Save and open the image.
        fig = plt.gcf()
        fig.set_size_inches(self._save_fig_width, self._save_fig_height)
        if folder is not None and file_name is not None:
            file_name_full = os.path.join(folder, file_name)
            plt.savefig(file_name_full, bbox_inches="tight")

        # also get a numpy version of it and return it
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # after conversion we can close the plot
        plt.close()

        return img

    def _average_data(self, values):
        """Average the valus over nets and repetitions.

        values has the dimension
        [numNets, numIntervals, numRepetitions, numAlgorithms]
        average over numNets and numRepetitions for regular plot
        more general: * dim 0 and 2 are repetition, which we average over
                      * dim 1 contains the y-value for each x-value
                      * dim 3 allows for multiple lines on the same plot
        """
        mean = values.mean(axis=(0, 2)).swapaxes(0, 1)
        std = np.std(values, axis=(0, 2)).swapaxes(0, 1)
        return mean, std

    def _add_title(self, filename):
        """Add title, labels, etc... to plot."""
        plt.xlabel(self._xlabel, fontsize=self._font_size)
        plt.ylabel(self._ylabel, fontsize=self._font_size)
        plt.title(self._title, fontsize=self._font_size, y=1.02)
        plt.legend(self._legend, loc="best", fontsize=self._font_size)

    def graph(
        self, x_values, y_values, folder, file_name, ref_idx, int_ticks=False
    ):
        """Plot the actual data and store the plot as pdf."""
        # reset plot
        plt.clf()
        plt.rcParams.update({"font.size": self._font_size})
        plt.rcParams["xtick.major.pad"] = "{}".format(self._labelpad * 3)
        plt.rcParams["ytick.major.pad"] = "{}".format(self._labelpad)
        plt.rcParams["xtick.labelsize"] = self._leg_font_size
        plt.rcParams["ytick.labelsize"] = self._leg_font_size

        x_mean, _ = self._average_data(x_values)
        y_mean, y_std = self._average_data(y_values)

        multiple_y = y_mean.tolist()
        std_y = y_std.tolist()
        x_mean = x_mean.tolist()

        x_min = max(self._x_min, min([min(l) for l in x_mean]))
        x_max = min(self._x_max, max([max(l) for l in x_mean]))
        plt.xlim(x_min, x_max)

        y_min = min([min(l) for l in multiple_y])
        y_max = max([max(l) for l in multiple_y])
        y_min = y_min / 1.1 if y_min > 0 else 2 ** (-26)
        y_min = max([2 ** (-26), y_min / 1.2])
        if y_max > 1.0 + 1e-2:
            y_max *= 1.02
        plt.ylim(y_min, y_max)

        if int_ticks:
            ax_current = plt.figure().gca()
            ax_current.xaxis.set_major_locator(MaxNLocator(integer=True))

        for i, y_values in enumerate(multiple_y):
            x_sorted, y_sorted = x_mean[i], y_values
            # xSorted, ySorted = (list(t) for t in
            #                     zip(*sorted(zip(xMean[i], y))))
            color = self._colors[i % len(self._colors)]
            if i is ref_idx:
                style = "--"
                x_sorted[0] = x_min
            else:
                style = "-"
            if self._logplot:
                plt.semilogy(
                    x_sorted,
                    y_sorted,
                    style,
                    linewidth=self._linewidth,
                    color=color,
                    alpha=0.7,
                    basey=2,
                )
            else:
                plt.plot(
                    x_sorted,
                    y_sorted,
                    style,
                    linewidth=self._linewidth,
                    color=color,
                    alpha=0.7,
                )
            x_sorted, std_y_sorted = x_mean[i], std_y[i]
            # xSorted, stdYSorted = (list(t) for t in
            #                        zip(*sorted(zip(xMean[i], stdY[i]))))
            plt.fill_between(
                x_sorted,
                np.array(y_sorted) - np.array(std_y_sorted),
                np.array(y_sorted) + np.array(std_y_sorted),
                alpha=0.15,
                facecolor=color,
            )

        self._add_title(file_name)

        # save image
        img = self._convert_to_img(folder, file_name)

        return img

    def graph_heat(self, values, folder, file, cmap):
        """Plot and store a heatmap."""
        # assume that values is of dimension [height, width] of resulting
        # image of heatmap

        # draw heatmap with matplotlib
        plt.close()
        fig = plt.gcf()
        try:
            fig.delaxes(fig.axes[1])
        except IndexError:
            pass

        color_mesh = plt.pcolormesh(values, cmap=cmap)
        fig.colorbar(color_mesh)
        plt.gca().axis([0, values.shape[1], 0, values.shape[0]])

        # create numpy version of image and return it
        return self._convert_to_img(folder, file)

    def graph_histo(self, values, folder, file, num_bins=75):
        """Plot and store a histogramm."""
        values, _ = self._average_data(values)
        values = values.transpose()

        # compute histogram
        counts, bins = np.apply_along_axis(
            np.histogram, arr=values, axis=0, bins=num_bins, density=True
        )
        counts = np.stack(counts, axis=1)
        bins = np.stack(bins, axis=1)
        midpoints = (bins[:-1] + bins[1:]) / 2.0

        # smooth counts
        # Potential windows: 'hanning', 'hamming', 'bartlett', 'blackman'
        filter = np.hanning(3)
        filter /= filter.sum()

        def smooth(arr):
            return np.convolve(arr, filter, mode="same")

        counts_smoothed = np.apply_along_axis(smooth, arr=counts, axis=0)

        # draw histogram
        plt.close()
        x_min = self._x_min
        x_max = self._x_max
        if x_min < 0.3 * x_max:
            x_max *= 0.4
        if self._logplot:

            def _f_plt(*args, **kwargs):
                return plt.semilogy(*args, **kwargs, basey=2)

        else:
            _f_plt = plt.plot

        for i in range(midpoints.shape[1]):
            _f_plt(
                midpoints[:, i],
                counts_smoothed[:, i],
                "-",
                linewidth=self._linewidth,
                color=self._colors[i],
                alpha=0.7,
            )
        plt.xlim(x_min, x_max)
        plt.ylim(counts_smoothed.min(), counts_smoothed.max())
        self._add_title(file)

        # save and return numpy version
        return self._convert_to_img(folder, file)
