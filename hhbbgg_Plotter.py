# %%
import os
import matplotlib
matplotlib.use('Agg')
import uproot
from hist import Hist, Stack, intervals
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from normalisation import getLumi

# Load CMS style including color-scheme
hep.style.use("CMS")

# Function to read histograms
def get_histogram(file_name, hist_name):
    return Hist(file_name[hist_name])


# Function to read and sum histograms
def sum_histograms(histograms):
    return sum(histograms)

# Function to return histogram ratio
def get_ratio(hist_a, hist_b):
    # Extract the bin edges and contents
    edges_a = hist_a.axes.edges[0]
    edges_b = hist_b.axes.edges[0]
    if not np.array_equal(edges_a, edges_b):
        raise ValueError("Histograms have different binning")
    # Extract the counts
    counts_a = hist_a.values()
    counts_b = hist_b.values()
    # Compute the errors for each histogram
    errors_a = np.sqrt(counts_a)  # Assuming Poisson statistics
    errors_b = np.sqrt(counts_b)  # Assuming Poisson statistics

    # Compute the ratio with handling of division by zero
    ratio_counts = np.divide(counts_a, counts_b, out=np.zeros_like(counts_a, dtype=float), where=counts_b!=0)
    ratio_hist = Hist(hist_a.axes[0])
    ratio_hist[...] = ratio_counts

    # Compute the relative errors
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors_a = np.divide(errors_a, counts_a, out=np.zeros_like(errors_a, dtype=float), where=counts_a!=0)
        relative_errors_b = np.divide(errors_b, counts_b, out=np.zeros_like(errors_b, dtype=float), where=counts_b!=0)
    # Propagate the errors for the ratio
    ratio_errors = ratio_counts * np.sqrt(relative_errors_a**2 + relative_errors_b**2)

    return ratio_hist, ratio_errors

def stack1d_histograms(uproot_loaded_filename, data_samples, mc_samples, signal_samples, histogram_names, legend_dict, xtitle_dict, output_directory):
    for hist_name in histogram_names:
        fig, (ax, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        fig.subplots_adjust(hspace=0.05)  # Adjust space between main plot and ratio plot

        # Plot data histogram
        data_histogram = sum_histograms([get_histogram(uproot_loaded_filename,f"{data_sample}/{hist_name}") for data_sample in data_samples])
        data_histogram.plot(ax=ax, stack=False, histtype='errorbar', yerr=True, xerr=True, color='black', label=f"Data", flow="sum")

        # Plot MC histograms as stack
        mc_stack = Stack(*[get_histogram(uproot_loaded_filename,f"{mc_sample}/{hist_name}") for mc_sample in mc_samples])
        mc_stack[::-1].plot(ax=ax, stack=True, histtype='fill', label=legend_dict.values(),flow="sum")

        # Plot signal histograms
        for signal_sample in signal_samples:
            signal_histogram = get_histogram(uproot_loaded_filename,f"{signal_sample}/{hist_name}")
            signal_histogram.plot(ax=ax, histtype='step',yerr=True, xerr=True, label=legend_dict[signal_sample], color="red")

        # Plot ratio plot
        ratio, error = get_ratio(data_histogram,sum_histograms(mc_stack))
        ratio.plot(ax=ax_ratio, histtype='errorbar', yerr=error, xerr=True, color='black', flow="sum")
        ax_ratio.axhline(1, linestyle='--', color='gray')
        ax_ratio.set_ylim(0, 3)
        ax_ratio.set_ylabel('Data / MC')

        # Style
        # x_axis_bining = data_histogram.axes.edges[0]
        # ax.set_xlim(min(x_axis_bining), max(x_axis_bining))
        ax.set_yscale('log')
        ax.set_ylim(0.1,1E8)
        ax.set_xlabel("")
        ax.set_ylabel("Events")
        hep.cms.label("",ax=ax, lumi='{0:.2f}'.format(getLumi()),loc=0,llabel="Work in progress")
        ax.legend(ncol=2,loc='upper right', fontsize=18)
        ax_ratio.set_xlabel(xtitle_dict[hist_name])
        ax_ratio.set_ylabel("Data/MC")

        # Save plots
        plt.savefig(f"{output_directory}/{hist_name}.pdf", bbox_inches='tight')

def main():
    # Open the ROOT file
    file_path = "outputfiles/hhbbgg_Analyzer.root"
    uproot_loaded_filename = uproot.open(file_path)

    # List of data histograms
    data_samples = ["Data_EraE", "Data_EraF", "Data_EraG"]

    # List of MC processes
    mc_samples = ["GGJets", "GJetPt20To40", "GJetPt40", "GluGluHToGG", "VBFHToGG", "VHToGG", "ttHToGG"]

    # List of signal processes
    signal_samples = ["GluGluToHH",]

    # Dictionary for legends
    legend_dict = {"GGJets":r"$\gamma\gamma$+jets", "GJetPt20To40":r"$\gamma$+jets ($20< p_T < 40$)", "GJetPt40":r"$\gamma$+jets ($p_T > 40$)", "GluGluHToGG":r"$gg\rightarrow\,H\rightarrow\gamma\gamma$", "VBFHToGG":r"$VBF\:H\rightarrow\gamma\gamma$", "VHToGG":r"$V\:H\rightarrow\gamma\gamma$", "ttHToGG":r"$t\bar{t}H\rightarrow\gamma\gamma$", "GluGluToHH":r"$gg\rightarrow\,HH$"}

    # List of histogram names to stack
    histogram_names = ["h_reg_preselection_dibjet_mass","h_reg_preselection_diphoton_mass","h_reg_preselection_bbgg_mass","h_reg_preselection_lead_pho_pt","h_reg_preselection_sublead_pho_pt","h_reg_preselection_dibjet_pt"]

    # Dictionary for x-axis title
    xtitle_dict = {"h_reg_preselection_dibjet_mass":r"$m_{b\bar{b}}$ [GeV]","h_reg_preselection_diphoton_mass":r"$m_{\gamma\gamma}$ [GeV]","h_reg_preselection_bbgg_mass":r"$m_{b\bar{b}\gamma\gamma}$ [GeV]","h_reg_preselection_lead_pho_pt":r"lead $\gamma\:p_T$ [GeV]","h_reg_preselection_sublead_pho_pt":r"sublead $\gamma\:p_T$ [GeV]","h_reg_preselection_dibjet_pt":r"$b\bar{b} p_T$ [GeV]"}

    # create the tdirectory to save plots
    output_directory = "stack_plots"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    stack1d_histograms(uproot_loaded_filename, data_samples, mc_samples, signal_samples, histogram_names, legend_dict, xtitle_dict, output_directory)

if __name__ == "__main__":
    main()

    # %%