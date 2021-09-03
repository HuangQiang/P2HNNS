import os
import re
import glob
import numpy as np
import matplotlib.pylab as plt
import matplotlib

from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from itertools import chain, count
from collections import defaultdict

from os import makedirs
from os.path import isdir, isfile, join

from plot_util  import *
from plot_other import *

# ------------------------------------------------------------------------------
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

method_labels_map = { 
    'FH':               'FH',
    'FH_Minus':         'FH$^-$',
    'NH':               'NH',
    'FH_wo_S':          'FH-wo-S',
    'FH_Minus_wo_S':    'FH$^{-}$-wo-S',
    'NH_wo_S':          'NH-wo-S', 
    'EH':               'EH',
    'Orig_EH':          'EH',
    'BH':               'BH', 
    'Orig_BH':          'BH',
    'MH':               'MH',
    'Orig_MH':          'MH',
    'Random_Scan':      'Random-Scan',
    'Sorted_Scan':      'Sorted-Scan',
    'Linear':           'Linear-Scan'
}

dataset_labels_map = {
    'Yelp':         'Yelp',
    'Music':        'Music-100',
    'GloVe100':     'GloVe',
    'Tiny1M':       'Tiny-1M',
    'Msong':        'Msong',
    'MovieLens150': 'MovieLens',
    'Netflix300':   'Netflix',
    'Yahoo300':     'Yahoo',
    'Mnist':        'Mnist',
    'Sift':         'Sift',
    'Gaussian':     'Gaussian',
    'Gist':         'Gist',
}

# datasets = ['Yelp', 'GloVe100']
datasets = ['Yelp', 'Music', 'GloVe100', 'Tiny1M', 'Msong']
dataset_labels = [dataset_labels_map[dataset] for dataset in datasets]

method_colors  = ['red', 'blue', 'green', 'purple', 'deepskyblue', 'darkorange', 
    'olive', 'deeppink', 'dodgerblue', 'dimgray']
method_markers = ['o', '^', 's', 'd', '*', 'p', 'x', 'v', 'D', '>']


# ------------------------------------------------------------------------------
def calc_width_and_height(n_datasets, n_rows):
    '''
    calc the width and height of figure

    :params n_datasets: number of dataset (integer)
    :params n_rows: number of rows (integer)
    :returns: width and height of figure
    '''
    fig_width  = 0.55 + 3.333 * n_datasets
    fig_height = 0.80 + 2.5 * n_rows
    
    return fig_width, fig_height


# ------------------------------------------------------------------------------
def get_filename(input_folder, dataset_name, method_name):
    '''
    get the file prefix 'dataset_method'

    :params input_folder: input folder (string)
    :params dataset_name: name of dataset (string)
    :params method_name:  name of method (string)
    :returns: file prefix (string)
    '''
    name = '%s%s_%s.out' % (input_folder, dataset_name, method_name)
    return name


# ------------------------------------------------------------------------------
def parse_res(filename, chosen_top_k):
    '''
    parse result and get info such as ratio, qtime, recall, index_size, 
    chosen_k, and the setting of different methods
    
    BH: m=2, l=8, b=0.90
    Indexing Time: 2.708386 Seconds
    Estimated Memory: 347.581116 MB
    cand=10000
    1	5.948251	2.960960	0.000000	0.000000	0.844941
    5	4.475743	2.954690	0.400000	0.000200	0.845279
    10	3.891794	2.953910	0.900000	0.000899	0.845703
    20	3.289422	2.963460	0.950000	0.001896	0.846547
    50	2.642880	2.985980	0.900000	0.004478	0.849082
    100	2.244649	3.012860	0.800000	0.007922	0.853307

    cand=50000
    1	3.905541	14.901140	6.000000	0.000120	4.222926
    5	2.863510	14.905370	4.800000	0.000480	4.223249
    10	2.626913	14.910181	5.300000	0.001061	4.223649
    20	2.392440	14.913270	4.850000	0.001941	4.224458
    50	2.081206	14.931760	4.560000	0.004558	4.227065
    100	1.852284	14.964050	4.500000	0.008987	4.231267
    '''
    setting_pattern = re.compile(r'\S+\s+.*=.*')

    setting_m = re.compile(r'.*(m)=(\d+).*')
    setting_l = re.compile(r'.*(l)=(\d+).*')
    setting_M = re.compile(r'.*(M)=(\d+).*')
    setting_s = re.compile(r'.*(s)=(\d+).*')
    setting_b = re.compile(r'.*(b)=(\d+\.\d+).*')

    param_settings = [setting_m, setting_l, setting_M, setting_s, setting_b]

    index_time_pattern   = re.compile(r'Indexing Time: (\d+\.\d+).*')
    memory_usage_pattern = re.compile(r'Estimated Memory: (\d+\.\d+).*')
    candidate_pattern    = re.compile(r'.*cand=(\d+).*')
    records_pattern      = re.compile(r'(\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)')

    params = {}
    with open(filename, 'r') as f:
        for line in f:
            res = setting_pattern.match(line)
            if res:
                for param_setting in param_settings:
                    tmp_res = param_setting.match(line)
                    if tmp_res is not None:
                        # print(tmp_res.groups())
                        params[tmp_res.group(1)] = tmp_res.group(2)
                # print("setting=", line)

            res = index_time_pattern.match(line)
            if res:
                chosen_k = float(res.group(1))
                # print('chosen_k=', chosen_k)
            
            res = memory_usage_pattern.match(line)
            if res:
                memory_usage = float(res.group(1))
                # print('memory_usage=', memory_usage)

            res = candidate_pattern.match(line)
            if res:
                cand = int(res.group(1))
                # print('cand=', cand)
            
            res = records_pattern.match(line)
            if res:
                top_k     = int(res.group(1))
                ratio     = float(res.group(2))
                qtime     = float(res.group(3))
                recall    = float(res.group(4))
                precision = float(res.group(5))
                fraction  = float(res.group(6))
                # print(top_k, ratio, qtime, recall, precision, fraction)

                if top_k == chosen_top_k:
                    yield ((cand, params), (top_k, chosen_k, memory_usage, 
                        ratio, qtime, recall, precision, fraction))


# ------------------------------------------------------------------------------
def getindexingtime(res):
    return res[1]
def getindexsize(res):
    return res[2]
def getratio(res):
    return res[3]
def gettime(res):
    return res[4]
def getrecall(res):
    return res[5]
def getprecision(res):
    return res[6]
def getfraction(res):
    return res[7]

def get_cand(res):
    return int(res[0][0])
def get_l(res):
    return int(res[0][1]['l'])
def get_m(res):
    return int(res[0][1]['m'])
def get_s(res):
    return int(res[0][1]['s'])
def get_time(res):
    return float(res[1][4])
def get_recall(res):
    return float(res[1][5])
def get_precision(res):
    return float(res[1][6])
def get_fraction(res):
    return float(res[1][7])


# ------------------------------------------------------------------------------
def lower_bound_curve(xys):
    '''
    get the time-recall curve by convex hull and interpolation

    :params xys: 2-dim array (np.array)
    :returns: time-recall curve with interpolation
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)
    
    hull = ConvexHull(xys)
    hull_vs = xys[hull.vertices]
    # hull_vs = np.array(sorted(hull_vs, key=lambda x:x[1]))
    # print("hull_vs: ", hull_vs)

    # find max pair (maxv0) and min pairs (v1s) from the convex hull
    v1s = []
    maxv0 = [-1, -1]
    for v0, v1 in zip(hull_vs, chain(hull_vs[1:], hull_vs[:1])):
        # print(v0, v1)
        if v0[1] > v1[1] and v0[0] > v1[0]:
            v1s = np.append(v1s, v1, axis=-1)
            if v0[1] > maxv0[1]:
                maxv0 = v0
    # print(v1s, maxv0)
                
    # interpolation: vs[:, 1] -> recall (x), vs[:, 0] -> time (y)
    vs = np.array(np.append(maxv0, v1s)).reshape(-1, 2) # 2-dim array
    f = interp1d(vs[:, 1], vs[:, 0])

    minx = np.min(vs[:, 1]) + 1e-6
    maxx = np.max(vs[:, 1]) - 1e-6
    x = np.arange(minx, maxx, 1.0) # the interval of interpolation: 1.0
    y = list(map(f, x))          # get time (y) by interpolation

    return x, y


# ------------------------------------------------------------------------------
def upper_bound_curve(xys, interval, is_sorted):
    '''
    get the time-ratio and precision-recall curves by convex hull and interpolation

    :params xys: 2-dim array (np.array)
    :params interval: the interval of interpolation (float)
    :params is_sorted: sort the convex hull or not (boolean)
    :returns: curve with interpolation
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)
    
    xs = xys[:, 0]
    if len(xs) > 2 and xs[-1] > 0:
        hull = ConvexHull(xys)
        hull_vs = xys[hull.vertices]
        if is_sorted:
            hull_vs = np.array(sorted(hull_vs, key=lambda x:x[1]))
        print("hull_vs: ", hull_vs)

        # find max pair (maxv0) and min pairs (v1s) from the convex hull
        v1s = []
        maxv0 = [-1, -1]
        for v0, v1 in zip(hull_vs, chain(hull_vs[1:], hull_vs[:1])):
            # print(v0, v1)
            if v0[1] > v1[1] and v0[0] < v1[0]:
                v1s = np.append(v1s, v1, axis=-1)
                if v0[1] > maxv0[1]:
                    maxv0 = v0
        print(v1s, maxv0)

        # interpolation: vs[:, 1] -> recall (x), vs[:, 0] -> time (y)
        vs = np.array(np.append(maxv0, v1s)).reshape(-1, 2) # 2-dim array
        if len(vs) >= 2:
            f = interp1d(vs[:, 1], vs[:, 0])

            minx = np.min(vs[:, 1]) + 1e-6
            maxx = np.max(vs[:, 1]) - 1e-6
            x = np.arange(minx, maxx, interval)
            y = list(map(f, x))          # get time (y) by interpolation

            return x, y
        else:
            return xys[:, 0], xys[:, 1]
    else:
        return xys[:, 0], xys[:, 1]


# ------------------------------------------------------------------------------
def lower_bound_curve2(xys):
    '''
    get the querytime-indexsize and querytime-indextime curve by convex hull

    :params xys: 2-dim array (np.array)
    :returns: querytime-indexsize and querytime-indextime curve
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)

    xs = xys[:, 0]
    if len(xs) > 2 and xs[-1] > 0:
        # conduct convex hull to find the curve
        hull = ConvexHull(xys)
        hull_vs = xys[hull.vertices]
        # print("hull_vs: ", hull_vs)
        
        ret_vs = []
        for v0, v1, v2 in zip(chain(hull_vs[-1:], hull_vs[:-1]), hull_vs, \
            chain(hull_vs[1:], hull_vs[:1])):

            # print(v0, v1, v2)
            if v0[0] < v1[0] or v1[0] < v2[0]:
                ret_vs = np.append(ret_vs, v1, axis=-1)

        # sort the results in ascending order of x without interpolation
        ret_vs = ret_vs.reshape((-1, 2))
        ret_vs = np.array(sorted(ret_vs, key=lambda x:x[0]))

        return ret_vs[:, 0], ret_vs[:, 1]
    else:
        return xys[:, 0], xys[:, 1]


# ------------------------------------------------------------------------------
def plot_time_fraction_recall(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the querytime-recall curves and fraction-recall curves for all methods 
    on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 2)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up two sub-figures
        ax_recall = plt.subplot(2, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Recall (%)')            # label of x-axis
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        
        ax_fraction = plt.subplot(2, n_datasets, n_datasets+di+1)
        plt.xlabel('Recall (%)')            # label of x-axis
        plt.xlim(0, 100)                    # limit (or range) of x-axis

        if di == 0:
            ax_recall.set_ylabel('Query Time (ms)')
            ax_fraction.set_ylabel('Fraction (%)')

        min_t_y = 1e9; max_t_y = -1e9
        min_f_y = 1e9; max_f_y = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall and fraction-recall results from disk
            time_recalls     = []
            fraction_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls     += [[gettime(res),     getrecall(res)]]
                fraction_recalls += [[getfraction(res), getrecall(res)]]

            time_recalls     = np.array(time_recalls)
            fraction_recalls = np.array(fraction_recalls)
            # print(time_recalls, fraction_recalls)
            
            # get the time-recall curve by convex hull and interpolation
            lower_recalls, lower_times = lower_bound_curve(time_recalls)
            min_t_y = min(min_t_y, np.min(lower_times))
            max_t_y = max(max_t_y, np.max(lower_times))
            ax_recall.semilogy(lower_recalls, lower_times, '-', 
                color=method_color, marker=method_marker, 
                label=method_label if di==0 else "", markevery=10, 
                markerfacecolor='none', markersize=10)

            # get the fraction-recall curve by convex hull
            lower_recalls, lower_fractions = lower_bound_curve(fraction_recalls)
            min_f_y = min(min_f_y, np.min(lower_fractions))
            max_f_y = max(max_f_y, np.max(lower_fractions))
            ax_fraction.semilogy(lower_recalls, lower_fractions, '-', 
                color=method_color, marker=method_marker, label="", 
                markevery=10, markerfacecolor='none', markersize=10, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax_recall,   min_t_y, max_t_y)
        plt_helper.set_y_axis_log10(ax_fraction, min_f_y, max_f_y)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_fraction_recall')


# ------------------------------------------------------------------------------
def plot_time_index_k(chosen_top_k, chosen_top_ks, recall_level, size_x_scales,\
    time_x_scales, methods, input_folder, output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params chosen_top_ks: a list of op_k values for drawing figure (list)
    :params recall_level: recall value for drawing figure (integer)
    :params size_x_scales: a list of x scales for index size (list)
    :params time_x_scales: a list of x scales for indexing time (list)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 3)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up three sub-figures
        ax_size = plt.subplot(3, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Index Size (MB)')       # label of x-axis

        ax_time = plt.subplot(3, n_datasets, n_datasets+di+1)
        plt.xlabel('Indexing Time (Seconds)') # label of x-axis

        ax_k = plt.subplot(3, n_datasets, 2*n_datasets+di+1)
        plt.xlabel('$k$')                   # label of x-axis

        if di == 0:
            ax_size.set_ylabel('Query Time (ms)')
            ax_time.set_ylabel('Query Time (ms)')
            ax_k.set_ylabel('Query Time (ms)')

        min_size_x = 1e9; max_size_x = -1e9
        min_size_y = 1e9; max_size_y = -1e9
        min_time_x = 1e9; max_time_x = -1e9
        min_time_y = 1e9; max_time_y = -1e9
        min_k_y    = 1e9; max_k_y    = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # ------------------------------------------------------------------
            #  query time vs. index size and indexing time
            # ------------------------------------------------------------------
            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for _,res in parse_res(filename, chosen_top_k):
                query_time = gettime(res)
                recall     = getrecall(res)
                index_time = getindexingtime(res)
                index_size = getindexsize(res)
                chosen_ks_dict[(index_time, index_size)] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            index_times, index_sizes, querytimes_at_recall = [], [], []
            for (index_time, index_size), recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] + recall_querytimes_)

                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                if np.max(recalls) > recall_level:
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)

                    # update results
                    index_times += [index_time]
                    index_sizes += [index_size]
                    querytimes_at_recall += [querytime_at_recall]
                    # print('interp, ', querytime_at_recall, index_size, index_time)
            
            index_times = np.array(index_times)
            index_sizes = np.array(index_sizes)
            querytimes_at_recall = np.array(querytimes_at_recall)
          
            # get the querytime-indexsize curve by convex hull
            isize_qtime = np.zeros(shape=(len(index_sizes), 2))
            isize_qtime[:, 0] = index_sizes
            isize_qtime[:, 1] = querytimes_at_recall

            lower_isizes, lower_qtimes = lower_bound_curve2(isize_qtime)
            if len(lower_isizes) > 0:
                # print(method, lower_isizes, lower_qtimes)
                min_size_x = min(min_size_x, np.min(lower_isizes))
                max_size_x = max(max_size_x, np.max(lower_isizes))
                min_size_y = min(min_size_y, np.min(lower_qtimes))
                max_size_y = max(max_size_y, np.max(lower_qtimes))
                ax_size.semilogy(lower_isizes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=10)

                # get the querytime-indextime curve by convex hull
                itime_qtime = np.zeros(shape=(len(index_times), 2))
                itime_qtime[:, 0] = index_times
                itime_qtime[:, 1] = querytimes_at_recall

                lower_itimes, lower_qtimes = lower_bound_curve2(itime_qtime)
                # print(method, lower_itimes, lower_qtimes)
                min_time_x = min(min_time_x, np.min(lower_itimes))
                max_time_x = max(max_time_x, np.max(lower_itimes))
                min_time_y = min(min_time_y, np.min(lower_qtimes))
                max_time_y = max(max_time_y, np.max(lower_qtimes))
                ax_time.semilogy(lower_itimes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label="", markerfacecolor='none', 
                    markersize=10, zorder=len(methods)-method_idx)

            # ------------------------------------------------------------------
            #  query time vs. k
            # ------------------------------------------------------------------
            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for chosen_top_k in chosen_top_ks:
                for _,res in parse_res(filename, chosen_top_k):
                    query_time = gettime(res)
                    recall     = getrecall(res)
                    chosen_ks_dict[chosen_top_k] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            chosen_ks, querytimes_at_recall = [], []
            for chosen_k, recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] + recall_querytimes_)

                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                if np.max(recalls) > recall_level:
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)

                    # update results
                    chosen_ks += [chosen_k]
                    querytimes_at_recall += [querytime_at_recall]

            chosen_ks = np.array(chosen_ks)
            querytimes_at_recall = np.array(querytimes_at_recall)

            min_k_y = min(min_k_y, np.min(querytimes_at_recall))
            max_k_y = max(max_k_y, np.max(querytimes_at_recall))
            ax_k.semilogy(chosen_ks, querytimes_at_recall, '-', color=method_color, 
                marker=method_marker, label="", markerfacecolor='none', 
                markersize=10, zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis 
        plt_helper.set_x_axis(ax_size, min_size_x, size_x_scales[di]*max_size_x)
        plt_helper.set_y_axis_log10(ax_size, min_size_y, max_size_y)

        plt_helper.set_x_axis(ax_time, min_time_x, time_x_scales[di]*max_time_x)
        plt_helper.set_y_axis_log10(ax_time, min_time_y, max_time_y)
        
        plt_helper.set_y_axis_log10(ax_k, min_k_y, max_k_y)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_index_k_%d' % recall_level)


# ------------------------------------------------------------------------------
def plot_time_recall_indextime(chosen_top_k, recall_level, time_x_scales, \
    methods, input_folder, output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params recall_level: recall value for drawing figure (integer)
    :params time_x_scales: a list of x scales for indexing time (list)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 2)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up sub-figure
        ax_recall = plt.subplot(2, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        plt.xlabel('Recall (%)')            # label of x-axis

        ax_time = plt.subplot(2, n_datasets, n_datasets+di+1)
        plt.xlabel('Indexing Time (Seconds)') # label of x-axis

        if di == 0:
            ax_recall.set_ylabel('Query Time (ms)')
            ax_time.set_ylabel('Query Time (ms)')

        min_r_y = 1e9; max_r_y = -1e9
        min_t_x = 1e9; max_t_x = -1e9
        min_t_y = 1e9; max_t_y = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # ------------------------------------------------------------------
            # query time vs. recall
            # ------------------------------------------------------------------
            time_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]

            time_recalls = np.array(time_recalls)
            # print(time_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            lower_recalls, lower_times = lower_bound_curve(time_recalls) 
            min_r_y = min(min_r_y, np.min(lower_times))
            max_r_y = max(max_r_y, np.max(lower_times)) 
            ax_recall.semilogy(lower_recalls, lower_times, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

            # ------------------------------------------------------------------
            #  query time vs. indexing time
            # ------------------------------------------------------------------
            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for _,res in parse_res(filename, chosen_top_k):
                query_time = gettime(res)
                recall     = getrecall(res)
                index_time = getindexingtime(res)

                chosen_ks_dict[index_time] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            index_times, querytimes_at_recall = [], []
            for index_time, recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] +recall_querytimes_)
                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                
                if np.max(recalls) > recall_level:                    
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)

                    # update results
                    index_times += [index_time]
                    querytimes_at_recall += [querytime_at_recall]
                    # print('interp, ', querytime_at_recall, index_time)
            
            index_times = np.array(index_times)
            querytimes_at_recall = np.array(querytimes_at_recall)

            # get the querytime-indextime curve by convex hull
            itime_qtimes = np.zeros(shape=(len(index_times), 2))
            itime_qtimes[:, 0] = index_times
            itime_qtimes[:, 1] = querytimes_at_recall

            lower_itimes, lower_qtimes = lower_bound_curve2(itime_qtimes)
            if len(lower_itimes) > 0:
                # print(method, lower_itimes, lower_qtimes)
                min_t_x = min(min_t_x, np.min(lower_itimes))
                max_t_x = max(max_t_x, np.max(lower_itimes))
                min_t_y = min(min_t_y, np.min(lower_qtimes))
                max_t_y = max(max_t_y, np.max(lower_qtimes))

                ax_time.semilogy(lower_itimes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label="", markerfacecolor='none', 
                    markersize=10, zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis 
        plt_helper.set_y_axis_log10(ax_recall, min_r_y, max_r_y)
        
        plt_helper.set_x_axis(ax_time, min_t_x, time_x_scales[di]*max_t_x)
        if max_t_y / min_t_y < 10:
            plt_helper.set_y_axis_close(ax_time, min_t_y, max_t_y)
        else:
            plt_helper.set_y_axis_log10(ax_time, min_t_y, max_t_y)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_recall_indextime_%d' % 
        recall_level)


# ------------------------------------------------------------------------------
def plot_params(chosen_top_k, dataset, input_folder, output_folder, \
    fig_width=19.2, fig_height=3.0):
    '''
    draw the querytime-recall curves for the parameters of NH and FH

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params dataset: name of dataset (string)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :params fig_width:  the width  of a figure (float)
    :params fig_height: the height of a figure (float)
    :returns: None
    '''
    plt.figure(figsize=(fig_width, fig_height))
    plt.rcParams.update({'font.size': 13})

    left_space   = 0.80
    bottom_space = 0.55
    top_space    = 0.30  # 1.2
    right_space  = 0.25
    width_space  = 0.24
    height_space = 0.37

    bottom = bottom_space / fig_height
    top    = (fig_height - top_space) / fig_height
    left   = left_space / fig_width
    right  = (fig_width - right_space) / fig_width 
    plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right, 
        wspace=width_space, hspace=height_space)

    # --------------------------------------------------------------------------
    # NH on t (\lambda = 2d)
    # --------------------------------------------------------------------------
    method = 'NH'
    ax = plt.subplot(1, 5, 1)
    ax.set_xlabel(r'Recall (%)')
    ax.set_ylabel(r'Query Time (ms)')
    ax.set_title('Impact of $t$ for %s' % method, fontsize=16)
    filename = get_filename(input_folder, dataset, method)
    print(filename, method, dataset)

    fix_s=2
    data = []
    for record in parse_res(filename, chosen_top_k):
        m      = get_m(record)
        s      = get_s(record)
        cand   = get_cand(record)
        time   = get_time(record)
        recall = get_recall(record)
        if s == fix_s: 
            data += [[m, s, cand, time, recall]]
    data = np.array(data)

    legend_name = ['$t=8$', '$t=16$', '$t=32$', '$t=64$', '$t=128$', '$t=256$']
    ms = [8, 16, 32, 64, 128, 256]
    for color, marker, m in zip(method_colors, method_markers, ms):
        data_mp = data[data[:, 0]==m]
        ax.plot(data_mp[:, -1], data_mp[:, -2], marker=marker,c=color, 
            markerfacecolor='none', markersize=7)

    plt.legend(legend_name, loc='upper right', ncol=2, fontsize=13)
    plt.xlim(0, 100)
    ax.set_yscale('log')
    # plt.ylim(1e-2, 1e3) # Yelp
    plt.ylim(1e-1, 1e5) # GloVe100

    # --------------------------------------------------------------------------
    # NH on \lambda (t = 256)
    # --------------------------------------------------------------------------
    method = 'NH'
    ax = plt.subplot(1, 5, 2)
    ax.set_xlabel(r'Recall (%)')
    ax.set_title('Impact of $\lambda$ for %s' % method, fontsize=16)
    filename = get_filename(input_folder, dataset, method)
    print(filename, method, dataset)

    fix_m=256
    data = []
    for record in parse_res(filename, chosen_top_k):
        m      = get_m(record)
        s      = get_s(record)
        cand   = get_cand(record)
        time   = get_time(record)
        recall = get_recall(record)
        if m == fix_m:
            data += [[m, s, cand, time, recall]]
    data = np.array(data)

    legend_name = ['$\lambda=1d$', '$\lambda=2d$', '$\lambda=4d$', '$\lambda=8d$']
    ss = [1, 2, 4, 8]
    for color, marker, s in zip(method_colors, method_markers, ss):
        data_mp = data[data[:, 1]==s]
        ax.plot(data_mp[:, -1], data_mp[:, -2], marker=marker, c=color, 
            markerfacecolor='none', markersize=7)

    plt.legend(legend_name, loc='upper right', ncol=2, fontsize=13)
    plt.xlim(0, 100)
    ax.set_yscale('log')
    # plt.ylim(1e-1, 1e2) # Yelp
    plt.ylim(1e-1, 1e5) # GloVe100

    # --------------------------------------------------------------------------
    #  FH on m (l = 4 and \lambda = 2d)
    # --------------------------------------------------------------------------
    method = 'FH'
    ax = plt.subplot(1, 5, 3)
    ax.set_xlabel(r'Recall (%)')
    ax.set_title('Impact of $m$ for %s' % method, fontsize=16)
    filename = get_filename(input_folder, dataset, method)
    print(filename, method, dataset)
    
    fix_l=4; fix_s=2
    data = []
    for record in parse_res(filename, chosen_top_k):
        m      = get_m(record)
        l      = get_l(record)
        s      = get_s(record)
        cand   = get_cand(record)
        time   = get_time(record)
        recall = get_recall(record)
        if l == fix_l and s == fix_s:
            data += [[m, l, s, cand, time, recall]]
    data = np.array(data)

    legend_name = ['$m=8$', '$m=16$', '$m=32$', '$m=64$', '$m=128$', '$m=256$']
    ms = [8, 16, 32, 64, 128, 256]
    for color, marker, m in zip(method_colors, method_markers, ms):
        data_mp = data[data[:, 0]==m]
        ax.plot(data_mp[:, -1], data_mp[:, -2], marker=marker, c=color, 
            markerfacecolor='none', markersize=7)

    plt.legend(legend_name, loc='upper right', ncol=2, fontsize=13)
    plt.xlim(0, 100)
    ax.set_yscale('log')
    # plt.ylim(1e-2, 1e2) # Yelp
    plt.ylim(1e-2, 1e4) # GloVe100

    # --------------------------------------------------------------------------
    #  FH on l (m = 16 and \lambda = 2d)
    # --------------------------------------------------------------------------
    method = 'FH'
    ax = plt.subplot(1, 5, 4)
    ax.set_xlabel(r'Recall (%)')
    ax.set_title('Impact of $l$ for %s' % method, fontsize=16)
    filename = get_filename(input_folder, dataset, method)
    print(filename, method, dataset)
    
    fix_m=16; fix_s=2
    data = []
    for record in parse_res(filename, chosen_top_k):
        m      = get_m(record)
        l      = get_l(record)
        s      = get_s(record)
        cand   = get_cand(record)
        time   = get_time(record)
        recall = get_recall(record)
        if m == fix_m and s == fix_s: 
            data += [[m, l, s, cand, time, recall]]
    data = np.array(data)

    legend_name = ['$l=2$', '$l=4$', '$l=6$', '$l=8$', '$l=10$']
    ls = [2, 4, 6, 8, 10]
    for color, marker, l in zip(method_colors, method_markers, ls):
        data_mp = data[data[:, 1]==l]
        ax.plot(data_mp[:, -1], data_mp[:, -2], marker=marker, c=color, 
            markerfacecolor='none', markersize=7)
    
    plt.legend(legend_name, loc='upper right', ncol=2, fontsize=13)
    plt.xlim(0, 100)
    ax.set_yscale('log')
    # plt.ylim(1e-2, 1e2) # Yelp
    plt.ylim(1e-2, 1e4) # GloVe100

    # --------------------------------------------------------------------------
    #  FH on \lambda (m = 16 and l = 4)
    # --------------------------------------------------------------------------
    method = 'FH'
    ax = plt.subplot(1, 5, 5)
    ax.set_xlabel(r'Recall (%)')
    ax.set_title('Impact of $\lambda$ for %s' % method, fontsize=16)
    filename = get_filename(input_folder, dataset, method)
    print(filename, method, dataset)
    
    fix_m=16; fix_l=4
    data = []
    for record in parse_res(filename, chosen_top_k):
        m      = get_m(record)
        l      = get_l(record)
        s      = get_s(record)
        cand   = get_cand(record)
        time   = get_time(record)
        recall = get_recall(record)
        if m == fix_m and l == fix_l:
            data += [[m, l, s, cand, time, recall]]
    data = np.array(data)

    legend_name = ['$\lambda=1d$', '$\lambda=2d$', '$\lambda=4d$', '$\lambda=8d$']
    ss = [1, 2, 4, 8]
    miny = 1e9; maxy = -1e9
    for color, marker, s in zip(method_colors, method_markers, ss):
        data_mp = data[data[:, 2]==s]
        ax.plot(data_mp[:, -1], data_mp[:, -2], marker=marker, c=color, 
            markerfacecolor='none', markersize=7)

    plt.legend(legend_name, loc='upper right', ncol=2, fontsize=13)
    plt.xlim(0, 100)
    ax.set_yscale('log')
    # plt.ylim(1e-2, 1e2) # Yelp
    plt.ylim(1e-2, 1e4) # GloVe100

    # --------------------------------------------------------------------------
    #  save and show figure
    # --------------------------------------------------------------------------
    plt.savefig('%s.png' % join(output_folder, 'params_%s' % dataset))
    plt.savefig('%s.eps' % join(output_folder, 'params_%s' % dataset))
    plt.savefig('%s.pdf' % join(output_folder, 'params_%s' % dataset))
    plt.show()


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    chosen_top_k  = 10
    chosen_top_ks = [1,5,10,20,50,100]

    # 1. plot curves of time vs. recall & fraction vs. recall
    input_folder  = "../results/"
    output_folder = "../figures/competitors/"
    methods       = ['FH', 'FH_Minus', 'NH', 'BH', 'MH', 'Random_Scan', 'Sorted_Scan']
    plot_time_fraction_recall(chosen_top_k, methods, input_folder, output_folder)

    # 2. plot curves of time vs. index (size and time) & time vs. k
    input_folder  = "../results/"
    output_folder = "../figures/competitors/"
    methods       = ['FH', 'FH_Minus', 'NH', 'BH', 'MH', 'Random_Scan', 'Sorted_Scan']
    size_x_scales = [0.3,0.3,0.3,0.3,0.3]; time_x_scales = [0.1,0.1,0.1,0.3,0.05]
    recall_levels = [80,70,60,50]
    for recall_level in recall_levels:
        plot_time_index_k(chosen_top_k, chosen_top_ks, recall_level, size_x_scales,
            time_x_scales, methods, input_folder, output_folder)

    # 3. plot curves of time vs. recall & time vs. indexing time
    input_folder  = "../results/"
    output_folder = "../figures/sampling/"
    methods       = ['FH', 'FH_Minus', 'NH', 'FH_wo_S', 'FH_Minus_wo_S', 'NH_wo_S']
    time_x_scales = [0.2, 0.1, 0.1, 0.2, 0.02]
    recall_levels = [80,70,60,50]
    for recall_level in recall_levels:
        plot_time_recall_indextime(chosen_top_k, recall_level, time_x_scales, 
            methods, input_folder, output_folder)

    # 4. plot parameters
    chosen_top_k  = 10
    datasets      = ['GloVe100', 'Music', 'Msong', 'Yelp', 'Tiny1M']
    input_folder  = "../results/"
    output_folder = "../figures/param/"
    for dataset in datasets:
        plot_params(chosen_top_k, dataset, input_folder, output_folder)

    # 5. plot curves of time vs. recall & time vs. indexing time for normalized data
    input_folder  = "../results_normalized/"
    output_folder = "../figures/normalized/"
    methods       = ['FH', 'NH', 'Orig_BH', 'Orig_MH']
    
    recall_level  = 50; time_x_scales = [0.1, 0.1, 0.1, 0.05, 0.02]
    plot_time_recall_indextime(chosen_top_k, recall_level, time_x_scales, methods, 
        input_folder, output_folder)

    recall_level  = 60; time_x_scales = [0.1, 0.1, 0.1, 0.05, 0.02]
    plot_time_recall_indextime(chosen_top_k, recall_level, time_x_scales, methods, 
        input_folder, output_folder)

    recall_level  = 70; time_x_scales = [0.1, 0.2, 0.1, 0.1, 0.04]
    plot_time_recall_indextime(chosen_top_k, recall_level, time_x_scales, methods, 
        input_folder, output_folder)
    
    recall_level  = 80; time_x_scales = [0.1, 0.2, 0.1, 0.1, 0.08]
    plot_time_recall_indextime(chosen_top_k, recall_level, time_x_scales, methods, 
        input_folder, output_folder)
    