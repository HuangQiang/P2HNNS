import os
import re
import numpy as np
import matplotlib.pylab as plt

from scipy.spatial     import ConvexHull
from itertools         import chain
from scipy.interpolate import interp1d
from collections       import defaultdict

from plot      import *
from plot_util import *


# ------------------------------------------------------------------------------
def plot_time_recall(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the querytime-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    fig_width, fig_height = calc_width_and_height(len(datasets), 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(1, len(datasets), di+1)
        plt.title(dataset_label)            # title
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        plt.xlabel('Recall (%)')            # label of x-axis
        if di == 0:                         # add label of y-axis at 1st dataset
            plt.ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall results
            time_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]

            time_recalls = np.array(time_recalls)
            # print(time_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            lower_recalls, lower_times = lower_bound_curve(time_recalls) 
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times)) 
            ax.semilogy(lower_recalls, lower_times, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_recall')
    plt.show()


# ------------------------------------------------------------------------------
def plot_fraction_recall(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the fraction-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    fig_width, fig_height = calc_width_and_height(len(datasets), 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(1, len(datasets), di+1)
        plt.title(dataset_label)            # title
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        plt.xlabel('Recall (%)')            # label of x-axis
        if di == 0:                         # add label of y-axis at 1st dataset
            plt.ylabel('Fraction (%)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get fraction-recall results
            fraction_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                fraction_recalls += [[getfraction(res), getrecall(res)]]

            fraction_recalls = np.array(fraction_recalls)
            # print(fraction_recalls)

            # get the fraction-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            # print('fraction_recall!!!!\n', fraction_recalls)
            lower_recalls, lower_fractions = lower_bound_curve(fraction_recalls) 
            miny = min(miny, np.min(lower_fractions))
            maxy = max(maxy, np.max(lower_fractions))
            ax.semilogy(lower_recalls, lower_fractions, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'fraction_recall')
    plt.show()


# ------------------------------------------------------------------------------
def plot_precision_recall(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the precision-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :returns: None
    '''
    fig_width, fig_height = calc_width_and_height(len(datasets), 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(1, len(datasets), di+1)
        plt.title(dataset_label)            # title
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        plt.xlabel('Recall (%)')            # label of x-axis
        if di == 0:                         # add label of y-axis for the 1st dataset
            plt.ylabel('Precision (%)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get precision-recall results
            precision_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                precision = getprecision(res)
                recall    = getrecall(res)
                if (recall > 0 and precision > 0): 
                    precision_recalls += [[precision, recall]]

            precision_recalls = np.array(precision_recalls)
            # print(precision_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            upper_recalls, upper_precisions = upper_bound_curve(precision_recalls, 1.0, True) 
            if len(upper_recalls) > 0:
                miny = min(miny, np.min(upper_precisions))
                maxy = max(maxy, np.max(upper_precisions)) 
                ax.semilogy(upper_recalls, upper_precisions, '-', 
                    color=method_color, marker=method_marker, 
                    label=method_label if di==0 else "", markevery=10, 
                    markerfacecolor='none', markersize=7, 
                    zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'precision_recall')
    plt.show()


# ------------------------------------------------------------------------------
def plot_time_recall_ratio(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the querytime-recall curves and querytime-ratio curves for all methods 
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
        plt.xlim(0, 100)
        
        ax_ratio = plt.subplot(2, n_datasets, n_datasets+di+1)
        plt.xlabel('Ratio')
        plt.xlim(1.0, 11.0) 
        plt.xticks([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])

        if di == 0:
            ax_recall.set_ylabel('Query Time (ms)')
            ax_ratio.set_ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get querytime-recall and querytime-ratio results from disk
            time_recalls = []
            time_ratios = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]
                time_ratios  += [[gettime(res), getratio(res)]]

            time_recalls = np.array(time_recalls)
            time_ratios  = np.array(time_ratios)
            # print(time_recalls, time_ratios)
            
            # get the querytime-recall curve by convex hull and interpolation
            lower_recalls, lower_times = lower_bound_curve(time_recalls)
            ax_recall.semilogy(lower_recalls, lower_times, '-', 
                color=method_color, marker=method_marker, 
                label=method_label if di==0 else "", markevery=10, 
                markerfacecolor='none', markersize=10)
            
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times))

            # get the querytime-ratio curve by convex hull
            upper_ratios, upper_times = upper_bound_curve(time_ratios, 0.2, False)
            ax_ratio.semilogy(upper_ratios, upper_times, '-', 
                color=method_color, marker=method_marker, label="", 
                markevery=5, markerfacecolor='none', markersize=10, 
                zorder=len(methods)-method_idx)
            
            miny = min(miny, np.min(upper_times))
            maxy = max(maxy, np.max(upper_times))
        
        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax_recall, miny, maxy)
        plt_helper.set_y_axis_log10(ax_ratio,  miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_recall_ratio')


# ------------------------------------------------------------------------------
def plot_time_index(chosen_top_k, recall_level, methods, input_folder, output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params recall_level: recall value for drawing figure (integer)
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
        ax_size = plt.subplot(2, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Index Size (MB)')       # label of x-axis

        ax_time = plt.subplot(2, n_datasets, n_datasets+di+1)
        plt.xlabel('Indexing Time (Seconds)')   # label of x-axis

        if di == 0:
            ax_size.set_ylabel('Query Time (ms)')
            ax_time.set_ylabel('Query Time (ms)')

        min_size_y = 1e9; max_size_y = -1e9
        min_time_y = 1e9; max_time_y = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

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
                    print('interp, ', querytime_at_recall, index_size, index_time)
            
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
                min_time_y = min(min_time_y, np.min(lower_qtimes))
                max_time_y = max(max_time_y, np.max(lower_qtimes))
                ax_time.semilogy(lower_itimes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label="", markerfacecolor='none', 
                    markersize=10, zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis 
        plt_helper.set_y_axis_log10(ax_size, min_size_y, max_size_y)
        plt_helper.set_y_axis_log10(ax_time, min_time_y, max_time_y)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_index')


# ------------------------------------------------------------------------------
def plot_time_indextime(chosen_top_k, recall_level, methods, input_folder, 
    output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params recall_level: recall value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up sub-figure
        ax_time = plt.subplot(1, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Indexing Time (Seconds)') # label of x-axis

        if di == 0:
            ax_time.set_ylabel('Query Time (ms)')

        miny = 1e9; maxy = -1e9
        minx = 1e9; maxx = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

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
                minx = min(minx, np.min(lower_itimes))
                maxx = max(maxx, np.max(lower_itimes))
                miny = min(miny, np.min(lower_qtimes))
                maxy = max(maxy, np.max(lower_qtimes))

                ax_time.semilogy(lower_itimes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=10, zorder=len(methods)-method_idx)

        # set up the limit (or range) of x-axis and y-axis
        if dataset == "Msong":
            plt_helper.set_x_axis(ax_time, minx, 0.02*maxx)
        else:
            plt_helper.set_x_axis(ax_time, minx, 0.22*maxx)
        plt_helper.set_y_axis_log10(ax_time, miny, maxy)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_indextime')


# ------------------------------------------------------------------------------
def plot_time_k(chosen_top_ks, recall_level, methods, input_folder, 
    output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_ks: top_k value for drawing figure (list)
    :params recall_level: recall value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up sub-figure
        ax_k = plt.subplot(1, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('$k$')                   # label of x-axis

        if di == 0:
            ax_k.set_ylabel('Query Time (ms)')

        miny = 1e9; maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

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

            miny = min(miny, np.min(querytimes_at_recall))
            maxy = max(maxy, np.max(querytimes_at_recall))
            ax_k.semilogy(chosen_ks, querytimes_at_recall, '-', 
                    color=method_color, marker=method_marker, 
                    label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=10, 
                    zorder=len(methods)-method_idx)
                    
        # set up the limit (or range) of y-axis 
        plt_helper.set_y_axis_log10(ax_k, miny, maxy)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_k')


# ------------------------------------------------------------------------------
def plot_nh_t(chosen_top_k, datasets, input_folder, output_folder, \
    fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=1.2, hspace=0.37)
    
    method = 'NH'
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_s=2
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if s == fix_s:
                print(m, s, cand, time, recall)
                data += [[m, s, cand, time, recall]]
        data = np.array(data)

        
        ms = [8, 16, 32, 64, 128, 256]
        maxy = -1e9
        miny = 1e9
        for color, marker, m in zip(method_colors, method_markers, ms):
            data_mp = data[data[:, 0]==m]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$t=%d$'%(m) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=3)
    plt_helper.plot_and_save(output_folder, 'varying_nh_t')


# ------------------------------------------------------------------------------
def plot_fh_m(chosen_top_k, datasets, input_folder, output_folder, \
    fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=1.2, hspace=0.37)
    
    method = 'FH'
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_l=4
        fix_s=2
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            l      = get_l(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if l == fix_l and s == fix_s:
                print(m, l, s, cand, time, recall)
                data += [[m, l, s, cand, time, recall]]
        data = np.array(data)

        
        ms = [8, 16, 32, 64, 128, 256]
        maxy = -1e9
        miny = 1e9
        for color, marker, m in zip(method_colors, method_markers, ms):
            data_mp = data[data[:, 0]==m]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$m=%d$'%(m) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=3)
    plt_helper.plot_and_save(output_folder, 'varying_fh_m')


# ------------------------------------------------------------------------------
def plot_fh_l(chosen_top_k, datasets, input_folder, output_folder, \
    fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    method = 'FH'
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_m=16
        fix_s=2
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            l      = get_l(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if m == fix_m and s == fix_s:
                print(m, l, s, cand, time, recall)
                data += [[m, l, s, cand, time, recall]]
        data = np.array(data)

        
        ls = [2, 4, 6, 8, 10]
        maxy = -1e9
        miny = 1e9
        for color, marker, l in zip(method_colors, method_markers, ls):
            data_mp = data[data[:, 1]==l]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$l=%d$'%(l) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=5)
    plt_helper.plot_and_save(output_folder, 'varying_fh_l')


# ------------------------------------------------------------------------------
def plot_fh_s(chosen_top_k, datasets, input_folder, output_folder, \
    fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    method = 'FH'
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_m=16
        fix_l=4
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            l      = get_l(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if m == fix_m and l == fix_l:
                print(m, l, s, cand, time, recall)
                data += [[m, l, s, cand, time, recall]]
        data = np.array(data)
        
        ss = [1, 2, 4, 8]
        maxy = -1e9
        miny = 1e9
        for color, marker, s in zip(method_colors, method_markers, ss):
            data_mp = data[data[:, 2]==s]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$\lambda=%d d$'%(s) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=4)
    plt_helper.plot_and_save(output_folder, 'varying_fh_s')

# ------------------------------------------------------------------------------
def plot_nh_s(chosen_top_k, datasets, input_folder, output_folder, \
    fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    method = 'NH'
    
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_m=256
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if m == fix_m:
                print(m, s, cand, time, recall)
                data += [[m, s, cand, time, recall]]
        data = np.array(data)
        
        ss = [1, 2, 4, 8]
        maxy = -1e9
        miny = 1e9
        for color, marker, s in zip(method_colors, method_markers, ss):
            data_mp = data[data[:, 1]==s]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$\lambda=%d d$'%(s) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=4)
    plt_helper.plot_and_save(output_folder, 'varying_nh_s')

# ------------------------------------------------------------------------------
if __name__ == '__main__':

    chosen_top_k  = 10
    input_folder  = "../results/"
    output_folder = "../figures/param/"
    datasets = ['Yelp', 'GloVe100']

    plot_nh_t(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.4)
    plot_nh_s(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.0)
    plot_fh_m(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.4)
    plot_fh_l(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.0)
    plot_fh_s(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.0)
