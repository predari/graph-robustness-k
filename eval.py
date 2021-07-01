#!/usr/bin/env python3

import simexpal
import pandas as pd
import yaml

def parse(run, f):
    output = yaml.load(f, Loader=yaml.Loader)
    if not output:
        return {}
    exps = output['Runs']
    exp = exps[0]
    d = {
        'experiment': run.experiment.name,
		'instance': run.instance.shortname,
        'nodes': exp['Nodes'],
        'edges': exp['Edges'],
        'k': exp['k'],
        'algorithm': exp['Algorithm'],
        'value': exp['Value'],
        'time': exp['Time'],
        'gain': exp['Gain'],
	}

    if d['gain'] < 0:
        d['gain'] *= -1.

    if 'Variant' in exp:
        d['variant'] = exp['Variant']

    if 'Linalg' in exp:
        d['linalg'] = exp['Linalg']
    
    if "Epsilon" in exp:
        d['epsilon'] = exp['Epsilon']
    
    if "Epsilon2" in exp:
        d['epsilon2'] = exp['Epsilon2']
    
    if "Heuristic" in exp:
        d['heuristic'] = exp['Heuristic']

    if "Threads" in exp:
        d['threads'] = exp['Threads']

    return d


cfg = simexpal.config_for_dir()
df = pd.DataFrame(cfg.collect_successful_results(parse))
#print(df.groupby('experiment').agg('mean'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def project(df, col, value):
    cols = [c for c in df.columns if c != col]
    return df[df[col] == value].filter(items=cols)

def print_df(df):
    print(df.to_string())


def plot_instance(df, instance_name, restrictions=[], filename=None):
    restricted_frame = project(df, "instance", instance_name)
    for k in restrictions:
        restricted_frame = project(restricted_frame, k[0], k[1])
        
    result_t = []
    result_gain = []
    result_name = []
    result_k = next(restricted_frame.iterrows())

    for index, row in restricted_frame.iterrows():
        experiment = row['experiment']
        algorithm_name = experiment
        heuristic = row['heuristic']
        if heuristic and pd.notnull(heuristic):
            algorithm_name += heuristic
        if heuristic == "Lpinv Diagonal":
            if row['linalg'] != 'LU':
                continue

        rename = {
            "greedy-3" : "main-resistances-exact",
            "sq-greedyLpinv Diagonal" : "main-resistances-approx",
            "sq-greedyRandom": "main-random",
            "sq-greedySimilarity": "main-similarity",
        }
        if algorithm_name in rename:
            result_name.append(rename[algorithm_name])
        else:
            result_name.append(algorithm_name)

        if algorithm_name == "random-averaged":
            result_t.append(0)
        else:
            result_t.append(row['time'])

        gain = row['gain']
        if gain < 0:
            gain *= -1
        result_gain.append(gain)


    len_results = len(result_t)
    x_pos = list(range(len(result_t)))

    # Sort by best gain
    x_pos.sort(key=lambda x: result_gain[x])
    x_pos_inv = [1]*len_results
    for i in range(len_results):
        x_pos_inv[x_pos[i]] = i
    x_pos = x_pos_inv


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(instance_name)
    fig.set_size_inches((10,10))
    fig._set_dpi(200)

    #ax1.set_xlabel('k')
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    #ax2.set_xscale('log')
    #ax2.set_yscale('log')
    ax1.set_ylabel('Total Effective Resistance Gain')
    #ax2.set_xlabel('k')
    ax2.set_ylabel('Time (s)')

    x_pos_shifted = [x - 0.5 for x in x_pos]
    ax1.bar(x_pos, result_gain, align='center', alpha = 0.5)
    ax2.bar(x_pos, result_t, align='center', alpha = 0.5)
    ax1.set_xticks(x_pos_shifted)
    ax1.set_xticklabels(result_name)
    ax2.set_xticks(x_pos_shifted)
    ax2.set_xticklabels(result_name)
    ax1.tick_params(axis='x', rotation=65)
    ax2.tick_params(axis='x', rotation=65)
    
    #ax1.legend()
    #ax2.legend()
    fig.tight_layout()
    if filename == None:
        filename = instance_name
    fig.savefig(filename + ".png")
    #plt.show()
    plt.close(fig)




    #for expname, r in results.items():
        # if expname.startswith("hillclimbing"):# or expname.startswith("random"):
        #     continue
        # r.sort(key=lambda x: x[0])
        # ks = np.array([x[0] for x in r])

        # def f(a, b):
        #     return (a-b)/b
        # values = np.array([f(r[i][1], subm_values[i]) for i in range(len(r))])
        # times = np.array([x[2] for x in r])

        # if (expname != "submodular-greedy"):
        #     ax1.plot(ks, values, 'o-', label=expname)
        # if expname != "random-averaged":
        #     ax2.plot(ks, times, 'o-', label=expname)
        # else:
        #     ax2.plot(np.array([]), np.array([]))

    #handles, labels = ax1.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    #plt.show()

#eval.plot_instance(eval.df, "arxiv-hephth", [["threads", 12], ["k", 20]])
def quick_plot(name, threads, k):
    plot_instance(df, name, [["threads", threads], ["k", 20]], name + "-" + str(k))

quick_plot("arxiv-hephth", 12, 20)
quick_plot("arxiv-heph", 12, 20)
quick_plot("arxiv-grqc", 12, 20)
quick_plot("facebook_ego_combined", 12, 20)
quick_plot("opsahl-powergrid", 12, 20)
quick_plot("deezer_europe", 12, 20)

quick_plot("arxiv-hephth", 12, 200)
quick_plot("arxiv-heph", 12, 200)
quick_plot("arxiv-grqc", 12, 200)
quick_plot("facebook_ego_combined", 12, 200)
quick_plot("opsahl-powergrid", 12, 200)
quick_plot("deezer_europe", 12, 200)
