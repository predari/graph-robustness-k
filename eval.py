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

    d = {}
    if "JLT-Test" in exp:
        d = {
            "instance": run.instance.shortname,
            'experiment': "JLT-Test",
            'nodes': exp['Nodes'],
            'edges': exp['Edges'],
            'rel-errors': exp['Rel-Errors']
        }
    else:
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
import matplotlib
import pandas as pd

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})



def project(df, col, value):
    cols = [c for c in df.columns if c != col]
    return df[df[col] == value].filter(items=cols)

def print_df(df):
    print(df.to_string())

def restrict_frame(restrictions):
    restricted_frame = df
    for k in restrictions:
        restricted_frame = project(restricted_frame, k[0], k[1])
    return restricted_frame


def plot_averaged(df, instance_names, experiment_restriction_list, experiment_names, filename=None):
    colors = ['b', 'g', 'r', (0.0, 0.7, 0.7)]
    restricted_frame = df
    
    ks = set()
    resistances_by_k = {}
    times_by_k = {}

    def analyze_experiment(restrictions):
        resistances = {}
        times = {}

        restricted_frame = restrict_frame(restrictions)
        for instance_name in instance_names:
            instance_frame = project(restricted_frame, "instance", instance_name)

            def insert(d, k1, v):
                if k1 not in d:
                    d[k1] = {}
                d[k1][instance_name] = v

            for row in instance_frame.iterrows():
                row = row[1]

                k = row['k']
                ks.add(k)

                insert(resistances, k, row['gain'])
                insert(times, k, row['time'])

        return resistances, times

    result_resistances = []
    result_times = []
    
    for restrictions in experiment_restriction_list:
        res, times = analyze_experiment(restrictions)
        result_resistances.append(res)
        result_times.append(times)
    
    submodular_resistances, submodular_times = analyze_experiment([["threads", 12], ["experiment", "submodular-greedy"]])


    x_pos = np.arange(len(ks))
    ks = sorted(list(ks))


    fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle(instance_name)
    fig._set_dpi(400)
    fig.set_size_inches((6, 4))

    ax1.set_xlabel('k')
    ax1.set_ylabel('Gain rel. Submodular')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Time rel. Submodular')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ks)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ks)
    ax1.set_ylim(0.9, 1.)
    ax2.set_ylim(0., 0.5)
    
    #ax2.legend(experiment_names)


    def mean(l):
        p = 1.
        for v in l:
            p *= v
        return p ** (1. / len(l))
    
    def offset(l, k):
        width = 0.8 / k
        return width * l - width / 2 * k + width / 2

    num_experiments = len(result_resistances)

    for j, (algorithm_resistances, algorithm_times) in enumerate(zip(result_resistances, result_times)):
        resistance_means = []
        time_means = []
        for k in ks:
            algorithm_k_resistances = algorithm_resistances[k]
            algorithm_k_times = algorithm_times[k]
            relative_resistances = []
            relative_times = []
            for instance_name in instance_names:
                relative_resistances.append(algorithm_k_resistances[instance_name] / submodular_resistances[k][instance_name])
                relative_times.append(algorithm_k_times[instance_name] / submodular_times[k][instance_name])

            res_mean = mean(relative_resistances)
            time_mean = mean(relative_times)
            resistance_means.append(res_mean)
            time_means.append(time_mean)
            print("Taking means of {} instances for k = {} and j = {}.".format(len(relative_resistances), k, j))
        
        ax1.bar(x_pos + offset(j, num_experiments), resistance_means, align='center', color=colors[j], alpha = 0.5, label=experiment_names[j], width=0.8 / num_experiments)
        ax2.bar(x_pos + offset(j, num_experiments), time_means, align='center', color=colors[j], alpha = 0.5, label=experiment_names[j], width=0.8 / num_experiments)

    
    #ax1.legend()
    #ax2.legend()
    plt.legend(ncol = 1, bbox_to_anchor=(1, 1), loc='lower right')

    fig.tight_layout()
    if filename == None:
        filename = "results_aggregated"
    
    fig.savefig(filename + ".pgf", transparent=True)
    fig.savefig(filename + ".png")
    #plt.show()

    plt.close(fig)




plot_averaged(df, ["deezer_europe", "opsahl-powergrid", "arxiv-grqc", "facebook_ego_combined", "arxiv-hephth", "arxiv-heph"], [ \
    [["experiment", "sq-greedy"], ["heuristic", "Lpinv Diagonal"], ["linalg", "LU"], ["threads", 12]] \
    ], ["Main-Resistances-Approx"], "results_aggregated_1")

plot_averaged(df, ["deezer_europe", "opsahl-powergrid", "arxiv-grqc", "facebook_ego_combined", "arxiv-hephth", "arxiv-heph"], [ \
    [["experiment", "sq-greedy"], ["heuristic", "Lpinv Diagonal"], ["linalg", "LU"], ["threads", 12]], \
    [["experiment", "stochastic-greedy"], ["threads", 12]], \
    [["experiment", "sq-greedy"], ["heuristic", "Similarity"], ["linalg", "LU"], ["threads", 12]], \
    ], ["Main-Resistances-Approx", "Stochastic-Submodular", "Main-Similarity"], "results_aggregated_3")

plot_averaged(df, ["deezer_europe", "opsahl-powergrid", "arxiv-grqc", "facebook_ego_combined", "arxiv-hephth", "arxiv-heph"], [ \
    [["experiment", "sq-greedy"], ["heuristic", "Lpinv Diagonal"], ["linalg", "LU"], ["threads", 12]], \
    [["experiment", "stochastic-greedy"], ["threads", 12]], \
    [["experiment", "sq-greedy"], ["heuristic", "Similarity"], ["linalg", "LU"], ["threads", 12]], \
    [["experiment", "sq-greedy"], ["heuristic", "Random"], ["linalg", "LU"], ["threads", 12]] \
    ], ["Main-Resistances-Approx", "Stochastic-Submodular", "Main-Similarity", "Main-Random"], "results_aggregated_4")








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
    fig.set_size_inches((4,2))
    fig._set_dpi(80)

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
    fig.savefig(filename + ".png", transparent=True)
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
    plot_instance(df, name, [["threads", threads], ["k", k]], name + "-" + str(k))

if False:
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
