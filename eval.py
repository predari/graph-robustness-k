#!/usr/bin/env python3

import simexpal
import pandas as pd
import yaml
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


colors = ['b', 'g', 'r', (0.0, 0.7, 0.7), (0.7, 0.7, 0.), (0.7, 0., 0.7), (0.8, 0.4, 0.1), (0.1, 0.8, 0.4), (0.4, 0.1, 0.8)]
markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']

def parse(run, f):
    output = yaml.load(f, Loader=yaml.Loader)
    if not output:
        return {}
    exps = output['Runs']
    exp = exps[0]

    d = {}
    d = exp.copy()
    d['Instance'] = run.instance.shortname
    d["Experiment"] =  run.experiment.name

    if "JLT-Test" in exp:
        d["Experiment"] = "JLT-Test"
        d["JLT-Test"] = True

        # d = {
        #     "instance": run.instance.shortname,
        #     'experiment': "JLT-Test",
        #     'nodes': exp['Nodes'],
        #     'edges': exp['Edges'],
        #     'rel-errors': exp['Rel-Errors'],
        #     'jlt-test': True
        # }
    else:
        if 'gain' in d and d['gain'] < 0:
            d['gain'] *= -1.
        # d = {
        #     'experiment': run.experiment.name,
        #     'instance': run.instance.shortname,
        #     'nodes': exp['Nodes'],
        #     'edges': exp['Edges'],
        #     'k': exp['k'],
        #     'algorithm': exp['Algorithm'],
        #     'value': exp['Value'],
        #     'time': exp['Time'],
        #     'gain': exp['Gain'],
        # }



    return d



cfg = simexpal.config_for_dir()
df = pd.DataFrame(cfg.collect_successful_results(parse))

def print_df(df):
    print(df.to_string())



jlt_df = df[df['JLT-Test'] == True]
df = df[df['JLT-Test'].isnull()]

#print(df.groupby('Experiment').agg('mean'))


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def d_to_str(d):
    return f"{d:.4f}"

def project(df, col, value):
    cols = [c for c in df.columns if c != col]
    return df[df[col] == value].filter(items=cols)


def restrict_frame(df, restrictions):
    restricted_frame = df
    for k, v in restrictions.items():
        restricted_frame = project(restricted_frame, k, v)
    return restricted_frame

def output_file(fig, filename):
    fig.savefig(filename + ".png")
    fig.savefig(filename + ".pgf", transparent=True)
    print("Write file " + filename + ".pgf, " + filename + ".png")


def draw_jlt(df):
    jlt_test_results = project(df, "JLT-Test", True)
    instances = set(jlt_test_results["Instance"].tolist())

    err_data = []
    err_data2 = []


    for instance_name in instances:
        inst_fr = project(jlt_test_results, "Instance", instance_name)
        err_data.append(np.array(inst_fr["Rel-Errors"].tolist()[0]))
        err_data2.append(np.array(inst_fr['Rel-Errors-2'].tolist()[0]))


    fig, ax = plt.subplots()
    ax.set_ylim(0., 1.)

    #ax.set_title("Relative Errors of JLT ")
    ax.boxplot(err_data, labels=instances, whis=[2.5,97.5])
    plt.ylabel('Rel. Error')
    plt.xticks(rotation=90)
    fig.tight_layout()

    output_file(fig, "jlt-test-2")    
    
    plt.cla()
    plt.clf()

    ax.set_ylim(0., 1.)

    fig, ax = plt.subplots()

    #ax.set_title("Relative Errors of JLT wrt. node subset. ")
    ax.boxplot(err_data2, labels=instances, whis=[2.5, 97.5])
    plt.ylabel('Rel. Error')
    fig.tight_layout()
    plt.xticks(rotation=90)
    fig.tight_layout()

    output_file(fig, "jlt-test")    

    plt.cla()
    plt.clf()


def draw_jlt_comparison(k):
    ns = np.arange(1000, 50000, 200)
    #col1 = 2 + 2 * 2 / (epsilon**2/2 - epsilon**3/3) * np.log(ns)
    fig, ax = plt.subplots()
    col1 = ns / (2 * k / math.log(1. / 0.9)) ** 0.5
    ax.plot(ns, col1)
    epsilon = 0.75
    col2 = np.log(col1) * 4 / (epsilon**2/2 - epsilon**3/3)
    ax.plot(ns, col2)
    ax.legend(["No JLT", "JLT"])
    ax.set_xlabel("n")
    ax.set_ylabel("count")
    ax.set_title("Solved linear eqns. ")

    output_file(fig, "jlt-cols"+d_to_str(k))


def geometric_mean(l, default=0., exclude_zero=True):
    found_any = False
    length = 0
    p = 1.
    for v in l:
        if v or (v == 0 and exclude_zero == False):
            p *= v
            length += 1
    if length == 0 or length == 0.0:
        return default
    return p ** (1. / length)



def analyze_experiment(df, restrictions, instance_names, ks=None, additional_exclusion=None):
    resistances = {}
    times = {}
    if not ks:
        ks = set()

    restricted_frame = restrict_frame(df, restrictions)
    for instance_name in instance_names:
        instance_frame = project(restricted_frame, "Instance", instance_name)

        def insert(d, k1, v):
            if k1 not in d:
                d[k1] = {}
            d[k1][instance_name] = v

        for row in instance_frame.iterrows():
            row = row[1]

            k = row['k']

            if additional_exclusion:
                if additional_exclusion(row, restrictions, instance_name):
                    continue
            ks.add(k)

            insert(resistances, k, row['Gain'])
            insert(times, k, row['Time'])

    return resistances, times, ks

def analyze_multiple_experiments(df, experiment_restriction_list, instance_names, additional_exclusion=None):
    ks = set()

    result_resistances = []
    result_times = []
    
    for restrictions in experiment_restriction_list:
        res, times, ks = analyze_experiment(df, restrictions, instance_names, ks, additional_exclusion)
        result_resistances.append(res)
        result_times.append(times)

    ks = sorted(list(ks))
    return result_resistances, result_times, ks

def plot_result_vs_time(df, instance_names, experiment_restriction_list, experiment_names, filename=None, additional_exclusion=None, output_values_text=None):
    result_resistances, result_times, ks = analyze_multiple_experiments(df, experiment_restriction_list, instance_names, additional_exclusion)

    fig, ax = plt.subplots()

    fig._set_dpi(400)
    fig.set_size_inches(4, 4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Marginal Gain')

    for j, (algorithm_resistances, algorithm_times) in enumerate(zip(result_resistances, result_times)):
            
        experiment_name = experiment_names[j]
        gain_value = geometric_mean(gain for k, algorithm_k_resistances in algorithm_resistances.items() for inst, gain in algorithm_k_resistances.items())
        time_value = geometric_mean(time for k, algorithm_k_times in algorithm_times.items() for inst, time in algorithm_k_times.items())
        if output_values_text:
            print(f"{experiment_name} & {gain_value:.4f} & {time_value:.4f} \\\\")
        plt.plot([time_value], [gain_value], markers[j], label=experiment_name)
    
    #plt.gca().invert_yaxis()

    plt.legend()
    output_file(fig, filename)
    plt.close(fig)

    


def plot_averaged(df, instance_names, experiment_restriction_list, experiment_names, reference_restrictions=None, filename=None, output_values_text=None):    
    result_resistances, result_times, ks = analyze_multiple_experiments(df, experiment_restriction_list, instance_names)

    if reference_restrictions:
        reference_resistances, reference_times, _ = analyze_experiment(df, reference_restrictions, instance_names)


    x_pos = np.arange(len(ks))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle(instance_name)
    fig._set_dpi(400)
    fig.set_size_inches((6, 4))

    ax1.set_xlabel('k')
    ax2.set_xlabel('k')
    if reference_restrictions:
        ax1.set_ylabel('Relative Gain')
        ax2.set_ylabel('Relative Time')
    else:
        ax1.set_ylabel('Gain')
        ax2.set_ylabel('Time')

    if reference_restrictions:
        ax1.set_ylim(0., 1.)
        ax2.set_ylim(0., 1.)
    else:
        pass


    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ks)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ks)
    
    #ax2.legend(experiment_names)

    
    def offset(l, k):
        width = 0.8 / k
        return width * l - width / 2 * k + width / 2

    num_experiments = len(result_resistances)

    if output_values_text:
        print("&" + " & ".join(d_to_str(v) for v in ks))

    s_res = ""
    s_t = ""

    for j, (algorithm_resistances, algorithm_times) in enumerate(zip(result_resistances, result_times)):
        resistance_means = []
        time_means = []
        for k in ks:
            if (reference_restrictions and (k not in reference_resistances or k not in reference_times)) or not k in algorithm_resistances or not k in algorithm_times:
                resistance_means.append(0)
                time_means.append(0)
                continue
            algorithm_k_resistances = algorithm_resistances[k]
            algorithm_k_times = algorithm_times[k]
            relative_resistances = []
            relative_times = []
            for instance_name in instance_names:
                if reference_restrictions:
                    if instance_name in algorithm_k_resistances and instance_name in reference_resistances[k]:
                        relative_resistances.append(algorithm_k_resistances[instance_name] / reference_resistances[k][instance_name])
                    if instance_name in algorithm_k_times and instance_name in reference_times[k]:
                        relative_times.append(algorithm_k_times[instance_name] / reference_times[k][instance_name])
                else:
                    if instance_name in algorithm_k_resistances:
                        relative_resistances.append(algorithm_k_resistances[instance_name])
                    if instance_name in algorithm_k_times:
                        relative_times.append(algorithm_k_times[instance_name])

            res_mean = geometric_mean(relative_resistances)
            time_mean = geometric_mean(relative_times)
            resistance_means.append(res_mean)
            time_means.append(time_mean)
            if len(relative_resistances) > 1 and not output_values_text:
                print("Taking means of {} instances for k = {} and j = {}.".format(len(relative_resistances), k, j))
        
        experiment_name = experiment_names[j]
        ax1.bar(x_pos + offset(j, num_experiments), resistance_means, align='center', color=colors[j], alpha = 0.5, label=experiment_name, width=0.8 / num_experiments)
        ax2.bar(x_pos + offset(j, num_experiments), time_means, align='center', color=colors[j], alpha = 0.5, label=experiment_name, width=0.8 / num_experiments)

        if output_values_text:
            s_res += f"{experiment_name} & " + " & ".join(d_to_str(s) for s in resistance_means) + "\\\\\n"
            s_t += f"{experiment_name} & " + " & ".join(d_to_str(s) for s in time_means) + "\\\\\n"

    if output_values_text: 
        print(s_res)
        print(s_t)

    
    #ax1.legend()
    #ax2.legend()
    plt.legend(ncol = 1, bbox_to_anchor=(1, 1), loc='lower right')

    fig.tight_layout()
    if filename == None:
        filename = "results_aggregated"
    
    output_file(fig, filename)

    plt.close(fig)



#draw_jlt(jlt_df)
for k in [5, 20, 50, 200]:
    draw_jlt_comparison(k)


large_instances = ["deezer_europe", "opsahl-powergrid", "arxiv-grqc", "facebook_ego_combined", "arxiv-hephth", "arxiv-heph"]
huge_instances = ["loc-brightkite", "flickr", "livemocha"]


restr_submodular = {"Threads": 12, "Experiment": "submodular-greedy"}

restr_stoch = {
    "Experiment": "stochastic-greedy",
    "Threads": 12
}

restr_similarity = {
    "Experiment": "sq-greedy",
    "Heuristic": "Similarity",
    "Linalg": "LU",
    "Epsilon": 0.9,
    "Threads": 12
}

restr_random = {
    "Experiment": "sq-greedy",
    "Heuristic": "Random",
    "Linalg": "LU",
    "Threads": 12
}

restr_similarity_jlt = {
    "Experiment": "sq-greedy",
    "Heuristic": "Similarity",
    "Linalg": "JLT via Sparse LU",
    "Epsilon": 0.9,
    "Threads": 12
}


restr_random_jlt = {
    "Experiment": "sq-greedy",
    "Heuristic": "Random",
    "Linalg": "JLT via Sparse LU",
    "Epsilon": 0.9,
    "Threads": 12
}

restr_lpinv_diag = {
    "Experiment": "sq-greedy",
    "Heuristic": "Lpinv Diagonal",
    "Linalg": "LU",
    "Threads": 12,
    "Epsilon": 0.9,
    "Epsilon2": 10.
}


def exclude_large_k(row, restrictions, instance_name):
    return row['k'] > 20 #and 'Linalg' in restrictions and restrictions['Linalg'] == "JLT via Sparse LU"

plot_result_vs_time(df, large_instances, [ restr_submodular, restr_stoch, restr_lpinv_diag, restr_similarity, restr_random, restr_similarity_jlt], ["Submodular", "Stochastic-Submodular", "Main-Resistances-Approx", "Main-Similarity", "Main-Random", "Main-Similarity-JLT"], "gain_vs_time", None, True)
plot_result_vs_time(df, large_instances, [ restr_submodular, restr_stoch, restr_lpinv_diag, restr_similarity, restr_random, restr_similarity_jlt], ["Submodular", "Stochastic-Submodular", "Main-Resistances-Approx", "Main-Similarity", "Main-Random", "Main-Similarity-JLT"], "gain_vs_time_small_k", exclude_large_k, True)

plot_averaged(df, large_instances, [ restr_stoch, restr_lpinv_diag, restr_similarity, restr_random, restr_similarity_jlt], ["Stochastic-Submodular", "Main-Resistances-Approx", "Main-Similarity", "Main-Random", "Main-Similarity-JLT"], restr_submodular, "results_aggregated_5", True)

for i in large_instances:
    plot_averaged(df, [i], [restr_stoch, restr_lpinv_diag, restr_similarity, restr_random, restr_similarity_jlt], ["Stochastic-Submodular", "Main-Resistances-Approx", "Main-Similarity", "Main-Random", "Main-Similarity-JLT"], restr_submodular, "results_"+i)



# Generated Instances
ba_instances = ["barabasi_albert_5_10000_3_"+str(i) for i in range(20)]
er_instances = ["erdos_renyi_10000_0.01_"+str(i) for i in range(20)]
ws_instances = ["watts_strogatz_10000_20_0.2_"+str(i) for i in range(20)]
#for i in generated_instances:
#    plot_averaged(df, [i], [restr_stoch, restr_lpinv_diag, restr_similarity, restr_random, restr_similarity_jlt], ["Stochastic-Submodular", "Main-Resistances-Approx", "Main-Similarity", "Main-Random", "Main-Similarity-JLT"], restr_submodular, "results_"+i)
for instances, name in zip([ba_instances, er_instances, ws_instances], ["barabasi-albert", "erdos-renyi", "watts-strogatz"]):
    plot_averaged(df, instances, [restr_similarity, restr_random, restr_similarity_jlt], ["Main-Similarity", "Main-Random", "Main-Similarity-JLT"], None, "results_"+name, output_values_text=True)


# Huge Instances
restr_similarity_lamg = {
    "Experiment": "sq-greedy",
    "Heuristic": "Similarity",
    "Linalg": "LAMG",
    "Epsilon": 0.9,
    "Threads": 12
}


restr_similarity_lamg_eps99 = {
    "Experiment": "sq-greedy",
    "Heuristic": "Similarity",
    "Linalg": "LAMG",
    "Epsilon": 0.99,
    "Threads": 12
}

restr_similarity_jlt_lamg = {
    "Experiment": "sq-greedy",
    "Heuristic": "Similarity",
    "Linalg": "JLT via LAMG",
    "Epsilon": 0.9,
    "Threads": 12
}

for i in huge_instances:
    plot_averaged(df, [i], [restr_similarity_lamg, restr_similarity_lamg_eps99, restr_similarity_jlt_lamg], ["Main-Similarity-LAMG", "Main-Similarity-LAMG-eps0.99", "Main-Similarity-JLT-LAMG"], None, "results_"+i)




def plot_instance(df, instance_name, restrictions=[], filename=None, flags=None):
    restricted_frame = project(df, "Instance", instance_name)
    restricted_frame = restrict_frame(restricted_frame, restrictions)

    result_t = []
    result_gain = []
    result_name = []
    result_k = next(restricted_frame.iterrows())

    for index, row in restricted_frame.iterrows():
        experiment = row['Experiment']
        algorithm_name = experiment
        heuristic = row['Heuristic']
        if heuristic and pd.notnull(heuristic):
            algorithm_name += heuristic
        
        if flags and flags["full"] == False:
            if row['Linalg'] not in ['LU'] and not pd.isna(row['Linalg']):
                continue
            if algorithm_name == "greedy-3":
                continue
        
        if heuristic == "Lpinv Diagonal":
            if row['Linalg'] != 'LU':
                continue

        rename = {
            "greedy-3" : "main-resistances-exact",
            "sq-greedyLpinv Diagonal" : "main-resistances",
            "sq-greedyRandom": "main-random",
            "sq-greedySimilarity": "main-similarity",
        }
        if algorithm_name in rename:
            result_name.append(rename[algorithm_name])
        else:
            result_name.append(algorithm_name)

        if algorithm_name == "Random-Averaged":
            result_t.append(0)
        else:
            result_t.append(row['Time'])

        gain = row['Gain']
        if gain < 0:
            gain *= -1
        result_gain.append(gain)

        if not (flags and flags["full"] == False): 
            if result_name[-1] == "main-random":
                result_name[-1] = "main-random-" + row["Linalg"]
            if result_name[-1] == "main-similarity":
                result_name[-1] = "main-similarity-" + row["Linalg"]
            if result_name[-1] == "main-resistance":
                result_name[-1] = "main-resistance-" + row["Linalg"]


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
    fig.set_size_inches((8,4))
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
    ax1.bar(x_pos, result_gain, align='center', alpha = 0.5, color=colors)
    ax2.bar(x_pos, result_t, align='center', alpha = 0.5, color=colors)
    ax1.set_xticks(x_pos_shifted)
    ax1.set_xticklabels(result_name)
    ax2.set_xticks(x_pos_shifted)
    ax2.set_xticklabels(result_name)
    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)
    
    #ax1.legend()
    #ax2.legend()
    fig.tight_layout()
    if filename == None:
        filename = instance_name
    output_file(fig, filename)

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

#eval.plot_instance(eval.df, "arxiv-hephth", [["Threads", 12], ["k", 20]])
def quick_plot(name, threads, k):
    plot_instance(df, name, [["Threads", threads], ["k", k]], name + "-" + d_to_str(k))

#plot_instance(df, "arxiv-heph", {"Threads": 12, "k": 20})


#for k in [2, 5, 20, 50, 200]:
#    for i in large_instances:
#        plot_instance(df, i, {"k": k}, i+"-"+d_to_str(k), flags={"full":False})



#medium_instances = ["erdos_renyi_1000_0.02.nkb", "erdos_renyi_3000_0.01.nkb", "watts_strogatz_1000_7_0.3.nkb", "watts_strogatz_3000_7_0.3.nkb", "barabasi_albert_2_1000_2.nkb", "barabasi_albert_2_3000_2.nkb"]



