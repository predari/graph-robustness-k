#!/usr/bin/env python3

import simexpal
import pandas
import yaml

def parse(run, f):
    output = yaml.load(f, Loader=yaml.Loader)
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
        'time': exp['Time']
	}

    if 'Variant' in exp:
        d['variant'] = exp['Variant']

    return d
	

# Runs: 
# - InstanceName: '/home/matthias/simexpal-launch/instances/WattsStrogatz_1000_7_0.3.gml'
#   Nodes: 1000
#   Edges: 7000
#   k: 1
#   Algorithm:  'Simulated Annealing'
#   Variant:  'Random'
#   Value:  82074.8
#   Time:    0.352734


cfg = simexpal.config_for_dir()
df = pandas.DataFrame(cfg.collect_successful_results(parse))
#print(df.groupby('experiment').agg('mean'))

import numpy as np
import matplotlib.pyplot as plt

#x1 = np.linspace(0.0, 5.0)
#x2 = np.linspace(0.0, 2.0)

#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#y2 = np.cos(2 * np.pi * x2)

#fig, (ax1, ax2) = plt.subplots(2, 1)
#fig.suptitle('A tale of 2 subplots')

def plot_instance(df, instance_name):
    instanceframe = df[df['instance'] == instance_name]
    results = {}
    for index, row in instanceframe.iterrows():
        exp = row['experiment']
        if exp not in results:
            results[exp] = []
        results[exp].append((row['k'], row['value'], row['time']))


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(instance_name)
    fig.set_size_inches((8,8))
    fig._set_dpi(200)

    ax1.set_xlabel('k')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax1.set_ylabel('resistance difference to submodular')
    ax2.set_xlabel('k')
    ax2.set_ylabel('time (s)')
    

    results['submodular-greedy'].sort(key=lambda x: x[0])
    subm_values = [x[1] for x in results['submodular-greedy']]

    for expname, r in results.items():
        if expname.startswith("hillclimbing"):# or expname.startswith("random"):
            continue
        r.sort(key=lambda x: x[0])
        ks = np.array([x[0] for x in r])

        def f(a, b):
            return (a-b)/b
        values = np.array([f(r[i][1], subm_values[i]) for i in range(len(r))])
        times = np.array([x[2] for x in r])

        if (expname != "submodular-greedy"):
            ax1.plot(ks, values, 'o-', label=expname)
        if expname != "random-averaged":
            ax2.plot(ks, times, 'o-', label=expname)
        else:
            ax2.plot(np.array([]), np.array([]))

    #handles, labels = ax1.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    ax1.legend()
    ax2.legend()
    fig.savefig(instance_name + ".png")
    #plt.show()

plot_instance(df, "WattsStrogatz_1000_7_0.3")
plot_instance(df, "BarabasiAlbert_200_400_8")
plot_instance(df, "opsahl-usairport")
#plot_instance(df, "facebook_combined")
plot_instance(df, "dimacs10-netscience")
plot_instance(df, "ErdosRenyi_3000_0.01")
