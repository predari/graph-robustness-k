

#ifndef GREEDY_PARAMS_HPP
#define GREEDY_PARAMS_HPP

#include <networkit/graph/Graph.hpp>

enum class HeuristicType {
    random,
    lpinvDiag,
    similarity
};


class GreedyParams {
public:
    GreedyParams(NetworKit::Graph &g, NetworKit::count k) : g(g), k(k) {
        n = g.numberOfNodes();
    }

    NetworKit::count k;
	NetworKit::count n;
    double solverEpsilon = 0.00001;
	double epsilon = 0.1;
	double epsilon2 = 0.1;
	NetworKit::Graph& g;
    unsigned int threads = 1;
    HeuristicType heuristic;
    NetworKit::count similarityIterations = 100;
    double similarityPhi = 0.25;
};

#endif