#ifndef ROBUSTNESS_SIMULATED_ANNEALING_HPP
#define ROBUSTNESS_SIMULATED_ANNEALING_HPP

#include <random>
#include <exception>
#include <cmath>

#include <simulatedAnnealing.hpp>
#include <utility>
#include <laplacian.hpp>

#include <Eigen/Dense>

#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>
#include <networkit/auxiliary/Random.hpp>


// TODO Some methods could be made static.

// Expectations of this algorithm include that the graph is connected and there is space to add (k+1) edges to the graph.


struct State {
    std::vector<NetworKit::Edge> edges;
    Eigen::MatrixXd lpinv;
    double energy;
};

struct StateTransition {
    std::vector<NetworKit::Edge> addedEdges;
    std::vector<unsigned int> removedEdges;
    double energy;
    bool energyCalculated = false;
};


// Interface for different variants.
class RobustnessSimulatedAnnealingBase : public virtual SimulatedAnnealing<State, StateTransition>{
public:
    virtual void init(NetworKit::Graph G, int k, double roundFactor) = 0;
    virtual void setInitialState(State const &s) = 0;
    virtual double getResultResistance() = 0;
    virtual std::vector<Edge> getResultEdges() = 0;
    //virtual void run() = 0;
};

template <int transitionMethod=0>
class RobustnessSimulatedAnnealing : public virtual SimulatedAnnealing<State, StateTransition>, public RobustnessSimulatedAnnealingBase {
public:
    void init (NetworKit::Graph G, int k, double roundFactor=1.0) {
        this->G = G;
        this->n = this->G.numberOfNodes();
        this->k = k;
        this->round = 0;
        this->maxRounds = k * roundFactor;
        //this->maxRounds = (int)(k * (std::log2(k)+1));

        gen = std::mt19937(Aux::Random::getSeed());
        dis_n = decltype(dis_n)(0, this->n-1);
        dis_k = decltype(dis_k)(0, this->k-1);

        // currentState.lpinv = laplacianPseudoinverse(G);
    }

    virtual void setInitialState(State const & s) {
        auto G_copy = G;
        for (auto e: s.edges) {
            G_copy.addEdge(e.u, e.v);
        }
        this->currentState.lpinv = laplacianPseudoinverse(G_copy);
        this->currentState.energy = this->currentState.lpinv.trace() * this->n;
        this->currentState.edges = s.edges;
        this->bestState = this->currentState;
    }

    virtual void setInitialTemperature() override {
        // We want the temperature to be at the order of magnitude of the gains we might get in a single iteration. Thus we determine gains by experiment.
        // We add and remove a few edges and measure the gain in total effective resistance. Then we use the difference between the maximum and the minimum as temperature.
        double max = (-1.0) * std::numeric_limits<double>::infinity();
        double min = std::numeric_limits<double>::infinity();
        for (int i = 0; i < std::log2(k+n) + 1; i++) {
            auto edgeIndex = dis_k(gen);
            auto e = this->currentState.edges[edgeIndex];
            double val = std::abs(laplacianPseudoinverseTraceDifference(this->currentState.lpinv, e.u, e.v, -1.0));
            if (val > max) { max = val; }
            if (val < min) { min = val; }

            auto u = dis_n(gen);
            auto v = dis_n(gen);
            val = std::abs(laplacianPseudoinverseTraceDifference(this->currentState.lpinv, u, v, 1.0));
            if (val > max) { max = val; }
            if (val < min) { min = val; }
        }
        if (max - min == std::numeric_limits<double>::infinity() || max - min == (-1.0) * std::numeric_limits<double>::infinity()) {
            this->temperature = 50 * std::abs(this->currentState.energy / (this->k + this->G.numberOfEdges()));
        } else {
            this->temperature = (max - min);
        }
    }



    virtual void summarize() {
        std::cout << "N: " << this->n << ". k: " << this->k << ". Round: " << this->round << ". Energy: " << this->currentState.energy << ". Temperature: " << this->temperature << std::endl;
        for (auto e: this->currentState.edges) {
            std::cout << "(" << e.u << ", " << e.v << "), ";
        }
        std::cout << std::endl;
    }

    virtual bool endIteration() override {
        return this->round > this->maxRounds;
    }

    std::vector<NetworKit::Edge> getEdges() {
        return this->currentState.edges;
    }
    double getResultResistance() {
        return this->getEnergy(this->currentState);
    }

    std::vector<Edge> getResultEdges() {
        return this->currentState.edges;
    }


protected:
    int maxRounds;
    int n;
    int k;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis_n;
    std::uniform_int_distribution<> dis_k;
    NetworKit::Graph G;
    Eigen::MatrixXd lpinv;

    virtual void initRound() override {
        if (this->verbose) {
            auto g_copy = G;
            for (auto e : this->currentState.edges) {
                g_copy.addEdge(e.u, e.v);
            }
            double energy = laplacianPseudoinverse(g_copy).trace() * n;

            std::cout << "Energy from Graph: " << energy << ". ";
            std::cout << "Energy from Pseudoinverse: " << this->currentState.lpinv.trace() * n << "\n";
            assert(std::abs(energy - this->getEnergy(this->currentState)) < 0.001);
        }
    }


    // Add and remove a random edge.
    virtual StateTransition randomTransition(State const &state) override {
        StateTransition update;

        int edgeIndex = -1;
        Edge addedEdge;

        auto canAddEdge = [&](int a, int b) {
            if ((a == b) || this->G.hasEdge(a, b)) {
                return false;
            }

            for (auto e: state.edges) {
                if (e == Edge(a, b) || e == Edge(b, a)) {
                    return false;
                }
            }
            return true;
        };

        auto randomEdgeByResistanceWeights = [&]() {
            std::vector<double> edgeWeights;
            double tr = state.lpinv.trace();

            double max = (-1.0) * std::numeric_limits<double>::infinity();
            for (auto e : state.edges) {
                auto i = e.u;
                auto j = e.v;
                double val = state.lpinv(i, i) + state.lpinv(j, j) - 2 * state.lpinv(i, j);
                //std::cout << val << ". " << state.lpinv(i, i) + state.lpinv(j, j) - 2 * state.lpinv(i, j) << "\n";
                //double val = 1/(n*state.lpinv(i, i) + n*state.lpinv(j, j) - *2*tr*/);
                edgeWeights.push_back(val);
                if (max < val) max = val;
            }
            for (auto &v : edgeWeights) {
                auto u = max - v;
                v = u*u*u;
            }
            std::discrete_distribution<> d_edges(edgeWeights.begin(), edgeWeights.end());
            edgeIndex = d_edges(gen);
        };

        auto nodeResistanceDistribution = [&]() {
            double tr = state.lpinv.trace();

            std::vector<double> nodeWeights;
            double min = std::numeric_limits<double>::infinity();
            for (int i = 0; i < n; i++) {
                double val = n * state.lpinv(i, i) + tr;
                if (val < min) { min = val; }
                nodeWeights.push_back(val);
            }
            for (auto &v : nodeWeights) {
                auto u = v - min;
                v = u*u*u;
            }

            std::discrete_distribution<> d_nodes(nodeWeights.begin(), nodeWeights.end());
            return d_nodes;
        };

        if (transitionMethod == 0) 
        {
            // Random edge replaced with random edge
            while (true) {
                int a = dis_n(gen);
                int b = dis_n(gen);
                if (canAddEdge(a, b)) {
                    addedEdge = Edge(a, b);
                    break;
                }
            }
            edgeIndex = dis_k(gen);
        } 
        else if (transitionMethod == 1) 
        {
            // Pick removed edge randomly weighted according to effective resistance of edge. Pick added edge by picking both nodes independently randomly weighted according to resistance centrality.

            // Pick edge to remove in O(|E|). Lower resistance is better
            randomEdgeByResistanceWeights();

            // Pick new edge in O(|V|). Higher resistance is better
            auto d_nodes = nodeResistanceDistribution();
            while (true) {
                int a = d_nodes(gen);
                int b = d_nodes(gen);
                if (canAddEdge(a, b)) {
                    addedEdge = Edge(a, b);
                    break;
                }

            }
        } else if (transitionMethod == 2) {
            // Pick removed edge randomly. Pick added edge as follows: first pick a set of medium sized node set randomly, then pick the best edge between those nodes.
            std::set<int> nodesSet;
            int numNodes = (std::log2(k) + 1);

            auto d_nodes = nodeResistanceDistribution();

            
            while(nodesSet.size() < numNodes) {
                nodesSet.insert(d_nodes(gen));
            }
            std::vector<int> nodes (nodesSet.begin(), nodesSet.end());

            NetworKit::Edge bestEdge;
            double bestGain = 0.0;
            bool haveEdge = false;
            for (int i = 0; i < numNodes; i++) {
                auto u = nodes[i];
                for (int j = 0; j < i; j++) {
                    auto v = nodes[j];
                    if (canAddEdge(u, v)) {
                        haveEdge = true;
                        double gain = (-1.0) * laplacianPseudoinverseTraceDifference(state.lpinv, u, v);
                        if (!haveEdge || gain > bestGain) {
                            bestGain = gain;
                            bestEdge = Edge(u, v);
                        }
                    }
                }
            }
            /*
            NetworKit::Edge bestEdge;
            double bestGain = 0.0;
            bool haveEdge = false;

           for (int i = 0; i < numNodes; i++) {
               auto u = d_nodes(gen);
               auto v = d_nodes(gen);
               if (canAddEdge(u, v)) {
                   haveEdge = true;
                    double gain = (-1.0) * laplacianPseudoinverseTraceDifference(state.lpinv, u, v);
                    if (!haveEdge || gain > bestGain) {
                        bestGain = gain;
                        bestEdge = Edge(u, v);
                    }
               }
           }
           */

            if (!haveEdge) {
                while (true) {
                    int a = dis_n(gen);
                    int b = dis_n(gen);
                    if (canAddEdge(a, b)) {
                        addedEdge = Edge(a, b);
                        break;
                    }
                }
            } else {
                addedEdge = bestEdge;
            }

            randomEdgeByResistanceWeights();
        } else {
            throw std::logic_error("not implemented!");
        }

        update.removedEdges.push_back(edgeIndex);
        update.addedEdges.push_back(addedEdge);

        return update;
    }

    virtual double getEnergy(State & s) override { return s.energy; };


    virtual double getUpdatedEnergy(State const & state, StateTransition & update) override {     
        if (update.energyCalculated) {
            return update.energy;
        }   
        if (update.addedEdges.size() == 1 && update.removedEdges.size() == 1) {
            update.energy = state.energy + this->n * laplacianPseudoinverseTraceDifference2(state.lpinv, update.addedEdges[0], state.edges[update.removedEdges[0]], 1.0, -1.0);
            update.energyCalculated = true;
            //std::cout << "Resistance: " << update.energy << std::endl;
        } else {
            throw std::logic_error("Not implemented!");
        }
        return update.energy;
    }

    virtual void updateLpinv(State &state, StateTransition& update) {
        // If the update is too large, compute from scratch.
        // TODO the sqrt(n) is a a placeholder for something that actually makes sense.
        if (update.removedEdges.size() + update.addedEdges.size() > std::sqrt(this->n)) {
            auto G_copy = G;
            auto edges = state.edges;
            auto removedEdges = update.removedEdges;
            std::sort(removedEdges.begin(), removedEdges.end(), std::greater<int>());

            for (auto i : removedEdges) {
                edges.erase(edges.begin() + i);
            }
            for (auto e : update.addedEdges) {
                edges.push_back(e);
            }
            for (auto e : edges) {
                G_copy.addEdge(e.u, e.v);
            }
            state.lpinv = laplacianPseudoinverse(G_copy);
        } else { // Otherwise perform iterative update.
            for (auto l: update.removedEdges) {
                updateLaplacianPseudoinverse(state.lpinv, state.edges[l], -1.0);
            }
            for (auto e: update.addedEdges) {
                updateLaplacianPseudoinverse(state.lpinv, e);
            }
        }
    }

    virtual void transition(State & state, StateTransition & update) override {
        state.energy = this->getUpdatedEnergy(state, update);
        this->updateLpinv(state, update); 

        // Remove in descending order
        std::sort(update.removedEdges.begin(), update.removedEdges.end(), std::greater<int>());
        for (auto e: update.removedEdges) {
            state.edges.erase(state.edges.begin() + e);
        }

        for (auto e: update.addedEdges) {
            state.edges.push_back(e);
        }
    }

    virtual void updateTemperature(double previousEnergy, double candidateEnergy, bool accepted) override {
        double r = std::pow(0.001, 1.0 / maxRounds);
        this->temperature *= r;
    }

    virtual double acceptanceProbability(double currentEnergy, double candidateEnergy, double temperature) override {
        if (candidateEnergy < currentEnergy) { return 1.0; }
        return 0.0;
        return std::exp((currentEnergy - candidateEnergy) / temperature);
    }


};
#endif // ROBUSTNESS_SIMULATED_ANNEALING_HPP