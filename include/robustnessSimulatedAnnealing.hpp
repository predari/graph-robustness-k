#ifndef ROBUSTNESS_SIMULATED_ANNEALING_HPP
#define ROBUSTNESS_SIMULATED_ANNEALING_HPP

#include <random>

#include <simulatedAnnealing.hpp>
#include <utility>
#include <laplacian.hpp>

#include <Eigen/Dense>

#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>


// TODO make random stuff deterministic
// TODO fix edge adding logic. Need to ensure that the number of edges that we add stays constant, probably better to preserve the original graph.
// TODO Some methods could be made static.

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
    bool lpinv_updated = false;
    Eigen::MatrixXd lpinv = Eigen::MatrixXd(0,0);
};

class RobustnessSimulatedAnnealing : public SimulatedAnnealing<State, StateTransition> {
public:
    void init (NetworKit::Graph G, int k) {
        this->G = G;
        this->n = this->G.numberOfNodes();
        this->k = k;
        this->round = 0;

        gen = std::mt19937(rd());
        dis = decltype(dis)(0, this->n-1);
        dis2 = decltype(dis2)(0, this->k-1);

        currentState.lpinv = laplacianPseudoinverse(G);
    }

    virtual void setInitialState(State const & s) {
        currentState.energy = 0.0;
        currentState.edges.clear();
        StateTransition update;
        update.removedEdges.clear();
        update.addedEdges = s.edges;
        this->transition(this->currentState, update);
        this->bestState = this->currentState;

    }

    virtual void setInitialTemperature() override {
        this->temperature = std::abs(this->currentState.energy / (this->k+this->G.numberOfEdges()));
    }



    virtual void summarize() {
        std::cout << "N: " << this->n << ". k: " << this->k << ". Round: " << this->round << ". Energy: " << this->currentState.energy << ". Temperature: " << this->temperature << std::endl;
        for (auto e: this->currentState.edges) {
            std::cout << "(" << e.u << ", " << e.v << "), ";
        }
        std::cout << std::endl;
    }

    virtual bool endIteration() override {
        return this->round > 400;
    }

    std::vector<NetworKit::Edge> getEdges() {
        return this->currentState.edges;
    }
    double getTotalEnergy() {
        return this->getEnergy(this->currentState);
    }


protected:
    int n;
    int k;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    std::uniform_int_distribution<> dis2;
    NetworKit::Graph G;
    Eigen::MatrixXd lpinv;


    // Add and remove a random edge.
    virtual StateTransition randomTransition(State const &state) override {
        StateTransition update;

        int edge_index = -1;
        Edge added;

        // Pick a random edge and move one of its nodes to a neighbor such that we obtain a valid edge.
        // Can produce endless loops so if we don't find anything after a bit, we back away.
        #ifdef ROBUSTNESS_SIMULATED_ANNEALING_PREFER_NEIGHBORS
        for (int i = 0; i < 50; i++) {
            int edge_index = dis2(gen);
            auto edge = state.edges[edge_index];
            bool first = (bool)dis2(gen) % 2;
            node n;
            if (first) {
                n = edge.u;
            } else {
                n = edge.v;
            }
            node replacement = NetworKit::GraphTools::randomNeighbor(this->G, n);
            if (first) {
                added = Edge(replacement, edge.v);
            } else {
                added = Edge(edge.u, replacement);
            }

            bool edgeIsInState = false;
            for (int i = 0; i < this->k; i++) {
                if (state.edges[i] == added|| state.edges[i] == Edge(added.v, added.u)) {
                    edgeIsInState = true;
                }
            }

            if (added.u != added.v && !this->G.hasEdge(added.u, added.v) && !edgeIsInState) {
                break;
            }
            edge_index = -1;
        }


        // If we didn't find anything, replace a random edge with another random edge
        if (edge_index == -1) {
        #endif
            while (true) {
                int a = dis(gen);
                int b = dis(gen);
                bool edgeIsInState = false;
                for (int i = 0; i < this->k; i++) {
                    if (state.edges[i] == Edge(a, b) || state.edges[i] == Edge(b, a)) {
                        edgeIsInState = true;
                    }
                }
                if (a != b && !this->G.hasEdge(a, b) && !edgeIsInState) {
                    added = Edge(a, b);
                    break;
                }
            }
            edge_index = dis2(gen);
        #ifdef ROBUSTNESS_SIMULATED_ANNEALING_PREFER_NEIGHBORS
        }
        #endif

        update.removedEdges.push_back(edge_index);
        update.addedEdges.push_back(added);

        return update;
    }

    virtual double getEnergy(State & s) override { return s.energy; };

    virtual void computeLpinv(State const &s, StateTransition & update) {
        if (!update.lpinv_updated) {
            update.lpinv = s.lpinv;

            for (auto l: update.removedEdges) {
                updateLaplacianPseudoinverse(update.lpinv, s.edges[l], -1.0);
            }
            for (auto e: update.addedEdges) {
                updateLaplacianPseudoinverse(update.lpinv, e);
            }
            update.lpinv_updated = true;
        }
    }

    virtual double getUpdatedEnergy(State const & state, StateTransition & update) override{        
        if (!update.energyCalculated)
        {
            if (update.addedEdges.size() == 1 && update.removedEdges.size() == 1) {
                update.energy = state.energy + laplacianPseudoinverseTraceDifference(state.lpinv, std::vector<NetworKit::Edge>{update.addedEdges[0], state.edges[update.removedEdges[0]]}, {1.0, -1.0}) * this->n;
                update.energyCalculated = true;
                //std::cout << "Resistance: " << update.energy << std::endl;
            } else {
                this->computeLpinv(state, update);

                update.energy = update.lpinv.trace() * this->n;

                update.energyCalculated = true;
            }
        }
        return update.energy;
    }

    virtual void transition(State & state, StateTransition & update) override {
        state.energy = this->getUpdatedEnergy(state, update);

        if (!update.lpinv_updated) {
            this->computeLpinv(state, update);
        }
        state.lpinv = update.lpinv;

        // Remove in descending order
        std::sort(update.removedEdges.begin(), update.removedEdges.end(), std::greater<int>());
        for (auto e: update.removedEdges) {
            state.edges.erase(state.edges.begin() + e);
        }

        for (auto e: update.addedEdges) {
            state.edges.push_back(e);
        }
    }
};
#endif // ROBUSTNESS_SIMULATED_ANNEALING_HPP