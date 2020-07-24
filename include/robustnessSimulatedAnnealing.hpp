#ifndef ROBUSTNESS_SIMULATED_ANNEALING_HPP
#define ROBUSTNESS_SIMULATED_ANNEALING_HPP

#include <random>

#include <simulatedAnnealing.hpp>
#include <utility>
#include <laplacian.hpp>

#include <Eigen/Dense>

#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>
#include <networkit/auxiliary/Random.hpp>


// TODO Some methods could be made static.
// TODO The computation of the lpinv is a bit more messy than it should be, because our StateTransition is more general than it needs to be.

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

        gen = std::mt19937(Aux::Random::getSeed());
        dis = decltype(dis)(0, this->n-1);
        dis2 = decltype(dis2)(0, this->k-1);

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
        return this->round > this->k;
    }

    std::vector<NetworKit::Edge> getEdges() {
        return this->currentState.edges;
    }
    double getTotalEnergy() {
        return this->getEnergy(this->currentState);
    }

    std::vector<Edge> getResultEdges() {
        return this->currentState.edges;
    }


protected:
    int n;
    int k;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    std::uniform_int_distribution<> dis2;
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

        int edge_index = -1;
        Edge added;

        if (edge_index == -1) {
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
        }

        update.removedEdges.push_back(edge_index);
        update.addedEdges.push_back(added);

        return update;
    }

    virtual double getEnergy(State & s) override { return s.energy; };

    virtual void computeLpinv(State const &s, StateTransition & update) {
        if (!update.lpinv_updated) {
            update.lpinv = s.lpinv;

            if (update.removedEdges.size() + update.addedEdges.size() > this->n / 2) {
                auto G_copy = G;
                auto edges = s.edges;
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
                update.lpinv = laplacianPseudoinverse(G_copy);
            } else {
                for (auto l: update.removedEdges) {
                    updateLaplacianPseudoinverse(update.lpinv, s.edges[l], -1.0);
                }
                for (auto e: update.addedEdges) {
                    updateLaplacianPseudoinverse(update.lpinv, e);
                }
            }
            update.lpinv_updated = true;
        }
    }

    virtual double getUpdatedEnergy(State const & state, StateTransition & update) override {        
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

    virtual void updateLpinv(State &state, StateTransition& update) {
        // If we've already computed the lpinv while computing the energy, then use it.
        if (update.lpinv_updated) {
            state.lpinv = update.lpinv;
        }
        // If the update is too large, compute from scratch.
        else if (update.removedEdges.size() + update.addedEdges.size() > this->n / 2) {
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

        /*if (!update.lpinv_updated) {
            this->computeLpinv(state, update);
        }
        state.lpinv = update.lpinv;*/

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
        double r = std::pow(0.01, 1.0 / k);
        this->temperature *= r;
    }

};
#endif // ROBUSTNESS_SIMULATED_ANNEALING_HPP