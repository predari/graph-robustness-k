#ifndef ROBUSTNESS_SIMULATED_ANNEALING_HPP
#define ROBUSTNESS_SIMULATED_ANNEALING_HPP

#include <random>

#include <simulatedAnnealing.hpp>
#include <utility>
#include <laplacian.hpp>

#include <networkit/algebraic/Vector.hpp>
#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>


// TODO make random stuff deterministic
// TODO fix edge adding logic. Need to ensure that the number of edges that we add stays constant, probably better to preserve the original graph.

struct State {
    std::vector<NetworKit::Edge> edges;
    std::vector<Vector> lpinv;
    double energy;
};

struct StateTransition {
    std::vector<NetworKit::Edge> addedEdges;
    std::vector<unsigned int> removedEdges;
    double energy;
    bool energyCalculated = false;
    std::vector<Vector> lpinv;
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

        auto laplacian = CSRMatrix::laplacianMatrix(G);
        solver.setupConnected(laplacian);

        for (int i = 0; i < this->n; i++) {
            Vector ePivot(n, 0);
            ePivot[i] = 1;
            ePivot -= 1.0/n;
            
            Vector lpinvCol (n, 0);
            solver.solve(ePivot, lpinvCol);
            currentState.lpinv.push_back(lpinvCol);
        }
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


protected:
    int n;
    int k;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    std::uniform_int_distribution<> dis2;
    NetworKit::Graph G;
    std::vector<Vector> lpinv;
    NetworKit::Lamg<CSRMatrix> solver;



    // Add and remove a random edge.
    virtual StateTransition randomTransition(State const &state) override {
        StateTransition update;

        int edge_index = -1;
        Edge added;

        // Pick a random edge and move one of its nodes to a neighbor such that we obtain a valid edge.
        // Can produce endless loops so if we don't find anything after a bit, we back away.
        /*
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
        */

        // If we didn't find anything, replace a random edge with another random edge
        //if (edge_index == -1) {
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
        //}

        update.removedEdges.push_back(edge_index);
        update.addedEdges.push_back(added);

        return update;
    }

    virtual double getEnergy(State & s) override { return s.energy; };

    virtual double getUpdatedEnergy(State const & s, StateTransition & update) override{
        //update.lpinv = std::vector<Vector>(currentState.lpinv);
        
        if (!update.energyCalculated)
        {
            update.lpinv.clear();
            update.lpinv.resize(this->n);
            for (int i = 0; i < this->n; i++) {
                update.lpinv[i] = Vector(n);
                for (int j = 0; j < this->n; j++) {
                    update.lpinv[i][j] = s.lpinv[i][j];
                }
            }

            for (auto l: update.removedEdges) {
                updateLaplacianPseudoinverse(update.lpinv, s.edges[l], -1.0);
            }
            for (auto e: update.addedEdges) {
                updateLaplacianPseudoinverse(update.lpinv, e);
            }
            double tr;
            for (int i = 0; i < this->n; i++) {
                tr += update.lpinv[i][i];
            }

            //std::cout << "Candidate total resistance: " << tr * this->n << std::endl;
            update.energy = 1.0 * tr * this->n;

            update.energyCalculated = true;
        }
        return update.energy;
    }

    virtual void transition(State & state, StateTransition & update) override {
        state.energy = this->getUpdatedEnergy(state, update);
        //this->lpinv = update.lpinv;
        state.lpinv.clear();
        state.lpinv.resize(this->n);
        for (int i = 0; i < this->n; i++) {
            state.lpinv[i] = Vector(n);
            for (int j = 0; j < this->n; j++) {
                state.lpinv[i][j] = update.lpinv[i][j];
            }
        }

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