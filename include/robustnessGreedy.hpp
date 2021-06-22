#ifndef ROBUSTNESS_GREEDY_H
#define ROBUSTNESS_GREEDY_H


#include <utility>
#include <random>
#include <exception>
#include <cmath>
#include <set>
#include <vector>
#include <map>

#include <laplacian.hpp>
#include <greedy.hpp>
#include <greedy_params.hpp>

#include <Eigen/Dense>
#include <networkit/graph/Graph.hpp>
#include <networkit/centrality/ApproxElectricalCloseness.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/components/ConnectedComponents.hpp>


namespace NetworKit {
inline bool operator<(const NetworKit::Edge& lhs, const NetworKit::Edge& rhs) {
    return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
}
};


using namespace NetworKit;



// The greedy algorithms in this file optimize for large -R_tot.


class RobustnessRandomAveraged : public virtual AbstractOptimizer<Edge> {
public:
    RobustnessRandomAveraged(GreedyParams params) : g(params.g), k(params.k) {}

    virtual void run() override {
        std::set<std::pair<unsigned int, unsigned int>> es;
        int n = g.numberOfNodes();
        if (n*(n-1)/2 - g.numberOfEdges() < k) {
            throw std::logic_error("Graph does not allow requested number of edges.");
        }
        auto lpinv = laplacianPseudoinverse(g);

        for (int repetitions = 0; repetitions < 10; repetitions++) {
            decltype(lpinv) lpinv_copy = lpinv;
            double r = static_cast<double>(repetitions);
            originalValue = r / (r+1.) * originalValue + 1. / (r+1.) * lpinv_copy.trace();

            result.clear();
            es.clear();
            for (int i = 0; i < k; i++) {
                do {
                    int u = std::rand() % n;
                    int v = std::rand() % n;
                    if (u > v) {
                        std::swap(u, v);
                    }

                    auto e = std::pair<unsigned int, unsigned int> (u, v);
                    if (u != v && !g.hasEdge(u,v) && !g.hasEdge(v,u) && es.count(std::pair<unsigned int, unsigned int>(u, v)) == 0) {
                        es.insert(e);
                        result.push_back(NetworKit::Edge(u, v));
                        updateLaplacianPseudoinverse(lpinv_copy, Edge(u, v));
                        break;
                    }
                } while (true);
            }
            resultValue = r / (r+1.) * resultValue + 1. / (r+1.) * lpinv_copy.trace();
        }

        originalValue *= n;
        resultValue *= n;
    }

    virtual double getResultValue() override {
        return resultValue;
    }

    virtual double getOriginalValue() override {
        return originalValue;
    }

    virtual std::vector<Edge> getResultItems() override {
        return result;
    }

    virtual bool isValidSolution() {
        return true;
    }

private:
    Graph &g;
    count k;
    std::vector<NetworKit::Edge> result;
    double originalValue = 1.;
    double resultValue = 1.;
};


class RobustnessSubmodularGreedy final : public SubmodularGreedy<Edge> {
public:
    RobustnessSubmodularGreedy(GreedyParams params) {
        this->g = params.g;
        this->n = g.numberOfNodes();
        this->k = params.k;

        // Compute pseudoinverse of laplacian
        this->lpinv = laplacianPseudoinverse(g);
        this->totalValue = this->lpinv.trace() * n * (-1.0);
        this->originalResistance = this->totalValue * (-1.);
    }

    virtual void addDefaultItems() {
        // Add edges to items of greedy
        std::vector<Edge> items;
        for (size_t i = 0; i < this->n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (i != j && !this->g.hasEdge(i, j)) {
                    items.push_back(Edge(i,j));
                }
            }
        }
        
        this->addItems(items);
    }

    virtual double getResultValue() override {
        return this->totalValue * (-1.0);
    }

    virtual double getOriginalValue() override {
        return this->originalResistance;
    }

    virtual std::vector<NetworKit::Edge> getResultItems() override {
        return this->results;
    }
    
private:

    virtual double objectiveDifference(Edge e) override {
        auto i = e.u;
        auto j = e.v;
        return this->n * (-1.0) * laplacianPseudoinverseTraceDifference(this->lpinv, i, j);
    }

    virtual void useItem(Edge e) override {
        updateLaplacianPseudoinverse(this->lpinv, e);
        //this->g.addEdge(e.u, e.v);
    }

    Eigen::MatrixXd lpinv;

    double originalResistance = 0.;

    Graph g;
    int n;
};


class RobustnessSqGreedy final : public AbstractOptimizer<NetworKit::Edge> {
public:
    RobustnessSqGreedy(GreedyParams params) {
        this->g = params.g;
        this->n = g.numberOfNodes();
        this->k = params.k;
        this->epsilon = params.epsilon;
        gen = std::mt19937(Aux::Random::getSeed());

        // Compute pseudoinverse of laplacian
        this->lpinv = laplacianPseudoinverse(g);
        this->totalValue = this->lpinv.trace() * n * (-1.0);
        this->originalResistance = -1. * totalValue;
    }

    virtual double getResultValue() override {
        return this->totalValue * (-1.0);
    }

    virtual double getOriginalValue() override {
        return this->originalResistance;
    }

    virtual std::vector<NetworKit::Edge> getResultItems() override {
        return this->results;
    }

    virtual bool isValidSolution() override {
        return this->validSolution;
    }

    virtual void run() override {
        this->round = 0;
        this->validSolution = false;
        this->results.clear();
        struct edge_cmp {
            bool operator() (const Edge& lhs, const Edge& rhs) const {
                return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
            }
        };
        std::set<Edge, edge_cmp> resultSet;

        if (k + g.numberOfEdges() > (n * (n-1) / 8*3)) {
            this->hasRun = true;
            std::cout << "Bad call to GreedySq, adding this many edges is not supported! Attempting to have " << k + g.numberOfEdges() << " edges, limit is " << n*(n-1) / 8 *3;
            return;
        }

        for (int r = 0; r < k; r++)
        {
            double bestGain = -std::numeric_limits<double>::infinity();
            Edge bestEdge;

            int it = 0;

            do {
                // Collect nodes set for current round
                std::set<NetworKit::node> nodes;
                unsigned int s = (unsigned int)std::sqrt(1.0 * (this->n * (this->n-1) /2 - this->g.numberOfEdges() - this->round) / k * std::log(1.0/epsilon)) + 2;
                //if (s > k-round) { s = k - round; }
                if (s < 10) { s = 10; }

                //if (s < 4) { s = 4; }
                if (s > n/2) { s = n/2; }
                double min = std::numeric_limits<double>::infinity();

                std::vector<double> nodeWeights;
                if (it++ < 2) { 
                    double tr = lpinv.trace();
                    for (int i = 0; i < n; i++) {
                        double val = n * lpinv(i, i) + tr;
                        if (val < min) { min = val; }
                        nodeWeights.push_back(val);
                    }
                    for (auto &v : nodeWeights) {
                        auto u = v - min;
                        v = u*u;
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        nodeWeights.push_back(1.0/n);
                    }
                }

                std::discrete_distribution<> distribution_nodes_heuristic (nodeWeights.begin(), nodeWeights.end());

                nodeWeights.clear();
                for (int i = 0; i < n; i++) {
                nodeWeights.push_back(1.0/n);
                }
                std::discrete_distribution<> distribution_nodes_uniform (nodeWeights.begin(), nodeWeights.end());

                while (nodes.size() < s / 2) {
                    nodes.insert(distribution_nodes_heuristic(gen));
                }
                while (nodes.size() < s) {
                    nodes.insert(distribution_nodes_uniform(gen));
                }
                std::vector<node> nodesVec {nodes.begin(), nodes.end()};
                

                // Determine best edge between nodes from node set
                auto edgeValid = [&](node u, node v){
                    if (this->g.hasEdge(u, v)) return false;
                    if (resultSet.count(Edge(u, v)) > 0 || resultSet.count(Edge(v, u)) > 0) { return false; }
                    return true;
                };
                for (int i = 0; i < s; i++) {
                    auto u = nodesVec[i];
                    for (int j = 0; j < i; j++) {
                        auto v = nodesVec[j];
                        if (edgeValid(u, v)) {
                            double gain = objectiveDifference(Edge(u, v));
                            if (gain > bestGain) {
                                bestEdge = Edge(u, v);
                                bestGain = gain;
                            }
                        }
                    }
                }
            } while (bestGain == -std::numeric_limits<double>::infinity());
            // Accept edge
            resultSet.insert(bestEdge);
            this->totalValue += bestGain;
            updateLaplacianPseudoinverse(this->lpinv, bestEdge);

            if (this->round == k-1)
            {
                this->validSolution = true;
                break;
            }
            this->round++;
        }
        this->hasRun = true;
        this->results = {resultSet.begin(), resultSet.end()};
    }

private:
    virtual double objectiveDifference(Edge e) {
        auto i = e.u;
        auto j = e.v;
        return this->n * (-1.0) * laplacianPseudoinverseTraceDifference(this->lpinv, i, j);
    }

    Graph g;
    int n;
    Eigen::MatrixXd lpinv;
    std::mt19937 gen;


	bool validSolution = false;
	int round=0;
	std::vector<Edge> results;
	double totalValue = 0.0;
    double originalResistance = 0.;
    double epsilon = 0.1;
    int k;
};




class RobustnessStochasticGreedy : public StochasticGreedy<Edge>{
public:
    RobustnessStochasticGreedy(GreedyParams params) {
        this->g = params.g;
        this->n = g.numberOfNodes();
        this->k = params.k;
        this->epsilon = params.epsilon;

        this->lpinv = laplacianPseudoinverse(g);
        this->totalValue = this->lpinv.trace() * n * (-1.0);
        this->originalResistance = -1. * this->totalValue;
    }

    virtual void addDefaultItems() override {
        // Add edges to items of greedy
        std::vector<Edge> items;
        for (size_t i = 0; i < this->n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (!this->g.hasEdge(i, j)) {
                    items.push_back(Edge(i,j));
                }
            }
        }
        
        this->addItems(items);
    }

    virtual std::vector<Edge> getResultItems() override {
        return this->results;
    }
    virtual double getResultValue() override {
        return this->totalValue * (-1.0);
    }

    virtual double getOriginalValue() override {
        return this->originalResistance;
    }


private:
    virtual double objectiveDifference(Edge e) override {
        return (-1.0) * laplacianPseudoinverseTraceDifference(lpinv, e.u, e.v) * n;
    }

    virtual void useItem(Edge e) override {
        updateLaplacianPseudoinverse(this->lpinv, e);
        //this->g.addEdge(e);
    }

    Eigen::MatrixXd lpinv;

    Graph g;
    int n;
    double originalResistance = 0.;
};






// Store laplacian as std::vector instead of Eigen::Matrix and update on demand. Appears to slightly improve performance over the normal submodular greedy; very slightly improved cache access properties
class RobustnessSubmodularGreedy2 final : public SubmodularGreedy<Edge>{
public:
    RobustnessSubmodularGreedy2(GreedyParams params) {
        this->g = params.g;
        this->n = g.numberOfNodes();
        this->k = params.k;
        this->age = std::vector<int>(n);

        // Compute pseudoinverse of laplacian
        auto lpinv = laplacianPseudoinverse(g);
        this->totalValue = lpinv.trace() * n * (-1.0);
        for (int i = 0; i < n; i++) {
            this->lpinvVec.push_back(lpinv.col(i));
        }
        this->originalResistance = -1. * this->totalValue;
    }

    virtual void addDefaultItems() {
        // Add edges to items of greedy
        std::vector<Edge> items;
        for (size_t i = 0; i < this->n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (i != j && !this->g.hasEdge(i, j)) {
                    items.push_back(Edge(i,j));
                }
            }
        }
        
        this->addItems(items);
    }

    virtual double getResultValue() override {
        return this->totalValue * (-1.0);
    }
    virtual double getOriginalValue() override {
        return this->originalResistance;
    }


    virtual std::vector<NetworKit::Edge> getResultItems() override{
        return this->results;
    }
    
private:

    virtual double objectiveDifference(Edge e) override {
        auto i = e.u;
        auto j = e.v;
        
        this->updateColumn(i);
        this->updateColumn(j);
        return (-1.0) * this->n * laplacianPseudoinverseTraceDifference(this->lpinvVec[i], i, this->lpinvVec[j], j);
    }

    virtual void useItem(Edge e) override {
        auto i = e.u;
        auto j = e.v;
        this->updateColumn(i);
        this->updateColumn(j);
        double R_ij = lpinvVec[i](i) + lpinvVec[j](j) - 2* lpinvVec[j](i);
        double w = 1.0 / (1.0 + R_ij);

        updateVec.push_back(this->lpinvVec[i] - this->lpinvVec[j]);
        updateW.push_back(w);
        //updateLaplacianPseudoinverse(this->lpinv, e);
        //this->g.addEdge(e.u, e.v);
    }

    void updateColumn(int i) {
        if (age[i] < this->round) {
            for (int r = age[i]; r < this->round; r++)
                lpinvVec[i] -= updateVec[r] * updateVec[r](i) * updateW[r];
            age[i] = this->round;
        }
    }

    Graph g;
    int n;
    double originalResistance;
    std::vector<Eigen::VectorXd> updateVec;
    std::vector<double> updateW;
    std::vector<int> age;
    std::vector<Eigen::VectorXd> lpinvVec;
};




#endif // ROBUSTNESS_GREEDY_H
