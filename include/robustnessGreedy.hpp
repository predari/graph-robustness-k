#ifndef ROBUSTNESS_GREEDY_H
#define ROBUSTNESS_GREEDY_H


#include <utility>
#include <random>
#include <exception>
#include <cmath>
#include <set>
#include <vector>

#include <laplacian.hpp>
#include <greedy.hpp>

#include <Eigen/Dense>
#include <networkit/graph/Graph.hpp>


namespace NetworKit {
inline bool operator<(const NetworKit::Edge& lhs, const NetworKit::Edge& rhs) {
    return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
}
};


using namespace NetworKit;



// The greedy algorithms in this file optimize for large -R_tot.

class RobustnessGreedy final : public SubmodularGreedy<Edge>{
public:
    void init(Graph G, int k) {
        this->G = G;
        this->n = G.numberOfNodes();
        this->k = k;

        // Compute pseudoinverse of laplacian
        this->lpinv = laplacianPseudoinverse(G);
        this->totalValue = this->lpinv.trace() * n * (-1.0);
    }

    virtual void addAllEdges() {
        // Add edges to items of greedy
        std::vector<Edge> items;
        for (size_t i = 0; i < this->n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (i != j && !this->G.hasEdge(i, j)) {
                    items.push_back(Edge(i,j));
                }
            }
        }
        
        this->addItems(items);
    }

    double getResultResistance() {
        return this->totalValue * (-1.0);
    }

    std::vector<NetworKit::Edge> getResultEdges() {
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
        //this->G.addEdge(e.u, e.v);
    }

    Eigen::MatrixXd lpinv;

    Graph G;
    int n;
};


class RobustnessSqGreedy final : public Algorithm {
public:
    void init(Graph G, int k, double epsilon = 0.1) {
        this->G = G;
        this->n = G.numberOfNodes();
        this->k = k;
        this->epsilon = epsilon;
        gen = std::mt19937(Aux::Random::getSeed());

        // Compute pseudoinverse of laplacian
        this->lpinv = laplacianPseudoinverse(G);
        this->totalValue = this->lpinv.trace() * n * (-1.0);
    }

    double getResultResistance() {
        return this->totalValue * (-1.0);
    }

    std::vector<NetworKit::Edge> getResultEdges() {
        return this->results;
    }

    bool isValidSolution() {
        return this->validSolution;
    }

    void run() {
        this->round = 0;
        this->validSolution = false;
        this->results.clear();
        struct edge_cmp {
            bool operator() (const Edge& lhs, const Edge& rhs) const {
                return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
            }
        };
        std::set<Edge, edge_cmp> resultSet;

        if (k + G.numberOfEdges() > (n * (n-1) / 8*3)) {
            this->hasRun = true;
            std::cout << "Bad call to GreedySq, adding this many edges is not supported! Attempting to have " << k + G.numberOfEdges() << " edges, limit is " << n*(n-1) / 8 *3;
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
                //int s = (int)std::log2(n)*2;
                //int s = //std::sqrt((n*(n-1) / 2) / k);
                unsigned int s = (unsigned int)std::sqrt(1.0 * (this->n * (this->n-1) /2 - this->G.numberOfEdges() - this->round) / k * std::log(1.0/epsilon)) + 2;
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

                std::discrete_distribution<> d_nodes_resistance (nodeWeights.begin(), nodeWeights.end());

                nodeWeights.clear();
                for (int i = 0; i < n; i++) {
                nodeWeights.push_back(1.0/n);
                }
                std::discrete_distribution<> d_nodes (nodeWeights.begin(), nodeWeights.end());

                while (nodes.size() < s / 2) {
                    nodes.insert(d_nodes_resistance(gen));
                }
                while (nodes.size() < s) {
                    nodes.insert(d_nodes(gen));
                }
                std::vector<node> nodesVec {nodes.begin(), nodes.end()};
                

                // Determine best edge between nodes from node set
                auto edgeValid = [&](node u, node v){
                    if (this->G.hasEdge(u, v)) return false;
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

    Graph G;
    int n;
    Eigen::MatrixXd lpinv;
    std::mt19937 gen;


	bool validSolution = false;
	int round=0;
	std::vector<Edge> results;
	double totalValue = 0.0;
    double epsilon = 0.1;
    int k;
};


class RobustnessStochasticGreedy : public StochasticGreedy<Edge>{
public:
    void init(Graph G, int k, double epsilon=0.1) {
        this->G = G;
        this->n = G.numberOfNodes();
        this->k = k;
        this->epsilon = epsilon;

        this->lpinv = laplacianPseudoinverse(G);
        this->totalValue = this->lpinv.trace() * n * (-1.0);
    }

    void addAllEdges() {
        // Add edges to items of greedy
        std::vector<Edge> items;
        for (size_t i = 0; i < this->n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (!this->G.hasEdge(i, j)) {
                    items.push_back(Edge(i,j));
                }
            }
        }
        
        this->addItems(items);
    }

    std::vector<Edge> getResultEdges() {
        return this->results;
    }
    double getResultResistance() {
        return this->totalValue * (-1.0);
    }
    

private:
    virtual double objectiveDifference(Edge e) override {
        return (-1.0) * laplacianPseudoinverseTraceDifference(lpinv, e.u, e.v) * n;
    }

    virtual void useItem(Edge e) override {
        updateLaplacianPseudoinverse(this->lpinv, e);
        //this->G.addEdge(e);
    }

    Eigen::MatrixXd lpinv;

    Graph G;
    int n;
};






// Store laplacian as std::vector instead of Eigen::Matrix and update on demand. Appears to slightly improve performance over the normal submodular greedy; very slightly improved cache access properties
class RobustnessGreedy2 final : public SubmodularGreedy<Edge>{
public:
    void init(Graph G, int k) {
        this->G = G;
        this->n = G.numberOfNodes();
        this->k = k;
        this->age = std::vector<int>(n);
        //for (int i = 0; i < n; i++) {
        //    age[i] = 0;
        //}

        // Compute pseudoinverse of laplacian
        auto lpinv = laplacianPseudoinverse(G);
        this->totalValue = lpinv.trace() * n * (-1.0);
        for (int i = 0; i < n; i++) {
            this->lpinv_vec.push_back(lpinv.col(i));
        }
    }

    virtual void addAllEdges() {
        // Add edges to items of greedy
        std::vector<Edge> items;
        for (size_t i = 0; i < this->n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (i != j && !this->G.hasEdge(i, j)) {
                    items.push_back(Edge(i,j));
                }
            }
        }
        
        this->addItems(items);
    }

    double getResultResistance() {
        return this->totalValue * (-1.0);
    }

    std::vector<NetworKit::Edge> getResultEdges() {
        return this->results;
    }
    
private:

    virtual double objectiveDifference(Edge e) override {
        auto i = e.u;
        auto j = e.v;
        
        this->updateColumn(i);
        this->updateColumn(j);
        return (-1.0) * this->n * laplacianPseudoinverseTraceDifference(this->lpinv_vec[i], i, this->lpinv_vec[j], j);
    }

    virtual void useItem(Edge e) override {
        auto i = e.u;
        auto j = e.v;
        this->updateColumn(i);
        this->updateColumn(j);
        double R_ij = lpinv_vec[i](i) + lpinv_vec[j](j) - 2* lpinv_vec[j](i);
        double w = 1.0 / (1.0 + R_ij);

        updateVec.push_back(this->lpinv_vec[i] - this->lpinv_vec[j]);
        updateW.push_back(w);
        //updateLaplacianPseudoinverse(this->lpinv, e);
        //this->G.addEdge(e.u, e.v);
    }

    void updateColumn(int i) {
        if (age[i] < this->round) {
            for (int r = age[i]; r < this->round; r++)
                lpinv_vec[i] -= updateVec[r] * updateVec[r](i) * updateW[r];
            age[i] = this->round;
        }
    }

    Graph G;
    int n;
    std::vector<Eigen::VectorXd> updateVec;
    std::vector<double> updateW;
    std::vector<int> age;
    std::vector<Eigen::VectorXd> lpinv_vec;
};




#endif // ROBUSTNESS_GREEDY_H
