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

#include <Eigen/Dense>
#include <networkit/graph/Graph.hpp>
#include <networkit/centrality/ApproxElectricalCloseness.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>


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
        this->originalResistance = this->totalValue * (-1.);
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

    double getOriginalResistance() {
        return this->originalResistance;
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

    double originalResistance = 0.;

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
        this->originalResistance = -1. * totalValue;
    }

    double getResultResistance() {
        return this->totalValue * (-1.0);
    }

    double getOriginalResistance() {
        return this->originalResistance;
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
    double originalResistance = 0.;
    double epsilon = 0.1;
    int k;
};





class RobustnessTreeGreedy final : public Algorithm {
public:
    RobustnessTreeGreedy(Graph &G, int k, double epsilon=0.1, double epsilon_tree_count = 0.1) : G(G), apx(G, epsilon_tree_count) {
        this->n = G.numberOfNodes();
        this->k = k;
        this->epsilon = epsilon;
    }

    void init() {
        gen = std::mt19937(Aux::Random::getSeed());

        // Compute total resistance at start
        apx.run();

        this->diag = apx.getDiagonal();
        this->totalValue = 0.;
        G.forNodes([&](node u) { this->totalValue -= static_cast<double>(this->n) * this->diag[u]; });
        originalResistance = -1. * totalValue;

        age.resize(n, -1);
        lpinvVec.resize(n);
        //double exact = -1.0 * static_cast<double>(this->n) * laplacianPseudoinverse(laplacianMatrix(G)).trace();
        //std::cout << "exact: " << exact << ", approx: " << this->totalValue << ", diff: " << exact-totalValue << "\n";
    }

    double getResultResistance() {
        return this->totalValue * (-1.0);
    }

    double getOriginalResistance() {
        return this->originalResistance;
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

        if (k + G.numberOfEdges() > (n * (n-1) / 8*3)) { // 3/4 of the complete graph.
            this->hasRun = true;
            std::cout << "Bad call to TreeGreedy, adding this many edges is not supported! Attempting to have " << k + G.numberOfEdges() << " edges, limit is " << n*(n-1) / 8 *3;
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

                // the random choice following this may fail, we use heuristic information the first time, uniform distribution if it fails
                if (it++ < 2) { 
                    double tr = -1. * this->totalValue;
                    G.forNodes([&](node u) {
                        double val = static_cast<double>(n) * diag[u] + tr;
                        if (val < min) { min = val; }
                        nodeWeights.push_back(val);
                    });
                    for (auto &v : nodeWeights) {
                        auto u = v - min;
                        v = u*u;
                    }
                } else {
                    G.forNodes([&](node u) {
                        nodeWeights.push_back(1.0 / static_cast<double>(n));
                    });
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

            if (this->round == k-1)
            {
                this->validSolution = true;

                auto L = laplacianMatrixSparse(G);
                auto cols = laplacianPseudoinverseColumns(L, {bestEdge.u, bestEdge.v});
                double traceDiff = laplacianPseudoinverseTraceDifference(cols[0], static_cast<int>(bestEdge.u), cols[1], static_cast<int>(bestEdge.v));
                this->totalValue -= traceDiff * static_cast<double>(n);
                G.addEdge(bestEdge.u, bestEdge.v);
                //this->totalValue = 0.;
                //G.forNodes([&](node u) { this->totalValue -= static_cast<double>(this->n) * diag[u]; });

                break;
            } else {
                G.addEdge(bestEdge.u, bestEdge.v);
                apx.edgeAdded(bestEdge.u, bestEdge.v);
                auto cols = apx.getEdgeLpinvVectors();
                Eigen::VectorXd colU(n), colV(n);
                for (int i = 0; i < n; i++) {
                    colU(i) = cols.first[i];
                    colV(i) = cols.second[i];
                }

                // Since these are the columns of the lpinv _after_ adding the new edge, we need to compute the resistance difference by hand.
                node u = bestEdge.u, v = bestEdge.v;
                double R_new = colU(u) + colV(v) - 2 * colU(v);

                updateVec.push_back(1. / (1. - R_new) * (colU - colV));
                updateW.push_back(1. - R_new);

                double traceDiff = (colU-colV).squaredNorm() * (1. / (1. - R_new));
                this->totalValue += traceDiff * static_cast<double>(n);
                this->diag = apx.getDiagonal();
                //this->totalValue = 0.;
                //G.forNodes([&](node u) { this->totalValue -= static_cast<double>(this->n) * diag[u]; });
            }
            this->round++;
        }
        this->hasRun = true;
        this->results = {resultSet.begin(), resultSet.end()};
        INFO("Computed columns: ", computedColumns);
    }

    
private:
    virtual double objectiveDifference(Edge e) {
        auto i = e.u;
        auto j = e.v;

        updateColumn(i);
        updateColumn(j);
        auto& col_i = lpinvVec[i];
        auto& col_j = lpinvVec[j];

        return static_cast<double>(this->n) * (-1.0) * laplacianPseudoinverseTraceDifference(col_i, i, col_j, j);
    }

    void updateColumn(int i) {
        if (age[i] == -1)
        {
            auto col = apx.approxColumn(i);
            lpinvVec[i] = Eigen::VectorXd(n);
            G.forNodes([&](node u) { lpinvVec[i](u) = col[u]; });
            computedColumns++;
        }
        else if (age[i] < this->round) 
        {
            for (int r = age[i]; r < this->round; r++)
                lpinvVec[i] -= updateVec[r] * updateVec[r](i) * updateW[r];
        }
        age[i] = this->round;
    }

    Graph& G;
    int n;
    	bool validSolution = false;
	int round=0;
	std::vector<Edge> results;
	double totalValue = 0.0;
    double epsilon = 0.1;
    double originalResistance = 0.;
    int k;

    std::mt19937 gen;
    std::vector<double> diag;
    ApproxElectricalCloseness apx;

    std::vector<Eigen::VectorXd> updateVec;
    std::vector<double> updateW;
    std::vector<int> age;
    std::vector<Eigen::VectorXd> lpinvVec;

    count computedColumns = 0;

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
        this->originalResistance = -1. * this->totalValue;
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

    double getOriginalResistance() {
        return this->originalResistance;
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
    double originalResistance = 0.;
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
            this->lpinvVec.push_back(lpinv.col(i));
        }
        this->originalResistance = -1. * this->totalValue;
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
    double getOriginalResistance() {
        return this->originalResistance;
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
        //this->G.addEdge(e.u, e.v);
    }

    void updateColumn(int i) {
        if (age[i] < this->round) {
            for (int r = age[i]; r < this->round; r++)
                lpinvVec[i] -= updateVec[r] * updateVec[r](i) * updateW[r];
            age[i] = this->round;
        }
    }

    Graph G;
    int n;
    double originalResistance;
    std::vector<Eigen::VectorXd> updateVec;
    std::vector<double> updateW;
    std::vector<int> age;
    std::vector<Eigen::VectorXd> lpinvVec;
};




#endif // ROBUSTNESS_GREEDY_H
