#ifndef ROBUSTNESS_UST_GREEDY_H
#define ROBUSTNESS_UST_GREEDY_H


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
#include <networkit/components/ConnectedComponents.hpp>


using namespace NetworKit;


enum class heuristicType {
    lpinvDiag,
    centrality
};

enum class linalgType {
    lu,
    lamg
};


template <linalgType linalg=linalgType::lu>
class RobustnessTreeGreedy final : public Algorithm {
public:
    RobustnessTreeGreedy(Graph &G, int k, double epsilon=0.1, double epsilon_tree_count = 0.1, heuristicType h=heuristicType::lpinvDiag) : G(G), apx(G, epsilon_tree_count) {
        this->n = G.numberOfNodes();
        this->k = k;
        this->epsilon = epsilon;
        this->heuristic = h;
    }

    void init() {
        gen = std::mt19937(Aux::Random::getSeed());

        // Compute total resistance at start
        apx.run();
        INFO("Usts: ", apx.getUstCount());

        this->diag = apx.getDiagonal();
        this->totalValue = 0.;
        G.forNodes([&](node u) { this->totalValue -= static_cast<double>(this->n) * this->diag[u]; });
        originalResistance = -1. * totalValue;

        age.resize(n, -1);
        lpinvVec.resize(n);
        laplacian = laplacianMatrixSparse(G);
        //double exact = -1.0 * static_cast<double>(this->n) * laplacianPseudoinverse(laplacianMatrix(G)).trace();
        //std::cout << "exact: " << exact << ", approx: " << this->totalValue << ", diff: " << exact-totalValue << "\n";

        if (linalg == linalgType::lamg) {
            lamg = SolverLamg()
        }
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
                unsigned int s = (unsigned int)std::sqrt(1.0 * (this->n * (this->n-1) /2 - this->G.numberOfEdges() - this->round) / k * std::log(1.0/epsilon));
                if (s < 2) { s = 2; }
                if (s > n/2) { s = n/2; }
                if (r == 0) { INFO("Columns in round 1: ", s); }

                double min = std::numeric_limits<double>::infinity();

                std::vector<double> nodeWeights;

                // the random choice following this may fail if all the vertex pairs are already present as edges, we use heuristic information the first time, uniform distribution if it fails
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
                        nodeWeights.push_back(1.0);
                    });
                }

                std::discrete_distribution<> distribution_nodes_heuristic (nodeWeights.begin(), nodeWeights.end());

                nodeWeights.clear();
                for (int i = 0; i < n; i++) {
                    nodeWeights.push_back(1.0);
                }
                std::discrete_distribution<> distribution_nodes_uniform (nodeWeights.begin(), nodeWeights.end());

                while (nodes.size() < s / 2) {
                    nodes.insert(distribution_nodes_heuristic(gen));
                }
                
                while (nodes.size() < s) {
                    nodes.insert(distribution_nodes_uniform(gen));
                }
                std::vector<node> nodesVec {nodes.begin(), nodes.end()};
                updateColumns(nodesVec);

                // Determine best edge between nodes from node set
                auto edgeValid = [&](node u, node v){
                    if (this->G.hasEdge(u, v) || this->G.hasEdge(v, u)) return false;
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

            auto u = bestEdge.u;
            auto v = bestEdge.v;
            G.addEdge(u, v);
            laplacian.coeffRef(u, u) += 1.;
            laplacian.coeffRef(v, v) += 1.;
            laplacian.coeffRef(u, v) -= 1.;
            laplacian.coeffRef(v, u) -= 1.;

            auto& colU = lpinvVec[u];
            auto& colV = lpinvVec[v];

            double R = colU(u) + colV(v) - 2*colU(v);
            double w = (1. / (1. + R));
            auto upv = colU - colV;

            double traceDiff = (colU - colV).squaredNorm() * w;
            this->totalValue += traceDiff * static_cast<double>(n);


            if (this->round < k-1) {
                apx.edgeAdded(u, v);
                this->diag = apx.getDiagonal();
                updateVec.push_back(upv);
                updateW.push_back(w);
            } else {
                this->validSolution = true;
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

        double difference = static_cast<double>(this->n) * (-1.0) * laplacianPseudoinverseTraceDifference(col_i, i, col_j, j);
        
        return difference;
    }

    void updateColumns(std::vector<node> indices) {

        if (linalg == linalgType::lu) {
            std::vector<node> notComputed;
            for (auto& ind: indices) {
                if (age[ind] == -1) {
                    notComputed.push_back(ind);
                }
            }
            try {
                auto cols = laplacianPseudoinverseColumns(laplacian, notComputed);

                for (int i = 0; i < notComputed.size(); i++) {
                    auto index = notComputed[i];
                    lpinvVec[index] = cols[i];
                    age[index] = this->round;
                }

            } catch (const std::logic_error& e) {
                std::cout << e.what() << std::endl;

                throw(e);
            }
            for (auto& ind: indices) {
                updateColumn(ind);
            }

        } else if (linalg == linalgType::lamg) {
            std::vector<node> notComputed;
            for (auto& ind: indices) {
                if (age[ind] == -1) {
                    notComputed.push_back(ind);
                }
            }


        }
        
        
    }


    void updateColumn(int i) {
        if (age[i] == -1)
        {
            lpinvVec[i] = laplacianPseudoinverseColumn(laplacian, i);
            this->computedColumns++;
        }
        else if (age[i] < this->round) 
        {
            for (int r = age[i]; r < this->round; r++)
                lpinvVec[i] -= updateVec[r] * updateVec[r](i) * updateW[r];
            /*
            auto col_exact = laplacianPseudoinverseColumn(laplacian, i);
            G.forNodes([&](node u) {
                double error = std::abs(lpinvVec[i](u) - col_exact(u));
                if (error > 0.001) {
                    std::cout << "Column Error! (" << i << ", " << u << "): " << error << ", rel " << error / std::abs(col_exact(u)) << ", round " << this->round << ", age: " << age[i] << std::endl;
                }
            });
            */
        }
        age[i] = this->round;
    }

    Graph& G;
	std::vector<Edge> results;

    int n;
    bool validSolution = false;
	int round=0;
    int k;

	double totalValue = 0.0;
    double epsilon = 0.1;
    double originalResistance = 0.;

    std::mt19937 gen;
    std::vector<double> diag;
    ApproxElectricalCloseness apx;

    std::vector<Eigen::VectorXd> updateVec;
    std::vector<double> updateW;
    std::vector<int> age;
    std::vector<Eigen::VectorXd> lpinvVec;

    count computedColumns = 0;

    Eigen::SparseMatrix<double> laplacian;
    heuristicType heuristic;

    std::unique_ptr<SolverLamg<CSRMatrix>> lamg;
};


#endif
