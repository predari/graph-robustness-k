#ifndef ROBUSTNESS_GREEDY_H
#define ROBUSTNESS_GREEDY_H


#include <utility>

#include <greedy.hpp>
#include <laplacian.hpp>

#include <Eigen/Dense>

#include <networkit/graph/Graph.hpp>


using namespace NetworKit;



// The greedy algorithms in this file optimize for large -R_tot.



inline bool operator<(Edge const &lhs, Edge const & rhs) {
    return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
}



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
                if (i != j && !this->G.hasEdge(i, j)) {
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
