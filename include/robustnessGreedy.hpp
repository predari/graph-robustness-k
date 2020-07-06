#ifndef ROBUSTNESS_GREEDY_H
#define ROBUSTNESS_GREEDY_H


#include <utility>
#include <greedy.hpp>
#include <laplacian.hpp>

#include <networkit/algebraic/Vector.hpp>
#include <networkit/algebraic/CSRMatrix.hpp>
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
        auto laplacian = CSRMatrix::laplacianMatrix(this->G);
        NetworKit::Lamg<CSRMatrix> solver;
        solver.setupConnected(laplacian);
        for (int i = 0; i < this->n; i++) {
            Vector ePivot(n, 0);
            ePivot[i] = 1;
            ePivot -= 1.0/n;
            
            Vector lpinvCol (n, 0);
            solver.solve(ePivot, lpinvCol);
            this->lpinv.push_back(lpinvCol);
            this->totalValue -= this->n * lpinvCol[i];
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

    
private:

    virtual double objectiveDifference(Edge e) override {
        auto i = e.u;
        auto j = e.v;
        return (-1.0) * laplacianPseudoinverseTraceDifference(lpinv[i], i, lpinv[j], j) * n;
    }

    virtual void useItem(Edge e) override {
        updateLaplacianPseudoinverse(this->lpinv, e, 1.0);
        //this->G.addEdge(e.u, e.v);
    }

    std::vector<Vector> lpinv;

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

        auto laplacian = CSRMatrix::laplacianMatrix(G);
        NetworKit::Lamg<CSRMatrix> solver;
        solver.setupConnected(laplacian);
        for (int i = 0; i < this->n; i++) {
            Vector ePivot(n, 0);
            ePivot[i] = 1;
            ePivot -= 1.0/n;
            
            Vector lpinvCol (n, 0);
            solver.solve(ePivot, lpinvCol);
            this->lpinv.push_back(lpinvCol);
            this->totalValue -= this->n * lpinvCol[i];
        }
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

private:
    virtual double objectiveDifference(Edge e) override {
        auto i = e.u;
        auto j = e.v;
        return (-1.0) * laplacianPseudoinverseTraceDifference(lpinv[i], i, lpinv[j], j) * n;
    }

    virtual void useItem(Edge e) override {
        updateLaplacianPseudoinverse(this->lpinv, e, 1.0);
        //this->G.addEdge(e);
    }

    std::vector<Vector> lpinv;

    Graph G;
    int n;
};


#endif // ROBUSTNESS_GREEDY_H
