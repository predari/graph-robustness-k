#ifndef ROBUSTNESS_GREEDY_H
#define ROBUSTNESS_GREEDY_H


#include <utility>
#include <greedy.hpp>
#include <laplacian.hpp>

#include <networkit/algebraic/Vector.hpp>
#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/graph/Graph.hpp>


typedef std::pair<int, int> R_Edge;
using namespace NetworKit;


class RobustnessGreedy final : public SubmodularGreedy<R_Edge>{
public:

    void init(Graph G, int k) {
        this->G = G;
        this->n = G.numberOfNodes();
        this->k = k;
    }
private:

    virtual double objectiveDifference(R_Edge e) override {
        auto i = e.first;
        auto j = e.second;
        return (-1.0) * laplacianPseudoinverseTraceDifference(lpinv[i], i, lpinv[j], j) * n;
    }

    virtual void useItem(R_Edge e) override {
        auto i = e.first;
        auto j = e.second;
        Vector col_i = lpinv[i];
        Vector col_j = lpinv[j];

        for (auto l = 0; l<n; l++) {
            lpinv[l] += laplacianPseudoinverseColumnDifference(col_i, i, col_j, j, l);
        }
        this->G.addEdge(e.first, e.second);
    }

    virtual void initRound() override {
        if (this->round == 0) {
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
            }

            // Add edges to items of greedy
            std::vector<R_Edge> items;
            for (size_t i = 0; i < this->n; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    if (i != j && !this->G.hasEdge(i, j)) {
                        items.push_back(R_Edge(i,j));
                    }
                }
            }
            
            this->addItems(items);
        }
    }

    std::vector<Vector> lpinv;

    Graph G;
    int n;
};

class RobustnessDiagonalGreedy : public SubmodularGreedy<R_Edge> {
public:
    virtual double objectiveDifference(R_Edge e) override {
        auto i = e.first;
        auto j = e.second;
        auto column_i = this->getColumn(i);
        auto column_j = this->getColumn(j);

        double conductance = 1.0;

        double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
        double w = 1.0 / (1.0 / conductance + R_ij);
        Vector updateVec = column_i - column_j;

        return Vector::innerProduct(updateVec, updateVec) * w * (1.0) * this->n;
    }

    virtual void useItem(R_Edge e) override  {
        // Save data to update columns later.
        lpinv_update upd;
        auto i = e.first;
        auto j = e.second;
        auto column_i = this->getColumn(i);
        auto column_j = this->getColumn(j);

        double conductance = 1.0;

        double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
        double w = 1.0 / (1.0 / conductance + R_ij);
        Vector updateVec = column_i - column_j;

        upd.edge = e;
        upd.w = w;
        upd.updateVec = updateVec;
        lpinv_updates[this->round] = upd;

        // Update Graph
        this->G.addEdge(e.first, e.second);
    }
    virtual void initRound() override  {
        this->resetItems();

        // Determine the r elements with maximal diagonal entry
        std::vector<int> nodes;
        if (this->round == 0) {
            this->diagonal = Vector (this->n, 0.0);
            for (int i = 0; i < this->n; i++) {
                this->diagonal[i] = this->getColumn(i)[i];
            }
            /*
            int pivot;
            auto resistances = approxEffectiveResistances(G, pivot);

            auto resistanceVec = Vector(resistances);

            Vector pivotCol = this->getColumn(pivot);
            this->diagonal = resistanceVec;
            this->diagonal -= Vector(this->n, pivotCol[pivot]);
            this->diagonal += pivotCol*2.0;
            */
        }

        std::priority_queue<std::pair<double, int>> q;
        for (int i = 0; i < this->n; i++) {
            q.push(std::pair<double, int>(this->diagonal[i], i));
        }
        for (int i = 0; i < this->r; i++) {
            if (q.empty()) {
                break;
            }
            auto v = q.top().second;
            nodes.push_back(v);
            q.pop();
        }

        std::vector<R_Edge> items;
        for (auto i = 0; i < nodes.size(); i++)
        {
            auto n1 = nodes[i];
            for (auto j = 0; j < i; j++)
            {
                auto n2 = nodes[j];
                if (i != j && !this->G.hasEdge(n1, n2)) {
                    items.push_back(R_Edge(n1, n2));
                }
            }
        }
        

        // Set up new edge candidates
        this->addItems(items);
    }
    virtual bool checkSolution() override {
        return (this->round+1 == this->k);
    }


    void init(Graph G, int k, int r) {
        this->G = G;
        this->n = G.numberOfNodes();
        this->k = k;
        this->lpinv_columns.resize(n);
        this->lpinv_updates.resize(k);
        this->diagonal = Vector(this->n);
        this->r = r;

        auto laplacian = CSRMatrix::laplacianMatrix(G);
        solver.setupConnected(laplacian);
    }

    Vector getColumn(int i) {
        auto& col_ = lpinv_columns[i];
        if (col_.lastUpdatedRound == this->round) {
            return col_.column;
        }
        if (col_.lastUpdatedRound == -1) {
            // Compute from scratch
            Vector ePivot(n, 0);
            ePivot[i] = 1;
            ePivot -= 1.0/n;
            
            Vector lpinvCol (n, 0);
            solver.solve(ePivot, lpinvCol);
            col_.column = lpinvCol;
            this->diagonal[i] = col_.column[i];
            col_.lastUpdatedRound = 0;
        }
        
        // Update up to current round
        for (auto j = col_.lastUpdatedRound; j < this->round; j++) {
            auto upd = this->lpinv_updates[j];
            col_.column += upd.updateVec * upd.updateVec[i] * upd.w * (-1.0);
            this->diagonal[i] = col_.column[i];
        }
        col_.lastUpdatedRound = this->round;
        return col_.column;
    }

    struct lpinv_update {
        Vector updateVec;
        R_Edge edge;
        double w;
    };
    struct lpinv_column {
        Vector column;
        int lastUpdatedRound=-1;
    };

private:
    std::vector<lpinv_update> lpinv_updates; 
    std::vector<lpinv_column> lpinv_columns;
    Vector diagonal;
    Graph G;
    int n;
    int k;
    int r;
    NetworKit::Lamg<CSRMatrix> solver;

};





class RobustnessStochasticGreedy : public StochasticGreedy<R_Edge>{
public:
    void init(Graph G, int k, double epsilon=0.1) {
        //this->G = G;
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
        }

        // Add edges to items of greedy
        std::vector<R_Edge> items;
        for (size_t i = 0; i < this->n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (i != j && !this->G.hasEdge(i, j)) {
                    items.push_back(R_Edge(i,j));
                }
            }
        }
        
        this->addItems(items);
    }

private:
    virtual double objectiveDifference(R_Edge e) override {
        auto i = e.first;
        auto j = e.second;
        return (-1.0) * laplacianPseudoinverseTraceDifference(lpinv[i], i, lpinv[j], j) * n;
    }

    virtual void useItem(R_Edge e) override {
        auto i = e.first;
        auto j = e.second;
        Vector& col_i = lpinv[i];
        Vector& col_j = lpinv[j];

        for (auto l = 0; l<n; l++) {
            lpinv[l] += laplacianPseudoinverseColumnDifference(col_i, i, col_j, j, l);
        }
        //this->G.addEdge(e.first, e.second);
    }

    virtual void initRound() override {
        /*if (this->round == 0) {
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
            }

            // Add edges to items of greedy
            std::vector<R_Edge> items;
            for (size_t i = 0; i < this->n; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    if (i != j && !this->G.hasEdge(i, j)) {
                        items.push_back(R_Edge(i,j));
                    }
                }
            }
            
            this->addItems(items);
        }*/
    }

    std::vector<Vector> lpinv;

    Graph G;
    int n;
};


#endif // ROBUSTNESS_GREEDY_H
