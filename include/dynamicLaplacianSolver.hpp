

#ifndef DYNAMIC_LAPLACIAN_SOLVER_HPP
#define DYNAMIC_LAPLACIAN_SOLVER_HPP

#include <laplacian.hpp>

#include <Eigen/Dense>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/algebraic/CSRMatrix.hpp>

#include <networkit/components/ConnectedComponents.hpp>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>


using namespace NetworKit;


class IDynamicLaplacianSolver {
public:
    virtual void setup(const Graph &g, double tolerance, count eqnsPerRound=200) = 0;
    virtual void computeColumns(std::vector<node> nodes) = 0;
    virtual void addEdge(node u, node v) = 0;
    virtual count getComputedColumnCount() = 0;
    virtual double totalResistanceDifferenceApprox(node u, node v) = 0;
    virtual double totalResistanceDifferenceExact(node u, node v) = 0;
    virtual ~IDynamicLaplacianSolver() {}
};


template <class MatrixType, class Solver>
class DynamicLaplacianSolver : virtual public IDynamicLaplacianSolver {
public:
    virtual void setup(const Graph &g, double tolerance, count eqnsPerRound = 200) override {
        this->laplacian = laplacianMatrix<MatrixType>(g);
        n = laplacian.rows();

        colAge.clear();
        colAge.resize(n, -1);
        cols.clear();
        cols.resize(n);
        updateVec.clear();
        updateW.clear();
        round = 0;

        rhs.clear();
        rhs.resize(omp_get_max_threads(), Eigen::VectorXd::Constant(n, -1.0/n));

        auto t0 = std::chrono::high_resolution_clock::now();

        setup_solver();
        
        auto t1 = std::chrono::high_resolution_clock::now();
        computeColumnSingle(0);

        auto t2 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd u = Eigen::VectorXd::Constant(n, 1.);
        for (int i = 0; i < 1000; i++) {
            u += u * 0.1;
        }
        volatile double a = u(0);
        auto t3 = std::chrono::high_resolution_clock::now();

        double duration0 = (t1 - t0).count();
        double duration1 = (t2 - t1).count();
        double duration2 = (t3 - t2).count() / 1000;

        ageToRecompute = std::max(static_cast<int>(duration1 / duration2), 1);
        roundsPerSolverUpdate = std::max(static_cast<int>(duration0 / duration2 / eqnsPerRound), 1);
    }


    Eigen::VectorXd getColumn(node i) {
        computeColumnSingle(i);
        return cols[i];
    }

    Eigen::MatrixXd solve(const Eigen::MatrixXd& rhs) {
        Eigen::MatrixXd sol = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            // solving failed
            throw std::logic_error("Solving failed.!");
        }


        Eigen::MatrixXd avgs = Eigen::VectorXd::Constant(n, 1. / n).transpose() * sol;
        sol -= Eigen::VectorXd::Constant(n, 1.) * avgs;
        computedColumns += rhs.cols();

        for (count age = solverAge; age < round; age++) {
            auto upv = updateVec[age];
            sol -= upv * upv.transpose() * rhs * updateW[age];
        }
        return sol;
    }

    virtual void computeColumnSingle(node i) {
        if (colAge[i] == -1 || round - colAge[i] > ageToRecompute) {
            cols[i] = solveLaplacianPseudoinverseColumn(solver, rhs[omp_get_thread_num()], n, i);
            computedColumns++;
            colAge[i] = solverAge;
        } 
        for (auto &age = colAge[i]; age < round; age++) {
            cols[i] -= updateVec[age] * updateVec[age](i) * updateW[age];
        }
    }

    virtual void computeColumns(std::vector<node> nodes) override {
        #pragma omp parallel for
        for (auto it = nodes.begin(); it < nodes.end(); it++) {
            computeColumnSingle(*it);
        }
    }

    virtual void addEdge(node u, node v) override {
        computeColumns({u, v});
        const Eigen::VectorXd& colU = cols[u];
        const Eigen::VectorXd& colV = cols[v];

        volatile double R = colU(u) + colV(v) - 2. * colU(v);
        double w = 1. / (1. + R);
        assert(w < 1);

        updateVec.push_back(colU - colV);
        updateW.push_back(w);

        rChange = (colU - colV).squaredNorm() * w * static_cast<double>(n);

    
        laplacian.coeffRef(u, u) += 1.;
        laplacian.coeffRef(v, v) += 1.;
        laplacian.coeffRef(u, v) -= 1.;
        laplacian.coeffRef(v, u) -= 1.;

        round++;

        if (round % roundsPerSolverUpdate == 0) {
            solver.~Solver();
            new (&solver) Solver();
            setup_solver();
        }
    }

    virtual count getComputedColumnCount() override {
        return computedColumns;
    }

    virtual double totalResistanceDifferenceApprox(node u, node v) {
        return static_cast<double>(this->n) * (-1.0) * laplacianPseudoinverseTraceDifference(getColumn(u), u, getColumn(v), v);
    }

    virtual double totalResistanceDifferenceExact(node u, node v) {
        return totalResistanceDifferenceApprox(u, v);
    }



protected:
    virtual void setup_solver() = 0;

    std::vector<int> colAge;
    std::vector<Eigen::VectorXd> cols;
    std::vector<Eigen::VectorXd> updateVec;
    std::vector<double> updateW;

    count n;
    count round = 0;

    std::vector<Eigen::VectorXd> rhs;
    MatrixType laplacian;
    MatrixType solverLaplacian;

	Solver solver; 
    double rChange = 0.;
    count computedColumns = 0;

    // Update solver every n rounds
    count roundsPerSolverUpdate = 1;
    // If a round was last computed this many rounds ago or more, compute by solving instead of updating.
    count ageToRecompute;

    count solverAge = 0;
};



template <class Solver>
class DynamicSparseLaplacianSolver : public DynamicLaplacianSolver<Eigen::SparseMatrix<double>, Solver> {
    virtual void setup_solver() override {
        this->solverLaplacian = this->laplacian;
        this->solverLaplacian.makeCompressed();
        this->solver.compute(this->solverLaplacian);

        if (this->solver.info() != Eigen::Success) {
            throw std::logic_error("Solver Setup failed.");
        }
        this->solverAge = this->round;
    }
};


typedef DynamicSparseLaplacianSolver <Eigen::SparseQR <Eigen::SparseMatrix <double>, Eigen::NaturalOrdering<int> > > SparseQRSolver;

typedef DynamicSparseLaplacianSolver <Eigen::LeastSquaresConjugateGradient <Eigen::SparseMatrix<double> > > SparseLeastSquaresSolver;

typedef DynamicSparseLaplacianSolver <Eigen::SparseLU <Eigen::SparseMatrix<double> , Eigen::COLAMDOrdering<int> > > SparseLUSolver;

typedef DynamicSparseLaplacianSolver <Eigen::SimplicialLLT <Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> > SparseLDLTSolver;

typedef DynamicSparseLaplacianSolver <Eigen::ConjugateGradient <Eigen::SparseMatrix <double>, Eigen::Lower|Eigen::Upper> > SparseCGSolver;


template <class Solver>
class DynamicDenseLaplacianSolver : public DynamicLaplacianSolver<Eigen::MatrixXd, Solver> {
protected:
    virtual void setup_solver() override {
        this->solverLaplacian = this->laplacian;
        this->solver.compute(this->solverLaplacian);
        if (this->solver.info() != Eigen::Success) {
            throw std::logic_error("Solver Setup failed.");
        }
        this->solverAge = this->round;
    }
};


typedef DynamicDenseLaplacianSolver<Eigen::LDLT<Eigen::MatrixXd> > DenseLDLTSolver;

typedef DynamicDenseLaplacianSolver<Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Upper | Eigen::Lower>> DenseCGSolver;




class LamgDynamicLaplacianSolver : virtual public IDynamicLaplacianSolver {
public:
    virtual void setup(const Graph &g, double tolerance, count eqnsPerRound = 200) override {
        this->laplacian = NetworKit::CSRMatrix::laplacianMatrix(g);
        n = g.numberOfNodes();

        colAge.resize(n, -1);
        cols.resize(n);
        solverAge = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        lamg.setup(laplacian);

        
        auto t1 = std::chrono::high_resolution_clock::now();
        solveSingleColumn(0);

        auto t2 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd u = Eigen::VectorXd::Constant(n, 1.);
        for (int i = 0; i < 1000; i++) {
            u += u * 0.1;
        }
        volatile double a = u(0);
        auto t3 = std::chrono::high_resolution_clock::now();
        double duration0 = (t1 - t0).count();
        double duration1 = (t2 - t1).count();
        double duration2 = (t3 - t2).count() / 1000;

        ageToRecompute = std::max(static_cast<int>(duration1 / duration2), 1);
        roundsPerSolverUpdate = std::max(static_cast<int>(duration0 / duration2 / eqnsPerRound), 1);
    }

    void solveSingleColumn(node u) {
        Vector rhs(n, -1.0 / static_cast<double>(n));
        Vector x (n);
        rhs[u] += 1.;
        auto status = lamg.solve(rhs, x);
        assert(status.converged);

        double avg = x.transpose() * Vector(n, 1.0 / static_cast<double>(n));
        x -= avg;

        cols[u] = Eigen::VectorXd(n);
        for (int i = 0; i < n; i++) {
            cols[u](i) = x[i];
        }

        colAge[u] = solverAge;
        computedColumns++;
    }


    Eigen::VectorXd getColumn(node u) {
        computeColumns({u});
        return cols[u];
    }

    virtual void computeColumns(std::vector<node> nodes) override {
        for (auto i: nodes) {
            if (colAge[i] == -1 || round - colAge[i] > ageToRecompute) {
                solveSingleColumn(i);
            } 
            for  (count r = colAge[i]; r < round; r++) {
                cols[i] -= updateVec[r] * (updateVec[r](i) * updateW[r]);
            }
            colAge[i] = round;
        }        
    }

    virtual void addEdge(node u, node v) override {
        computeColumns({u, v});
        auto colU = getColumn(u);
        auto colV = getColumn(v);

        double R = colU(u) + colV(v) - 2*colU(v);
        double w = (1. / (1. + R));
        assert(w < 1.);
        auto upv = colU - colV;

        rChange = (colU - colV).squaredNorm() * w * static_cast<double>(n);

        laplacian.setValue(u, u, laplacian(u, u) + 1.);
        laplacian.setValue(v, v, laplacian(v, v) + 1.);
        laplacian.setValue(u, v, laplacian(u, v) - 1.);
        laplacian.setValue(v, u, laplacian(v, u) - 1.);

        updateVec.push_back(upv);
        updateW.push_back(w);
        round++;

        if (round % roundsPerSolverUpdate == 0) {
            lamg.~Lamg<CSRMatrix>();
            new (&lamg) Lamg<CSRMatrix>();
            lamg.setup(laplacian);

            solverAge = round;
        }
    }

    virtual count getComputedColumnCount() override {
        return computedColumns;
    }

    virtual double totalResistanceDifferenceApprox(node u, node v) {
        return static_cast<double>(this->n) * (-1.0) * laplacianPseudoinverseTraceDifference(getColumn(u), u, getColumn(v), v);
    }

    virtual double totalResistanceDifferenceExact(node u, node v) {
        return totalResistanceDifferenceApprox(u, v);
    }



private:
    std::vector<count> colAge;
    std::vector<Eigen::VectorXd> cols;
    std::vector<Eigen::VectorXd> updateVec;
    std::vector<double> updateW;

    count n;
    count round = 0;
    CSRMatrix laplacian;

    double rChange = 0.;
    count computedColumns;

    // Update solver every n rounds
    count roundsPerSolverUpdate = 10;
    // If a round was last computed this many rounds ago or more, compute by solving instead of updating.
    count ageToRecompute;

    count solverAge = 0;

    Lamg<CSRMatrix> lamg;
};




template <class MatrixType, class DynamicSolver>
class JLTSolver : virtual public IDynamicLaplacianSolver {
public:
    virtual void setup(const Graph &g, double tolerance, count eqnsPerRound=200) override {
        n = g.numberOfNodes();
        m = g.numberOfEdges();

        epsilon = tolerance;

        l = jltDimension(n, epsilon);

        G = g;

        solver.setup(g, tolerance, 2*l + 2);
        incidence = incidenceMatrix(g);

        computeIntermediateMatrices();
    }

    virtual void computeColumns(std::vector<node> nodes) override {
        // pass
    }


    virtual void addEdge(node u, node v) override {
        G.addEdge(u, v);
        solver.addEdge(u, v);
        m++;

        incidence = incidenceMatrix(G);
        //updateIncidenceMatrix(incidence, u, v);
        computeIntermediateMatrices();
    }

    virtual count getComputedColumnCount() override {
        return solver.getComputedColumnCount();
    }

    virtual double totalResistanceDifferenceApprox(node u, node v) override {
        double R = effR(u, v);
        double phiNormSq = phiNormSquared(u, v);

        return n / (1. + R) * phiNormSq;
    }

    virtual double totalResistanceDifferenceExact(node u, node v) override {
        return solver.totalResistanceDifferenceExact(u, v);
    }

    double effR(node u, node v) {
        return (PBL.col(u) - PBL.col(v)).squaredNorm();
    }

    double phiNormSquared(node u, node v) {
        return (PL.col(u) - PL.col(v)).squaredNorm();
    }


private:
    int jltDimension(int n, double epsilon) {
        return 2 * std::max(std::log(n) / (epsilon * epsilon / 2 - epsilon * epsilon * epsilon / 3), 1.);
    }

    void computeIntermediateMatrices() {
        // Generate projection matrices

        auto random_projection = [&](count n_rows, count n_cols) {
            auto normal = [&](int) { return d(gen) / std::sqrt(n_rows); };
            Eigen::MatrixXd P = Eigen::MatrixXd::NullaryExpr(n_rows, n_cols, normal);
            Eigen::VectorXd avg = P * Eigen::VectorXd::Constant(n_cols, 1. / n_cols);
            P -= avg * Eigen::MatrixXd::Constant(1, n_cols, 1.);
            return P;
        };

        P_n = random_projection(l, n);
        P_m = random_projection(l, m);


        // Compute columns of P L^\dagger and P' B^T L^\dagger where B is the incidence matrix of G
        // We first compute the transposes of the targets. For the first, solve LX = P^T, for the second solve LX = B P^T

        Eigen::MatrixXd rhs1 = P_n.transpose(); 
        Eigen::MatrixXd rhs2 = incidence * P_m.transpose();

        PL = solver.solve(rhs1);
        PBL = solver.solve(rhs2);

        PL.transposeInPlace();
        PBL.transposeInPlace();
    }



    DynamicSolver solver;

    count n, m;
    count l;

    Eigen::MatrixXd P_n, P_m;
    Eigen::MatrixXd PL, PBL;

    double epsilon;


    std::mt19937 gen{1};
    std::normal_distribution<> d{0, 1};

    Graph G;
    Eigen::SparseMatrix<double> incidence;
};


typedef JLTSolver<Eigen::SparseMatrix<double>, SparseLUSolver> JLTLUSolver;

#endif // DYNAMIC_LAPLACIAN_SOLVER_HPP