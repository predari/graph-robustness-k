#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <vector>
#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/graph/Graph.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>


template <class MatrixType>
MatrixType laplacianMatrix(NetworKit::Graph const & g);

template <>
inline Eigen::MatrixXd laplacianMatrix<Eigen::MatrixXd>(NetworKit::Graph const & g) {
	auto n = g.numberOfNodes();
	Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(n, n);
	g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
		if (u == v) {
			std::cout << "Warning: Graph has edge with equal target and destination!";
		}
		laplacian(u, u) += weight;
		laplacian(v, v) += weight;
		laplacian(u, v) -= weight;
		laplacian(v, u) -= weight;
	});
	return laplacian;
}

template <>
inline Eigen::SparseMatrix<double> laplacianMatrix<Eigen::SparseMatrix<double>>(NetworKit::Graph const & g) {
	auto n = g.numberOfNodes();
	auto m = g.numberOfEdges();

	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(3*m);
	Eigen::VectorXd diagonal = Eigen::VectorXd::Constant(n, 0.0) ;
	g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
		if (u == v) {
			std::cout << "Warning: Graph has edge with equal target and destination!";
		}
		tripletList.push_back(T(u, v, -1.));
		tripletList.push_back(T(v, u, -1.));
		diagonal(u) += 1.;
		diagonal(v) += 1.;
	});
	for (int u = 0; u < n; u++) {
		tripletList.push_back(T(u, u, diagonal(u)));
	}
	Eigen::SparseMatrix<double> laplacian(n, n);
	laplacian.setFromTriplets(tripletList.begin(), tripletList.end());

	return laplacian;
}


inline Eigen::SparseMatrix<double> adjacencyMatrix(NetworKit::Graph const & g) {
	auto n = g.numberOfNodes();
	auto m = g.numberOfEdges();

	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(3*m);
	g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
		if (u == v) {
			std::cout << "Warning: Graph has edge with equal target and destination!";
		}
		tripletList.push_back(T(u, v, 1.));
		tripletList.push_back(T(v, u, 1.));
	});
	Eigen::SparseMatrix<double> adjacency(n, n);
	adjacency.setFromTriplets(tripletList.begin(), tripletList.end());

	return adjacency;
}

inline Eigen::SparseMatrix<double> degreeMatrix(NetworKit::Graph const & g) {
	auto n = g.numberOfNodes();
	auto m = g.numberOfEdges();

	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(3*n);
	Eigen::VectorXd diagonal = Eigen::VectorXd::Constant(n, 0.0) ;
	g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
		if (u == v) {
			std::cout << "Warning: Graph has edge with equal target and destination!";
		}
		diagonal(u) += 1.;
		diagonal(v) += 1.;
	});
	for (int u = 0; u < n; u++) {
		tripletList.push_back(T(u, u, diagonal(u)));
	}
	Eigen::SparseMatrix<double> dmat(n, n);
	dmat.setFromTriplets(tripletList.begin(), tripletList.end());

	return dmat;
}



Eigen::SparseMatrix<double> laplacianMatrixSparse(NetworKit::Graph const & g);

Eigen::MatrixXd laplacianPseudoinverse(NetworKit::Graph const & g);
Eigen::MatrixXd laplacianPseudoinverse(Eigen::MatrixXd laplacian);

Eigen::VectorXd laplacianPseudoinverseColumn(const Eigen::SparseMatrix<double> & L, int k);
std::vector<Eigen::VectorXd> laplacianPseudoinverseColumns(const Eigen::SparseMatrix<double> &L, std::vector<NetworKit::node> indices);
std::vector<NetworKit::Vector> laplacianPseudoinverseColumns(const NetworKit::CSRMatrix &laplacian, std::vector<NetworKit::node> indices, double tol=0.1);

void updateLaplacian(Eigen::SparseMatrix<double> &laplacian, NetworKit::node a, NetworKit::node b);


// Update formula for the pseudoinverse of the Laplacian as an edge is added to the graph.
// Add edge (i, j) and compute difference between the k-th column of the pseudoinverse.
// Takes the columns of the pseudoinverse that correspond to the vertices i and j.
// Add this to the old to get the new.
Eigen::VectorXd laplacianPseudoinverseColumnDifference(Eigen::MatrixXd const & lpinv, int i, int j, int k, double conductance = 1.0);

// Update formula for the trace of the lap pinv as an edge is added to the graph.
// Add edge (i, j) and compute the difference of the traces of the pseudoinverses.
// Add this to the old to get the new.
double laplacianPseudoinverseTraceDifference(Eigen::VectorXd const & column_i, int i, Eigen::VectorXd const &column_j, int j, double conductance = 1.0);
double laplacianPseudoinverseTraceDifference(Eigen::MatrixXd const & lpinv, int i, int j, double conductance = 1.0);
double laplacianPseudoinverseTraceDifference(Eigen::MatrixXd const & lpinv, std::vector<NetworKit::Edge> edges, std::vector<double> conductances=std::vector<double>());

// Update formula for the trace of the lpinv as two edges are added to the graph.
//double laplacianPseudoinverseTraceDifference2(Eigen::MatrixXd const & lpinv, Edge e1, Edge e2, double conductance1=1.0, double conductance2=1.0);
double laplacianPseudoinverseTraceDifference2(Eigen::MatrixXd const & lpinv, NetworKit::Edge e1, NetworKit::Edge e2, double conductance1=1.0, double conductance2=1.0);


void updateLaplacianPseudoinverse(Eigen::MatrixXd & lpinv, NetworKit::Edge e, double conductance = 1.0);
Eigen::MatrixXd updateLaplacianPseudoinverseCopy(Eigen::MatrixXd const & lpinv, NetworKit::Edge e, double conductance = 1.0);

// Compute a stochastic approximation of the diagonal of the Moore-Penrose pseudoinverse of the laplacian matrix of a graph.
Eigen::VectorXd approxLaplacianPseudoinverseDiagonal(NetworKit::Graph const &G, double epsilon = 0.1);
Eigen::VectorXd approxEffectiveResistances(NetworKit::Graph const &G, int &pivot, double epsilon = 0.1);


#endif // LAPLACIAN_H