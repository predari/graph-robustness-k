#include <laplacian.hpp>


#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>


std::vector<double> approxEffectiveResistances(Graph const & G, int &pivot)
{
	ApproxEffectiveResistance resAlg(G);
	resAlg.init();
	resAlg.numberOfUSTs = resAlg.computeNumberOfUSTs();
	resAlg.run();
	auto resistances = resAlg.getApproxEffectiveResistances();
	pivot = resAlg.getRoot();
	return resistances;
}

Eigen::MatrixXd laplacianMatrix(Graph const & g) {
	auto n = g.numberOfNodes();
	Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(n, n);
	// TODO make sure this is zero-initialized
	g.forEdges([&](node u, node v, double weight) {
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

Eigen::MatrixXd laplacianPseudoinverse(Eigen::MatrixXd lp) {
	int n = lp.rows();
	Eigen::MatrixXd J = Eigen::MatrixXd::Constant(n, n, 1.0 / n);
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);

	return (lp + J).llt().solve(I) - J;
}
Eigen::MatrixXd laplacianPseudoinverse(Graph const & g) {
	return laplacianPseudoinverse(laplacianMatrix(g));
}

// Compute the column of the pseudoinverse from scratch. 
// If multiple of this are needed then the solver should be reused for better performance.
Vector computeLaplacianPseudoinverseColumn(CSRMatrix const &laplacian, int index, bool connected)
{
	NetworKit::Lamg<CSRMatrix> solver;
	if (connected)
	{
		solver.setupConnected(laplacian);
	} else {
		solver.setup(laplacian);
	}
	int n = laplacian.numberOfColumns();

	Vector ePivot(n, 0);
	ePivot[index] = 1;
	ePivot -= 1.0 / n;

	Vector lpinvPivotColumn(n, 0);
	solver.solve(ePivot, lpinvPivotColumn);
	return lpinvPivotColumn;
}


Vector laplacianPseudoinverseColumnDifference(Vector const & column_i, int i, Vector const & column_j, int j, int k, double conductance)
{
	double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
	double w = 1.0 / (1.0 / conductance + R_ij);
	Vector v = column_i - column_j;
	return v * (v[k] * w * (-1.0));
}

Eigen::VectorXd laplacianPseudoinverseColumnDifference(Eigen::MatrixXd const & lpinv, int i, int j, int k, double conductance) {
	double R_ij = lpinv(i,i) + lpinv(j,j) - 2*lpinv(i, j);
	double w = 1.0 / (1.0 / conductance + R_ij);
	Eigen::VectorXd v = lpinv.col(i) - lpinv.col(j);
	return v * (v(k) * w * (-1.0));
}


// Update formula for the trace of the lap pinv as an edge is added to the graph.
// Add edge (i, j) and compute the difference of the traces of the pseudoinverses.
// Depends only on i-th and j-th column of lpinv.
// Add this to the old to get the new.
double laplacianPseudoinverseTraceDifference(Vector const &column_i, int i, Vector const &column_j, int j, double conductance)
{
	double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
	double w = 1.0 / (1.0 / conductance + R_ij);
	Vector v = column_i - column_j;
	return Vector::innerProduct(v, v) * w * (-1.0);
}

double laplacianPseudoinverseTraceDifference(Eigen::VectorXd const & column_i, int i, Eigen::VectorXd const &column_j, int j, double conductance)
{
	double R_ij = column_i(i) + column_j(j) - 2 * column_i(j);
	double w = 1.0 / (1.0 / conductance + R_ij);
	return (column_i - column_j).squaredNorm() * w * (-1.0);
}


double laplacianPseudoinverseTraceDifference(Eigen::MatrixXd const & lpinv, int i, int j, double conductance) {
	double R_ij = lpinv(i,i) + lpinv(j,j) - 2*lpinv(i, j);
	double w = 1.0 / (1.0 / conductance + R_ij);
	//Eigen::VectorXd v = lpinv.col(i) - lpinv.col(j);
	//return v.squaredNorm() * w * (-1.0);

	// Avoid allocation of vector
	auto & col_i = lpinv.col(i);
	auto & col_j = lpinv.col(j);
	return (col_i - col_j).squaredNorm() * w * (-1.0);
	//return (col_i.squaredNorm() + col_j.squaredNorm() - 2 *  col_i.dot(col_j)) * w * (-1.0);
}

// Update formula for the trace of the lap pinv as two edges are added to the graph.
// Based on Sherman-Morrison formula.
double laplacianPseudoinverseTraceDifference2(std::vector<Vector> const & columns, Edge e1, Edge e2, double conductance1, double conductance2) 
{
	auto e1_vec = columns[e1.u] - columns[e1.v];
	auto e2_vec = columns[e2.u] - columns[e2.v];
	double a = conductance1 + e1_vec[e1.u] - e1_vec[e1.v]; 
	double b = e1_vec[e2.u] - e1_vec[e2.v];
	double c = conductance2 + e2_vec[e2.u] - e2_vec[e2.v];
	double det = a*c-b*b;
	return (-1.0) / det * (Vector::innerProduct(e1_vec, e1_vec) * c + 2.0*Vector::innerProduct(e1_vec, e2_vec) * b + Vector::innerProduct(e2_vec, e2_vec) * a);
}

// For small numbers of edges added (O(k^3)). Based on Woodbury matrix identity.
double laplacianPseudoinverseTraceDifference(Eigen::MatrixXd const & lpinv, std::vector<Edge> edges, std::vector<double> conductances)
{
	int k = edges.size();
	int n = lpinv.rows();
	Eigen::MatrixXd u = Eigen::MatrixXd::Zero (n, k);
	Eigen::MatrixXd c_inv = Eigen::MatrixXd::Zero(k, k);
	for (int i = 0; i < k; i++) {
		double resistance = 1.0;
		if (i < conductances.size()) {
			resistance = 1.0/conductances[i];
		}
		c_inv(i,i) = resistance;
		auto e = edges[i];
		u(e.u, i) = 1.0;
		u(e.v, i) = -1.0;
	}
	//auto a = lpinv * u;
	//return (-1.0) * (a * (c_inv + u.transpose() * lpinv * u).inverse() * a.transpose()).trace();
	return (-1.0) * (lpinv * u * (c_inv + u.transpose() * lpinv * u).inverse() * u.transpose() * lpinv).trace();
}

void updateLaplacianPseudoinverse(std::vector<Vector> & columns, Edge e, double conductance) {
	auto i = e.u;
	auto j = e.v;
	int n = columns.size();
	std::vector<Vector> changes(n);
	for (int k = 0; k < n; k++) {
		changes[k] = laplacianPseudoinverseColumnDifference(columns[i], i, columns[j], j, k, conductance);
	}
	for (int k = 0; k < n; k++) {
		columns[k] += changes[k];
	}
}

void updateLaplacianPseudoinverse(Eigen::MatrixXd & lpinv, Edge e, double conductance) {
	auto i = e.u;
	auto j = e.v;
	double R_ij = lpinv(i,i) + lpinv(j,j) - 2*lpinv(i, j);
	double w_negative = -1.0 / (1.0 / conductance + R_ij);
	Eigen::VectorXd v = lpinv.col(i) - lpinv.col(j);
	//Eigen::VectorXd col_i = lpinv.col(i);
	//Eigen::VectorXd col_j = lpinv.col(j);

	int n = lpinv.cols();
	for (int i = 0; i < n; i++) {
		lpinv.col(i) += v(i) * w_negative * v;
	}
	
	// This version causes pagefaults on my system which slows down the entire thing by a factor of 10 ...
	//lpinv += (w_negative * v) * v.transpose();
}

Eigen::MatrixXd updateLaplacianPseudoinverseCopy(Eigen::MatrixXd const & lpinv, Edge e, double conductance) {
	auto i = e.u;
	auto j = e.v;
	double R_ij = lpinv(i,i) + lpinv(j,j) - 2*lpinv(i, j);
	double w = 1.0 / (1.0 / conductance + R_ij);
	Eigen::VectorXd v = lpinv.col(i) - lpinv.col(j);
	
	return lpinv - w * v * v.transpose();
}


// Compute a stochastic approximation of the diagonal of the Moore-Penrose pseudoinverse of the laplacian matrix of a graph.
std::vector<double> approxLaplacianPseudoinverseDiagonal(Graph const &G, double epsilon)
{
	// TODO: Check that graph is connected
	auto n = G.numberOfNodes();

	int pivot;
	auto resistances = approxEffectiveResistances(G, pivot);

	auto L = CSRMatrix::laplacianMatrix(G);
	auto lpinvPivotColumn = computeLaplacianPseudoinverseColumn(L, pivot, true);

	Vector lpinvDiag(n, 0);
	std::vector<double> lpinvDiagStdVec;
	for (auto i = 0; i < n; i++)
	{
		lpinvDiag[i] = resistances[i] - lpinvPivotColumn[pivot] + 2 * lpinvPivotColumn[i];
		lpinvDiagStdVec.push_back(lpinvDiag[i]);
	}

	return lpinvDiagStdVec;
}