#include <laplacian.hpp>


#include <Eigen/Dense>


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


// Compute the column of the pseudoinverse from scratch. 
// If multiple of this are needed then the solver should be reused for better performance.
Vector computeLaplacianPseudoinverseColumn(CSRMatrix const &laplacian, int index, bool connected)
{
	NetworKit::Lamg<CSRMatrix> solver;
	if (connected)
	{
		solver.setupConnected(laplacian);
	}
	else
	{
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

// Update formula for the pseudoinverse of the Laplacian as an edge is added to the graph.
// Add edge (i, j) and compute difference between the k-th column of the pseudoinverse.
// Takes the columns of the pseudoinverse that correspond to the vertices i and j.
// Add this to the old to get the new.
Vector laplacianPseudoinverseColumnDifference(Vector const & column_i, int i, Vector const & column_j, int j, int k, double conductance)
{
	double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
	double w = 1.0 / (1.0 / conductance + R_ij);
	Vector v = column_i - column_j;
	return v * (v[k] * w * (-1.0));
}

// Update formula for the trace of the lap pinv as an edge is added to the graph.
// Add edge (i, j) and compute the difference of the traces of the pseudoinverses.
// Add this to the old to get the new.
double laplacianPseudoinverseTraceDifference(Vector const &column_i, int i, Vector const &column_j, int j, double conductance)
{
	double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
	double w = 1.0 / (1.0 / conductance + R_ij);
	Vector v = column_i - column_j;
	return Vector::innerProduct(v, v) * w * (-1.0);
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


// Compute a stochastic approximation of the diagonal of the Moore-Penrose pseudoinverse of the laplacian matrix of a graph.
std::vector<double> approxLaplacianPseudoinverseDiagonal(Graph const &G, double epsilon)
{
	// TODO: Check that graph is claplacianPseudoinverseTraceDifferenceonnected
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