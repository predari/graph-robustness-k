#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/centrality/ApproxEffectiveResistance.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/algebraic/Vector.hpp>

using Eigen::MatrixXd;
using namespace NetworKit;

std::vector<double> approxEffectiveResistances(Graph G)
{
	ApproxEffectiveResistance resAlg(G);
	resAlg.init();
	resAlg.numberOfUSTs = resAlg.computeNumberOfUSTs();
	resAlg.run();
	auto resistances = resAlg.getApproxEffectiveResistances();
	return resistances;
}

// Compute a stochastic approximation of the diagonal of the Moore-Penrose pseudoinverse of the laplacian matrix of a graph.
std::vector<double> approxLaplacianPseudoinverseDiagonal(Graph G, double epsilon=0.1)
{
	// TODO: Check that graph is connected
	auto n = G.numberOfNodes();

	ApproxEffectiveResistance resAlg(G, epsilon);
	resAlg.init();
	resAlg.numberOfUSTs = resAlg.computeNumberOfUSTs();
	resAlg.run();
	auto resistances = resAlg.getApproxEffectiveResistances();
	auto pivot = resAlg.getRoot();


	auto L = CSRMatrix::laplacianMatrix(G);

	NetworKit::Lamg<CSRMatrix> solver;
	solver.setupConnected(L);

	Vector ePivot(n, 0);
	ePivot[pivot] = 1;
	ePivot.apply([&](auto d) { return d - 1.0/n; });

	Vector lpinvPivotColumn (n, 0);
	solver.solve(ePivot, lpinvPivotColumn);

	Vector lpinvDiag (n, 0);
	std::vector<double> lpinvDiagStdVec;
	for (auto i = 0; i < n; i++) {
		lpinvDiag[i] = resistances[i] - lpinvPivotColumn[pivot] + 2*lpinvPivotColumn[i];
		lpinvDiagStdVec.push_back(lpinvDiag[i]);
	}

	return lpinvDiagStdVec;
}

int main()
{
	Aux::Random::setSeed(42, false);
	auto G1 = NetworKit::ErdosRenyiGenerator(
				  100, 0.15, false)
				  .generate();

	Graph G2 = NetworKit::ErdosRenyiGenerator(10, 0.15, false).generate();
	auto G3 = NetworKit::ErdosRenyiGenerator(10, 0.15, false).generate();
	G2.append(G3);
	for (int i = 0; i < 20; i++)
	{
		auto v = G2.randomNode();
		auto w = G2.randomNode();
		if (v != w)
		{
			G2.addEdge(v, w);
		}
	}

	Graph G = Graph();
	G.addNodes(4);
	G.addEdge(0, 1);
	G.addEdge(0, 2);
	G.addEdge(1, 2);
	G.addEdge(1, 3);
	G.addEdge(2, 3);
	std::cout << G.toString();

	auto resistances = approxLaplacianPseudoinverseDiagonal(G);
	std::cout << "Resistances: ";
	for (auto const &r : resistances)
		std::cout << r << ' ';
}
