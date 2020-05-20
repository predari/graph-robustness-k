#include <iostream>
#include <vector>
#include <iterator>
#include <optional>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>

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

std::vector<double> approxEffectiveResistances(Graph G, int &pivot)
{
	ApproxEffectiveResistance resAlg(G);
	resAlg.init();
	resAlg.numberOfUSTs = resAlg.computeNumberOfUSTs();
	resAlg.run();
	auto resistances = resAlg.getApproxEffectiveResistances();
	pivot = resAlg.getRoot();
	return resistances;
}

Vector computeLaplacianPseudoinverseColumn(CSRMatrix const &laplacian, int index, bool connected = false)
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
Vector laplacianPseudoinverseColummDifference(Vector column_i, int i, Vector column_j, int j, int k, double conductance = 1.0)
{
	double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
	double w = 1.0 / (1.0 / conductance + R_ij);
	Vector m_e = column_i - column_j;
	return m_e * m_e[k] * w * (-1.0);
}

// Update formula for the trace of the lap pinv as an edge is added to the graph.
// Add edge (i, j) and compute the difference of the traces of the pseudoinverses.
// Add this to the old to get the new.
double laplacianPseudoinverseTraceDifference(Vector column_i, int i, Vector column_j, int j, double conductance = 1.0)
{
	double R_ij = column_i[i] + column_j[j] - 2 * column_i[j];
	double w = 1.0 / (1.0 / conductance + R_ij);
	Vector m_e = column_i - column_j;
	return Vector::innerProduct(m_e, m_e) * w * (-1.0);
}

// Compute a stochastic approximation of the diagonal of the Moore-Penrose pseudoinverse of the laplacian matrix of a graph.
std::vector<double> approxLaplacianPseudoinverseDiagonal(Graph &G, double epsilon = 0.1)
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

template <class Item, class Scalar = double>
class Greedy : Algorithm
{
public:
	virtual std::vector<Item> collectItems() = 0;
	// Determine the marginal gain from adding the item. 
	virtual Scalar objectiveDifference(Item c) = 0;
	// Called after the item is added to the result.
	virtual void useItem(Item c) = 0;
	// Check wether the current solution is acceptable.
	virtual bool foundSolution() = 0;

	// Called immediately before the item is evaluated to check if it still should be considered.
	virtual bool isItemAcceptable(Item c) { return true; }

	// Select an item from the collection of items. Returns the item and its value if there is a suitable item, sets found to false otherwise.
	virtual void selection(std::vector<Item> items, Item &item, Scalar &value, bool found)
	{
		Scalar bestValue;
		Item bestItem;
		found = false;
		for (int i = 0; i < items.size(); i++)
		{
			auto c = items[i];
			if (!isItemAcceptable(c))
				continue;

			if (!found)
			{
				bestValue = objectiveDifference(c);
				bestItem = c;
				found = true;
			}
			else
			{
				auto val = objectiveDifference(c);
				if (bestValue < val)
				{
					bestValue = val;
					bestItem = c;
				}
			}
		}
		if (found)
		{
			value = bestValue;
			item = bestItem;
		} // Else only return found=false
	}

	// Number of items in the solution.
	int size() { return this->round; }
	// Return the solution items
	std::vector<Item> getResultItems() { return results; }
	Scalar getTotalValue() { return totalValue; }

	void run() override
	{
		round = 1;
		totalValue = 0;
		results.clear();
		while (true)
		{
			auto items = collectItems();
			Scalar value;
			bool haveValidItem;
			Item selectedItem;
			selection(items, selectedItem, value, haveValidItem);
			if (haveValidItem)
			{
				useItem(selectedItem);
				totalValue += value;
				results.push_back(selectedItem);
				if (foundSolution()) {
					this->isSolution = true;
					break;
				}
				round++;
			}
			else
			{
				break;
			}
		}
		hasRun = true;
	}

private:
	bool isSolution = false;
	int round;
	std::vector<Item> results;
	Scalar totalValue;
};




// Greedy algorithm test class. Finds the largest k numbers in {n-1, ..., 0}.
class GreedyTest : public Greedy<int>
{
public:
	GreedyTest(int n, int k)
	{
		for (int i = n - 1; i >= 0; i--)
		{
			_items.push_back(i);
		}
		numberOfElements = k;
	}
	std::vector<int> collectItems() override
	{
		return _items;
	}

	double objectiveDifference(int c) override
	{
		return c;
	}

	void useItem(int c) override
	{
		numbers.push_back(c);
		for (auto it = _items.begin(); it != _items.end();)
		{
			if (*it == c)
			{
				_items.erase(it);
			}
			else
			{
				it++;
			}
		}
	}

	bool foundSolution() override
	{
		return size() == numberOfElements;
	}

private:
	int numberOfElements;
	std::vector<int> _items;
	std::vector<int> numbers;
};

void test()
{
	// Create Example Graphs
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

	// Test Laplacian pinv approximation
	Graph G = Graph();
	G.addNodes(4);
	G.addEdge(0, 1);
	G.addEdge(0, 2);
	G.addEdge(1, 2);
	G.addEdge(1, 3);
	G.addEdge(2, 3);

	auto diagonal = approxLaplacianPseudoinverseDiagonal(G);
	assert(std::abs(diagonal[0] - 0.3125) < 0.05);
	assert(std::abs(diagonal[1] - 0.1875) < 0.05);
	assert(std::abs(diagonal[2] - 0.1876) < 0.05);
	assert(std::abs(diagonal[3] - 0.3125) < 0.05);

	// Test Laplacian pinv update formulas
	CSRMatrix Lmat = CSRMatrix::laplacianMatrix(G);
	auto col0 = computeLaplacianPseudoinverseColumn(Lmat, 0);
	auto col3 = computeLaplacianPseudoinverseColumn(Lmat, 3);
	assert(std::abs(laplacianPseudoinverseTraceDifference(col0, 0, col3, 3) + 0.25) < 0.001);
	auto diffvec = laplacianPseudoinverseColummDifference(col0, 0, col3, 3, 0);
	std::vector<double> exp{-0.125, 0.0, 0.0, 0.125};
	for (int i = 0; i < 4; i++)
	{
		assert(std::abs(diffvec[i] - exp[i]) < 0.001);
	}

	// Test Greedy
	GreedyTest greedyTest{10, 5};
	greedyTest.run();
	for (int i = 0; i < 5; i++)
	{
		assert(greedyTest.getResultItems()[i] == 9 - i);
	}
	assert(greedyTest.size() == 5);
	assert(greedyTest.getTotalValue() == 9 + 8 + 7 + 6 + 5);
}

int main()
{
	test();
}
