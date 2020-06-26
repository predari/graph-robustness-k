#include <iostream>
#include <iterator>
#include <optional>
#include <queue>
#include <vector>
#include <set>
#include <chrono>
#include <utility>

#include <cassert>
#include <cmath>

#include <Eigen/Dense>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/centrality/ApproxEffectiveResistance.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/auxiliary/Random.hpp>



#include <greedy.hpp>
#include <laplacian.hpp>
#include <robustnessGreedy.hpp>
#include <robustnessSimulatedAnnealing.hpp>



void testLaplacian()
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
	auto diffvec = laplacianPseudoinverseColumnDifference(col0, 0, col3, 3, 0);
	std::vector<double> exp{-0.125, 0.0, 0.0, 0.125};
	for (int i = 0; i < 4; i++)
	{
		assert(std::abs(diffvec[i] - exp[i]) < 0.001);
	}

	/*
	// Test Greedy
	GreedyTest greedyTest{10, 5};
	greedyTest.run();
	for (int i = 0; i < 5; i++)
	{
		assert(greedyTest.getResultItems()[i] == 9 - i);
	}
	assert(greedyTest.size() == 5);
	assert(greedyTest.getTotalValue() == 9 + 8 + 7 + 6 + 5);
	*/
}

void testRobustnessGreedy() {
	
	Graph G1;
	G1.addNodes(4);
	G1.addEdge(0, 1);
	G1.addEdge(0, 2);
	G1.addEdge(1, 2);
	G1.addEdge(1, 3);
	G1.addEdge(2, 3);
	
	Graph G2;
	G2.addNodes(6);
	std::vector<std::pair<unsigned long, unsigned long>> edges = {{0, 1}, {0,2}, {1,3}, {2,3}, {1,2}, {1,4}, {3,5}, {4,5}};
	for (auto p: edges) {
		G2.addEdge(p.first, p.second);
	}

	RobustnessGreedy rg2;
	rg2.init(G2, 2);
	rg2.run();
	assert(std::abs(rg2.getTotalValue() - 4.351) < 0.01);

	//rg2.summarize();
	

	Graph G3;
	G3.addNodes(14);
	for (size_t i = 0; i < 14; i++)
	{
		G3.addEdge(i, (i+1) % 14);
	}
	G3.addEdge(4, 13);
	G3.addEdge(5, 10);

	RobustnessGreedy rg3;
	rg3.init(G3, 4);
	rg3.run();
	assert(std::abs(rg3.getTotalValue() - 76.789) < 0.01);
	//rg3.summarize();

	//rgd3.summarize();
}

std::vector<NetworKit::Edge> randomEdges(NetworKit::Graph const & G, int n, int k) {
	std::vector<NetworKit::Edge> result;
	result.clear();
	std::set<std::pair<unsigned int, unsigned int>> es;
	for (int i = 0; i < k; i++) {
		do {
			int u = std::rand() % n;
			int v = std::rand() % n;
			if (u > v) {
				std::swap(u, v);
			}
			auto e = std::pair<unsigned int, unsigned int> (u, v);
			if (!G.hasEdge(u,v) && es.count(std::pair<unsigned int, unsigned int>(u, v)) == 0) {
				es.insert(e);
				result.push_back(NetworKit::Edge(u, v));
				break;
			}
		} while (true);
	}
	return result;
}

void experiment() {
    Aux::Random::setSeed(1, true);
	int n = 300;
	int k = 20;
	Graph smallworld = NetworKit::ErdosRenyiGenerator(n, 0.3, false).generate();
	smallworld.forEdges([](NetworKit::node u, NetworKit::node v) { std::cout << "(" << u << ", " << v << "), "; });
	std::cout << "\n";
	
	RobustnessGreedy rg;
	auto t1 = std::chrono::high_resolution_clock::now();
	rg.init(smallworld, k);
	rg.run();
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;
	rg.summarize();
	
    Aux::Random::setSeed(1, true);
	
	
	RobustnessStochasticGreedy rgd;
	auto t3 = std::chrono::high_resolution_clock::now();
	rgd.init(smallworld, k, 0.8);
	rgd.addAllEdges();
	rgd.run();
	auto t4 = std::chrono::high_resolution_clock::now();
	std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::seconds>(t4-t3).count() << std::endl;
	rgd.summarize();
	

	auto t5 = std::chrono::high_resolution_clock::now();
	RobustnessSimulatedAnnealing rsa;
	rsa.init(smallworld, k);
	rsa.setInitialTemperature();
	State s;
	s.edges = randomEdges(smallworld, n, k);
	//s.edges = rgd.getResultItems();
	rsa.setInitialState(s);
	rsa.setInitialTemperature();
	rsa.summarize();
	rsa.run();
	auto t6 = std::chrono::high_resolution_clock::now();
	rsa.summarize();
	std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::seconds>(t6-t5).count() << std::endl;
}

int main()
{
	//testLaplacian();
	//testRobustnessGreedy();
	experiment();
}
