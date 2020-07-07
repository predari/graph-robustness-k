#include <chrono>
#include <iostream>
#include <iterator>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include <cassert>
#include <cstdlib>
#include <cmath>

#include <Eigen/Dense>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/centrality/ApproxEffectiveResistance.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/BarabasiAlbertGenerator.hpp>
#include <networkit/generators/WattsStrogatzGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/io/EdgeListReader.hpp>
#include <networkit/io/GMLGraphReader.hpp>



#include <greedy.hpp>
#include <laplacian.hpp>
#include <robustnessGreedy.hpp>
#include <robustnessSimulatedAnnealing.hpp>



#include <algorithm>



char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}



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

std::vector<NetworKit::Edge> randomEdges(NetworKit::Graph const & G, int k) {
	std::vector<NetworKit::Edge> result;
	result.clear();
	std::set<std::pair<unsigned int, unsigned int>> es;
	int n = G.numberOfNodes();
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
    using deci = std::chrono::duration<int, std::ratio<1, 10>>;
    Aux::Random::setSeed(1, true);
	int n = 100;
	int k = 1000;
	double p = 0.05;
	Graph smallworld = NetworKit::ErdosRenyiGenerator(n, p, false).generate();
	//smallworld.forEdges([](NetworKit::node u, NetworKit::node v) { std::cout << "(" << u << ", " << v << "), "; });
	std::cout << "Nodes: " << n << ", Edges: " << smallworld.numberOfEdges() << "\n";
	
	RobustnessGreedy rg;
	auto t1 = std::chrono::high_resolution_clock::now();
	rg.init(smallworld, k);
	rg.addAllEdges();
	rg.run();
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Duration: " << std::chrono::duration_cast<deci>(t2-t1).count() << std::endl;
	rg.summarize();
	
    Aux::Random::setSeed(1, true);
	
	
	RobustnessStochasticGreedy rgd;
	auto t3 = std::chrono::high_resolution_clock::now();
	rgd.init(smallworld, k, 0.8);
	rgd.addAllEdges();
	rgd.run();
	auto t4 = std::chrono::high_resolution_clock::now();
	std::cout << "Duration: " << std::chrono::duration_cast<deci>(t4-t3).count() << std::endl;
	rgd.summarize();
	


/*

	auto t5 = std::chrono::high_resolution_clock::now();
	State s;
	s.edges = randomEdges(smallworld, k);
	auto t6 = std::chrono::high_resolution_clock::now();

	auto t7 = std::chrono::high_resolution_clock::now();
	RobustnessSimulatedAnnealing rsa;
	rsa.init(smallworld, k);
	rsa.setInitialTemperature();
	//s.edges = rgd.getResultItems();
	rsa.setInitialState(s);
	rsa.setInitialTemperature();
	double randomVal = rsa.getTotalValue();
	rsa.summarize();
	rsa.run();
	auto t8 = std::chrono::high_resolution_clock::now();
	rsa.summarize();
	std::cout << "Duration: " << std::chrono::duration_cast<deci>(t6-t5).count() << std::endl;
*/


	auto g = smallworld;
	using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
	std::cout << "Nodes: " << g.numberOfNodes() << ", Edges: " << g.numberOfEdges() << ", k=" << k << std::endl;
	std::cout << "Submodular Greedy: \t\t\t\t" << (-1.0) * rg.getTotalValue() << ", \t\t"<< std::chrono::duration_cast<scnds>(t2-t1).count() << "s" << std::endl;
	std::cout << "Stochastic Submodular Greedy: \t\t\t" << (-1.0) * rgd.getTotalValue() << ", \t\t"<< std::chrono::duration_cast<scnds>(t4-t3).count() << "s" << std::endl;
	//std::cout << "Random Edges: \t\t\t\t\t" << randomVal << ", \t\t"<< std::chrono::duration_cast<scnds>(t6-t5).count() << "s" << std::endl;
	//std::cout << "Simulated Annealing: \t\t\t\t" << rsa.getTotalValue() << ", \t\t"<< std::chrono::duration_cast<scnds>(t8-t7).count() << "s" << std::endl;

}

int main(int argc, char* argv[])
{
	using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
	//omp_set_num_threads(1);

	if (argc < 2) {
		std::cout << "Error: Call without arguments. Use --help for help.\n";
		return 1;
	}

	bool run_tests = false;
	bool run_submodular = false;
	bool run_stochastic = false;
	bool run_random = false;

	bool verbose = false;
	bool print_values_only = false;

	bool override_k = false;
	int k_override;

	if (cmdOptionExists(argv, argv+argc, "-h") || cmdOptionExists(argv, argv+argc, "--help")) {
		std::cout << 
		"EXAMPLE CALL\n"
    	"\trobustness -a1 -a2 -g1 -g2\n"
		"OPTIONS:\n" 
		"\t-v --verbose\n\n"
		"\t--values-only\n\t\tWrite only total effective resistance values to stdout\n\n"
		"\t--override-k n\n\t\tOverride the k value preset of the instance\n\n"
		"\t-a0\t\tRandom Edges\n"
		"\t-a1\t\tSubmodular Greedy\n"
		"\t-a2\t\tStochastic submodular greedy\n"
		"\t-a3\t\tSimulated Annealing\n"
		"\n"
		"\t-ger0\t\tErdos Renyi instance 1\n"
		"\t-ger1\t\tErdos Renyi instance 2\n"
		"\t-ger2\t\tErdos Renyi instance 3\n"
		"\t-ger3\t\tErdos Renyi instance 4\n";
	}

	if (cmdOptionExists(argv, argv+argc, "-v") || cmdOptionExists(argv, argv+argc, "--verbose")) {
		verbose = true;
	}
	if (cmdOptionExists(argv, argv+argc, "--values-only")) {
		print_values_only = true;
	}
	if (cmdOptionExists(argv, argv+argc, "--override-k")) {
		auto k_string = getCmdOption(argv, argv+argc, "--override-k");
		k_override = std::atoi(k_string);
		if (k_override == 0) {
			std::cout << "Error: Bad argument to --override-k" << '\n';
			return 1;
		}
		override_k = true;
		std::cout << "Overriding k: " << k_override << std::endl;
	}
	if (cmdOptionExists(argv, argv+argc, "-a1") || cmdOptionExists(argv, argv+argc, "--submodular-greedy")) {
		run_submodular = true;
	}
	if (cmdOptionExists(argv, argv+argc, "-a2") || cmdOptionExists(argv, argv+argc, "--stochastic-submodular-greedy")) {
		run_stochastic = true;
	}
	if (cmdOptionExists(argv, argv+argc, "-a0") || cmdOptionExists(argv, argv+argc, "--random")) {
		run_random = true;
	}

	bool run_experiments = true; //run_submodular || run_stochastic || run_random;



	struct Instance {
		Graph g;
		int k;
		std::string name = "";
		std::string graphParamDescription = "";
		std::string description = "";
	};
	std::vector<Instance> instances;

	auto addErdosRenyiInstance = [&](int n, int k, double p) {
		Instance inst;
		Aux::Random::setSeed(1, true);
		inst.g = NetworKit::ErdosRenyiGenerator(n, p, false).generate();
		inst.k = k;
		inst.name = "ErdosRenyiGraph";
		std::stringstream s;
		s << "p: " << p;
		inst.graphParamDescription = s.str();
		instances.push_back(inst);
	};

	if (cmdOptionExists(argv, argv+argc, "-ger0")) {
		addErdosRenyiInstance(10, 10, 0.4);
	}
	if (cmdOptionExists(argv, argv+argc, "-ger1")) {
		addErdosRenyiInstance(30, 100, 0.2);
	}
	if (cmdOptionExists(argv, argv+argc, "-ger2")) {
		addErdosRenyiInstance(100, 1000, 0.05);
	}
	if (cmdOptionExists(argv, argv+argc, "-ger3")) {
		addErdosRenyiInstance(128, 500, 0.03);
	}
	if (cmdOptionExists(argv, argv+argc, "-ger3")) {
		addErdosRenyiInstance(300, 5000, 0.05);
	}

	auto addWattsStrogatzInstance = [&](int nodes, int neighbors, double p, int k) {
		Instance inst;
		Aux::Random::setSeed(1, true);
		inst.g = NetworKit::WattsStrogatzGenerator(nodes, neighbors, p).generate();
		inst.k = k;
		inst.name = "WattsStrogatzGraph";
		std::stringstream s;
		s << "neighbors: " << neighbors << ", p: " << p;
		inst.graphParamDescription = s.str();
		instances.push_back(inst);
	};
	if (cmdOptionExists(argv, argv+argc, "-gws0")) {
		addErdosRenyiInstance(10, 3, 0.4, 5);
	}
	if (cmdOptionExists(argv, argv+argc, "-gws1")) {
		addErdosRenyiInstance(30, 5, 0.4, 10);
	}
	if (cmdOptionExists(argv, argv+argc, "-gws2")) {
		addErdosRenyiInstance(100, 5, 0.5, 100);
	}
	if (cmdOptionExists(argv, argv+argc, "-gws3")) {
		addErdosRenyiInstance(300, 7, 0.5, 1000);
	}
	if (cmdOptionExists(argv, argv+argc, "-gws4")) {
		addErdosRenyiInstance(1000, 7, 0.3, 1000);
	}

	auto addBarabasiAlbertInstance = [&](int n_attachments, int n_max, int n_0, int k) {
		Instance inst;
		Aux::Random::setSeed(1, true);
		inst.g = NetworKit::BarabasiAlbertGenerator(n_attachments, n_max, n_0).generate();
		inst.k = k;
		inst.name = "BarabasiAlbertGraph";
		std::stringstream s;
		s << "n_attachments: " << n_attachments << ", n_max: " << n_max << ", n_0: " << n_0;
		inst.graphParamDescription = s.str();
		instances.push_back(inst);
	};
	if (cmdOptionExists(argv, argv+argc, "-gba0")) {
		addBarabasiAlbertInstance(2, 128, 2, 10);
	}


	if (cmdOptionExists(argv,argv+argc, "-gpwr0")) {
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		Instance inst;
		inst.g = reader.read("../example-graphs/opsahl-powergrid/out.opsahl-powergrid");
		inst.k = 10;
		inst.name = "US Powergrid";
		inst.description = "This undirected network contains information about the power grid of the Western States of the United States of America. An edge represents a power supply line. A node is either a generator, a transformator or a substation.\nhttp://konect.uni-koblenz.de/networks/opsahl-powergrid";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv,argv+argc, "-gpwr1")) {
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		Instance inst;
		inst.g = reader.read("../example-graphs/opsahl-powergrid/out.opsahl-powergrid");
		inst.k = 2000;
		inst.name = "US Powergrid";
		inst.description = "This undirected network contains information about the power grid of the Western States of the United States of America. An edge represents a power supply line. A node is either a generator, a transformator or a substation.\nhttp://konect.uni-koblenz.de/networks/opsahl-powergrid";
		instances.push_back(inst);
	}

	if (cmdOptionExists(argv, argv+argc, "-gi0")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../example-graphs/internet-topology-zoo/AsnetAm.gml");
		inst.k = 10;
		inst.name = "ASNET-AM";
		inst.description = "Armenia Backbone, Customer IP network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gi1")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../example-graphs/internet-topology-zoo/Bellsouth.gml");
		inst.k = 10;
		inst.name = "Bell South";
		inst.description = "USA south east Backbone, Customer, Transit IP network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gi2")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../example-graphs/internet-topology-zoo/Deltacom.gml");
		inst.k = 10;
		inst.name = "ITC Deltacom";
		inst.description = "USA south east Backbone, Transit fibre network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gi3")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../example-graphs/internet-topology-zoo/Ion.gml");
		inst.k = 10;
		inst.name = "ION";
		inst.description = "USA NY region Backbone, Customer, Transit fibre network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gi4")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../example-graphs/internet-topology-zoo/UsCarrier.gml");
		inst.k = 10;
		inst.name = "US Carrier";
		inst.description = "USA south east backbone, customer fibre network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gi5")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../example-graphs/internet-topology-zoo/Dfn.gml");
		inst.k = 10;
		inst.name = "DFN";
		inst.description = "Germany DFN Backbone, Testbed IP network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}



	if (cmdOptionExists(argv, argv+argc, "-gfb1")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		auto g = reader.read("../example-graphs/facebook/3980.edges");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 10;
		inst.name = "Facebook Ego 3980";
		inst.description = "https://snap.stanford.edu/data/egonets-Facebook.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gfb2")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		auto g = reader.read("../example-graphs/facebook/686.edges");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 10;
		inst.name = "Facebook Ego 686";
		inst.description = "https://snap.stanford.edu/data/egonets-Facebook.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gfb3")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		auto g = reader.read("../example-graphs/facebook/3437.edges");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 10;
		inst.name = "Facebook Ego 3437";
		inst.description = "https://snap.stanford.edu/data/egonets-Facebook.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argv, argv+argc, "-gfb0")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(0), "%", true, false);
		inst.g = reader.read("../example-graphs/facebook/facebook_combined.txt");
		//inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g);
		inst.k = 10;
		inst.name = "Facebook Ego Full";
		inst.description = "https://snap.stanford.edu/data/egonets-Facebook.html";
		instances.push_back(inst);
	}



	//std::cout << "Threads: " << omp_get_num_threads() << std::endl;




	if (run_experiments) {
		for (auto inst: instances) {
			auto & g = inst.g;
			auto k = inst.k;
			int n = g.numberOfNodes();
			if (!print_values_only)
				std::cout << inst.name << ". Nodes: " << n << ", Edges: " << g.numberOfEdges() << ". " << inst.graphParamDescription << "\n";
			if (verbose) {
				g.forEdges([](NetworKit::node u, NetworKit::node v) { std::cout << "(" << u << ", " << v << "), "; });
				std::cout << std::endl;
			}

			NetworKit::ConnectedComponents comp {g};
			comp.run();
			if (comp.numberOfComponents() != 1) {
				std::cout << "Error: Instance " << inst.name << " is not connected!";
				return 1;
			}

			if (run_submodular) {
				Aux::Random::setSeed(1, true);
				RobustnessGreedy rg;
				auto t1 = std::chrono::high_resolution_clock::now();
				rg.init(g, k);
				rg.addAllEdges();
				rg.run();
				auto t2 = std::chrono::high_resolution_clock::now();
				if (!verbose) {
					std::cout << "Submodular Greedy Result";
					if (!print_values_only)
						std::cout << ". Duration: " << std::chrono::duration_cast<scnds>(t2-t1).count();
					std::cout << ". Total Effective Resistence: " << (-1.0) * rg.getTotalValue() << std::endl;
				} else {
					rg.summarize();
				}
			}
			if (run_stochastic) {
				Aux::Random::setSeed(1, true);
				RobustnessStochasticGreedy rs;
				auto t3 = std::chrono::high_resolution_clock::now();
				rs.init(g, k, 0.8);
				rs.addAllEdges();
				rs.run();
				auto t4 = std::chrono::high_resolution_clock::now();
				if (!verbose) {
					std::cout << "Stochastic Greedy Result";
					if (!print_values_only)
						std::cout << ". Duration: " << std::chrono::duration_cast<scnds>(t4-t3).count();
					std::cout << ". Total Effective Resistence: " << (-1.0) * rs.getTotalValue() << std::endl;
				} else {
					rs.summarize();
				}
			}
			if (run_random) {
				Aux::Random::setSeed(1, true);
				auto t5 = std::chrono::high_resolution_clock::now();
				State s;
				s.edges = randomEdges(g, k);
				auto t6 = std::chrono::high_resolution_clock::now();

				// We use the simulated annealing class here only to compute the cost without too much effort.
				RobustnessSimulatedAnnealing rsa;
				rsa.init(g, k);
				rsa.setInitialState(s);
				if (!verbose) {
					std::cout << "Random Edges Result";
					if (!print_values_only)
						std::cout << ". Duration: " << std::chrono::duration_cast<scnds>(t6-t5).count();
					std::cout << ". Total Effective Resistence: " << rsa.getTotalValue() << std::endl;
				} else {
					for (auto& e: s.edges) {
						std::cout << "(" << e.u << ", " << e.v << "), ";
					}
				}				
			}
		}
	}

	//testLaplacian();
	//testRobustnessGreedy();
	//experiment();
	return 0;
}
