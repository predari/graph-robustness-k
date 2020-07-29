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
#include <exception>

#include <cassert>
#include <cstdlib>
#include <cmath>

#include <Eigen/Dense>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
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


using namespace NetworKit;



char* getCmdOption(int count, char ** begin, const std::string & option)
{
	char** end = begin + count;
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(int count, char** begin, const std::string& option)
{
    return std::find(begin, begin+count, option) != begin+count;
}



void testLaplacian(bool verbose)
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

	/*
	auto diagonal = approxLaplacianPseudoinverseDiagonal(G);
	assert(std::abs(diagonal[0] - 0.3125) < 0.05);
	assert(std::abs(diagonal[1] - 0.1875) < 0.05);
	assert(std::abs(diagonal[2] - 0.1876) < 0.05);
	assert(std::abs(diagonal[3] - 0.3125) < 0.05);
	*/

	// Test Laplacian pinv update formulas
	auto Lmat = laplacianMatrix(G);
	auto lpinv = laplacianPseudoinverse(G);
	auto diffvec = laplacianPseudoinverseColumnDifference(lpinv, 0, 3, 0);
	if (verbose) {
		std::cout << "Laplacian: \n" << Lmat << "\n";
		std::cout << "Laplacian pinv: \n" << lpinv << "\n";
	}
	std::vector<double> expected{-0.125, 0.0, 0.0, 0.125};
	for (int i = 0; i < 4; i++)
	{
		assert(std::abs(diffvec(i) - expected[i]) < 0.001);
	}

	if (verbose) {
		std::cout << "Laplacian trace difference upon adding (0,3): " << laplacianPseudoinverseTraceDifference(lpinv, 0, 3) << "\n";
		std::cout << "Laplacian trace difference upon removing (0,3) and adding (0,3): " << 
			laplacianPseudoinverseTraceDifference(lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 3)}, {1.0, -1.0}) << "\n";
		std::cout << "Laplacian trace difference upon removing (0,3) and adding (0,1): " << 
			laplacianPseudoinverseTraceDifference(lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 1)}, {1.0, -1.0}) << "\n";
	}
	assert(std::abs(laplacianPseudoinverseTraceDifference(lpinv, 0, 3) + 0.25) < 0.001);
	assert(std::abs(laplacianPseudoinverseTraceDifference(lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 3)}, {1.0, -1.0})) < 0.001);
	assert(std::abs(laplacianPseudoinverseTraceDifference(lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 1)}, {1.0, -1.0})) < 0.001);
	//assert(std::abs(laplacianPseudoinverseTraceDifference(col0, 0, col3, 3) + 0.25) < 0.001);

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
			if (u != v && !G.hasEdge(u,v) && es.count(std::pair<unsigned int, unsigned int>(u, v)) == 0) {
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
	omp_set_num_threads(1);

	if (argc < 2) {
		std::cout << "Error: Call without arguments. Use --help for help.\n";
		return 1;
	}

	bool run_tests = false;
	bool run_experiments = true;

	bool run_random = false;
	bool run_random_avg = false;
	bool run_submodular = false;
	bool run_submodular2 = false;
	bool run_stochastic = false;
	bool run_simulated_annealing = false;
	bool run_combined = false;

	bool heuristic_0 = false;
	bool heuristic_1 = false;
	bool heuristic_2 = false;

	bool verbose = false;
	bool vv = false;

	bool override_k = false;
	int k_override;
	double roundFactor = 1.0;
	int seed = 1;

	std::string helpstring = "EXAMPLE CALL\n"
    	"\trobustness -a1 -a2 -i graph.gml\n"
		"OPTIONS:\n" 
		"\t-v --verbose\n\t\tAlso output edge lists\n"
		"\t--override-k n\n\t\tOverride the k value preset of the instance\n\n"
		"\t-a0\t\tRandom Edges\n"
		"\t-a1\t\tSubmodular Greedy\n"
		"\t-a2\t\tStochastic submodular greedy\n"
		"\t-a3\t\tSimulated Annealing\n"
		"\n";


	if (argv == 0) { std::cout << "Error!"; return 1; }
	if (argc < 3) { std::cout << helpstring; return 1; }

	// Return next arg and increment i
	auto nextArg = [&](int &i) {
		if (i+1 >= argc || argv[i+1] == 0) {
			std::cout << "Error!";
			throw std::exception();
		}
		i++;
		return argv[i];
	}

	struct Instance {
		Graph g;
		int k;
		std::string name = "";
		std::string graphParamDescription = "";
		std::string description = "";
	};
	std::vector<Instance> instances;


	for (int i = 1; i < argc; i++) {
		if (argv[i] == 0) { std::cout << "Error!"; return 1; }
		std::string arg = argv[i];

		if (arg == "-h" || arg == "--help") {
			std::cout << helpstring;
			return 0;
		}

		if (arg == "-v" || arg == "--verbose") {
			verbose = true;
			continue;
		}
		if (arg == "-vv" || arg == "--very-verbose") {
			verbose = true;
			vv = true;
			continue;
		}

		if (arg == "-k" || arg == "--override-k" || arg == "--set-k") {
			std::string k_string = nextArg(i);
			k_override = std::atoi(k_string);
			if (k_override == 0) { std::cout << "Error: Bad argument to --set-k!"; return 1; }
			override_k = true;
			continue;
		}

		if (arg == "--round-factor" || arg == "-r") {
			std::string rf_string = nextArg(i);
			roundFactor = std::tod(rf_string);
			continue;
		}
		if (arg == "-a1" || arg == "--submodular-greedy") { run_submodular = true; continue; }
		if (arg == "-a11") { run_submodular2 = true; continue; }
		if (arg == "-a2" || arg == "--stochastic-submodular-greedy") { run_stochastic = true; continue; }
		if (arg == "-a3" || arg == "--simulated-annealing") { run_simulated_annealing = true; continue; }
		if (arg == "-a0" || arg == "--random-avg") { run_random_avg = true; continue; }
		if (arg == "-a00" || arg == "--random") { run_random = true; continue; }
		if (arg == "-a4" || arg == "--combined") { run_combined = true; continue; }

		if (arg == "-h0") { heuristic_0 = true; continue; }
		if (arg == "-h1") { heuristic_1 = true; continue; }
		if (arg == "-h2") { heuristic_2 = true; continue; }
		if (arg == "-t") { run_tests = true; run_experiments = false; continue; }


		if (arg == "-i" || arg == "--instance") {
			Instance inst;
			std::string filename = nextArg(i);
			std::string prefix = "../instances/";
			inst.name = prefix + filename;

			auto hasEnding = [&] (std::string const &fullString, std::string const &ending) {
				if (fullString.length() >= ending.length()) {
					return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
				} else {
					return false;
				}
			};
			if (hasEnding(filename, ".gml")) {
				GMLGraphReader reader;
				auto g = reader.read(filename);
				inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
			} else if (hasEnding(filename, ".edges")) {
				NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
				auto g = reader.read(filename);
				inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
			}
			instances.push_back(inst);
			continue;
		}
	}


	if (!heuristic_1 && !heuristic_2) {
		heuristic_0 = true;
	}


	// TODO: move this to simexpal.
	auto addErdosRenyiInstance = [&](int n, int k, double p) {
		Instance inst;
		Aux::Random::setSeed(seed, true);
		inst.g = NetworKit::ErdosRenyiGenerator(n, p, false).generate();
		inst.k = k;
		inst.name = "ErdosRenyiGraph";
		std::stringstream s;
		s << "p: " << p;
		inst.graphParamDescription = s.str();
		instances.push_back(inst);
	};

	if (cmdOptionExists(argc, argv, "-ger0")) {
		addErdosRenyiInstance(10, 10, 0.4);
	}
	if (cmdOptionExists(argc, argv, "-ger1")) {
		addErdosRenyiInstance(30, 100, 0.3);
	}
	if (cmdOptionExists(argc, argv, "-ger2")) {
		addErdosRenyiInstance(100, 1000, 0.1);
	}
	if (cmdOptionExists(argc, argv, "-ger3")) {
		addErdosRenyiInstance(128, 500, 0.1);
	}
	if (cmdOptionExists(argc, argv, "-ger4")) {
		addErdosRenyiInstance(300, 5000, 0.05);
	}
	if (cmdOptionExists(argc, argv, "-ger5")) {
		addErdosRenyiInstance(600, 300, 0.05);
	}
	if (cmdOptionExists(argc, argv, "-ger6")) {
		addErdosRenyiInstance(1000, 5000, 0.02);
	}
	if (cmdOptionExists(argc, argv, "-ger7")) {
		addErdosRenyiInstance(3000, 5000, 0.01);
	}

	auto addWattsStrogatzInstance = [&](int nodes, int neighbors, double p, int k) {
		Instance inst;
		Aux::Random::setSeed(seed, true);
		inst.g = NetworKit::WattsStrogatzGenerator(nodes, neighbors, p).generate();
		inst.k = k;
		inst.name = "WattsStrogatzGraph";
		std::stringstream s;
		s << "neighbors: " << neighbors << ", p: " << p;
		inst.graphParamDescription = s.str();
		instances.push_back(inst);
	};
	if (cmdOptionExists(argc, argv, "-gws0")) {
		addWattsStrogatzInstance(10, 3, 0.4, 5);
	}
	if (cmdOptionExists(argc, argv, "-gws1")) {
		addWattsStrogatzInstance(30, 5, 0.4, 10);
	}
	if (cmdOptionExists(argc, argv, "-gws2")) {
		addWattsStrogatzInstance(100, 5, 0.5, 100);
	}
	if (cmdOptionExists(argc, argv, "-gws3")) {
		addWattsStrogatzInstance(300, 7, 0.5, 1000);
	}
	if (cmdOptionExists(argc, argv, "-gws4")) {
		addWattsStrogatzInstance(1000, 7, 0.3, 1000);
	}

	auto addBarabasiAlbertInstance = [&](int n_attachments, int n_max, int n_0, int k) {
		Instance inst;
		Aux::Random::setSeed(seed, true);
		inst.g = NetworKit::BarabasiAlbertGenerator(n_attachments, n_max, n_0).generate();
		inst.k = k;
		inst.name = "BarabasiAlbertGraph";
		std::stringstream s;
		s << "n_attachments: " << n_attachments << ", n_max: " << n_max << ", n_0: " << n_0;
		inst.graphParamDescription = s.str();
		instances.push_back(inst);
	};
	if (cmdOptionExists(argc, argv, "-gba0")) {
		addBarabasiAlbertInstance(2, 128, 2, 10);
	}
	if (cmdOptionExists(argc, argv, "-gba1")) {
		addBarabasiAlbertInstance(2, 1000, 2, 200);
	}


	if (cmdOptionExists(argc, argv, "-gpwr0")) {
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		Instance inst;
		auto g = reader.read("../instances/opsahl-powergrid/out.opsahl-powergrid");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 10;
		inst.name = "US Powergrid";
		inst.description = "This undirected network contains information about the power grid of the Western States of the United States of America. An edge represents a power supply line. A node is either a generator, a transformator or a substation.\nhttp://konect.uni-koblenz.de/networks/opsahl-powergrid";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gpwr1")) {
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		Instance inst;
		auto g = reader.read("../instances/opsahl-powergrid/out.opsahl-powergrid");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 2000;
		inst.name = "US Powergrid";
		inst.description = "This undirected network contains information about the power grid of the Western States of the United States of America. An edge represents a power supply line. A node is either a generator, a transformator or a substation.\nhttp://konect.uni-koblenz.de/networks/opsahl-powergrid";
		instances.push_back(inst);
	}

	if (cmdOptionExists(argc, argv, "-gi0")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../instances/internet-topology-zoo/AsnetAm.gml");
		inst.k = 10;
		inst.name = "ASNET-AM";
		inst.description = "Armenia Backbone, Customer IP network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gi1")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../instances/internet-topology-zoo/Bellsouth.gml");
		inst.k = 10;
		inst.name = "Bell South";
		inst.description = "USA south east Backbone, Customer, Transit IP network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gi2")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../instances/internet-topology-zoo/Deltacom.gml");
		inst.k = 10;
		inst.name = "ITC Deltacom";
		inst.description = "USA south east Backbone, Transit fibre network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gi3")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../instances/internet-topology-zoo/Ion.gml");
		inst.k = 10;
		inst.name = "ION";
		inst.description = "USA NY region Backbone, Customer, Transit fibre network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gi4")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../instances/internet-topology-zoo/UsCarrier.gml");
		inst.k = 10;
		inst.name = "US Carrier";
		inst.description = "USA south east backbone, customer fibre network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gi5")) {
		Instance inst;
		NetworKit::GMLGraphReader reader;
		inst.g = reader.read("../instances/internet-topology-zoo/Dfn.gml");
		inst.k = 10;
		inst.name = "DFN";
		inst.description = "Germany DFN Backbone, Testbed IP network topology. \nhttp://www.topology-zoo.org/dataset.html";
		instances.push_back(inst);
	}



	if (cmdOptionExists(argc, argv, "-gfb1")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		auto g = reader.read("../instances/facebook/3980.edges");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 10;
		inst.name = "Facebook Ego 3980";
		inst.description = "https://snap.stanford.edu/data/egonets-Facebook.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gfb2")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		auto g = reader.read("../instances/facebook/686.edges");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 10;
		inst.name = "Facebook Ego 686";
		inst.description = "https://snap.stanford.edu/data/egonets-Facebook.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gfb3")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(1), "%", true, false);
		auto g = reader.read("../instances/facebook/3437.edges");
		inst.g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		inst.k = 10;
		inst.name = "Facebook Ego 3437";
		inst.description = "https://snap.stanford.edu/data/egonets-Facebook.html";
		instances.push_back(inst);
	}
	if (cmdOptionExists(argc, argv, "-gfb0")) {
		Instance inst;
		NetworKit::EdgeListReader reader (' ', NetworKit::node(0), "%", true, false);
		inst.g = reader.read("../instances/facebook/facebook_combined.txt");
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
			if (override_k) {
				k = k_override;
			}

			std::cout << "- Instance: \n";
			std::cout << "\tName: " << inst.name << "\n";
			std::cout << "\tNodes: " << n << "\n";
			std::cout << "\tEdges: " << g.numberOfEdges() << "\n";
			std::cout << "\tk: " << k << "\n";
			if (verbose) {
				std::cout << "\tEdgeList: [";
				g.forEdges([](NetworKit::node u, NetworKit::node v) { std::cout << "(" << u << ", " << v << "), "; });
				std::cout << "]\n" << std::endl;

			}
			std::cout << "\tRuns: \n";


			NetworKit::ConnectedComponents comp {g};
			comp.run();
			if (comp.numberOfComponents() != 1) {
				std::cout << "Error: Instance " << inst.name << " is not connected!\n";
				return 1;
			}

			auto write_result = [&](std::string name, double value, std::chrono::nanoseconds duration, std::vector<NetworKit::Edge> edges, std::string variant_name="") {
				std::cout << "\t\t- Algorithm:\t" << name << "\n";
				if (variant_name != "") {
					std::cout << "\t\t  Variant:\t" << variant_name << "\n";
				}
				std::cout << "\t\t  Value:\t" << value << "\n";
				using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
				std::cout << "\t\t  Time:\t\t" << std::chrono::duration_cast<scnds>(duration).count() << "\n";
				if (verbose) {
					std::cout << "\t\t  EdgeList:\t[";
					for (auto e: edges) { std::cout << "(" << e.u << ", " << e.v << "), "; }
					std::cout << "]\n" << std::endl;
				}
			};

			if (run_random) {
				Aux::Random::setSeed(seed, true);
				auto t1 = std::chrono::high_resolution_clock::now();
				auto edges = randomEdges(g, k);
				auto t2 = std::chrono::high_resolution_clock::now();

				auto G_copy = g;
				for (auto e : edges) {
					G_copy.addEdge(e.u, e.v);
				}
				double resistance = G_copy.numberOfNodes() * laplacianPseudoinverse(G_copy).trace();

				write_result("Random Edges", resistance, t2 - t1, edges, "");
			}

			if (run_random_avg) {
				Aux::Random::setSeed(seed, true);
				double resistance = 0.0;
				std::chrono::nanoseconds duration;
				int rnds = 10;
				for (int i = 0; i < rnds; i++) {
					auto t1 = std::chrono::high_resolution_clock::now();
					auto edges = randomEdges(g, k);
					auto t2 = std::chrono::high_resolution_clock::now();

					auto G_copy = g;
					for (auto e : edges) {
						G_copy.addEdge(e.u, e.v);
					}
					duration += (t2 - t1);
					resistance += G_copy.numberOfNodes() * laplacianPseudoinverse(G_copy).trace();
				}
				write_result("Random Edges averaged", resistance / rnds, duration / rnds, {}, "");
			}


			if (run_submodular) {
				Aux::Random::setSeed(seed, true);
				RobustnessGreedy rg;
				auto t1 = std::chrono::high_resolution_clock::now();
				rg.init(g, k);
				rg.addAllEdges();
				rg.run();
				auto t2 = std::chrono::high_resolution_clock::now();
				write_result("Submodular Greedy", rg.getResultResistance(), t2 - t1, rg.getResultEdges(), "");
			}
			if (run_submodular2) {
				Aux::Random::setSeed(seed, true);
				RobustnessGreedy2 rg;
				auto t1 = std::chrono::high_resolution_clock::now();
				rg.init(g, k);
				rg.addAllEdges();
				rg.run();
				auto t2 = std::chrono::high_resolution_clock::now();
				write_result("Submodular Greedy", rg.getResultResistance(), t2 - t1, rg.getResultEdges(), "Lpinv Updates On Demand");
			}
			if (run_stochastic) {
				Aux::Random::setSeed(seed, true);
				RobustnessStochasticGreedy rs;
				auto t1 = std::chrono::high_resolution_clock::now();
				rs.init(g, k, 0.5);
				rs.addAllEdges();
				rs.run();
				auto t2 = std::chrono::high_resolution_clock::now();
				write_result("Submodular Greedy", rs.getResultResistance(), t2 - t1, rs.getResultEdges(), "");
			}


			if (run_simulated_annealing || run_combined) {
				auto run_and_analyze = [&](RobustnessSimulatedAnnealingBase& rsa, int variant, bool combined) {
					Aux::Random::setSeed(seed, true);
					auto t1 = std::chrono::high_resolution_clock::now();
					int variation;
		
					State s;
					if (combined) {
						RobustnessGreedy rg;
						rg.init(g, k);
						rg.addAllEdges();
						rg.run();
						s.edges = rg.getResultItems();
					} else {
						s.edges = randomEdges(g, k);
					}
					rsa.init(g, k, roundFactor);
					rsa.setInitialState(s);

					rsa.run();
					auto t2 = std::chrono::high_resolution_clock::now();

					double value = rsa.getResultResistance();

					std::vector<std::string> variant_names = {"Random", "Resistance-Based", "Resistance-Based, Multiple"};

					std::string name = "Simulated Annealing";
					if (combined) { name = "Submodular Greedy + Simulated Annealing"; }
					write_result(name, value, t2 - t1, rsa.getResultEdges(), variant_names[variant]);
				};

				if (heuristic_0) {
					const int variant = 0;
					if (run_combined) {
						RobustnessSimulatedAnnealing<variant> rsa;
						run_and_analyze(rsa, variant, true);
					}
					if (run_simulated_annealing) {
						RobustnessSimulatedAnnealing<variant> rsa;
						run_and_analyze(rsa, variant, false);
					}
				}
				if (heuristic_1) {
					const int variant = 1;
					if (run_combined) {
						RobustnessSimulatedAnnealing<variant> rsa;
						run_and_analyze(rsa, variant, true);
					}
					if (run_simulated_annealing) {
						RobustnessSimulatedAnnealing<variant> rsa;
						run_and_analyze(rsa, variant, false);
					}
				}
				if (heuristic_2) {
					const int variant = 2;
					if (run_combined) {
						RobustnessSimulatedAnnealing<variant> rsa;
						run_and_analyze(rsa, variant, true);
					}
					if (run_simulated_annealing) {
						RobustnessSimulatedAnnealing<variant> rsa;
						run_and_analyze(rsa, variant, false);
					}
				}
			}
		}
	}

	if (run_tests) {
		testLaplacian(verbose);
	}

	//testRobustnessGreedy();
	//experiment();
	return 0;
}
