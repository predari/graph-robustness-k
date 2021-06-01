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
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/BarabasiAlbertGenerator.hpp>
#include <networkit/generators/WattsStrogatzGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/generators/HyperbolicGenerator.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/io/EdgeListReader.hpp>
#include <networkit/io/NetworkitBinaryReader.hpp>
#include <networkit/io/GMLGraphReader.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>

#include <networkit/centrality/ApproxElectricalCloseness.hpp>


#include <greedy.hpp>
#include <laplacian.hpp>
#include <robustnessGreedy.hpp>
#include <robustnessUSTGreedy.hpp>
//#include <robustnessSimulatedAnnealing.hpp>



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



void testLaplacian(int seed, bool verbose=false)
{
	// Create Example Graphs
	Aux::Random::setSeed(seed, false);

	// Test Laplacian pinv approximation
	Graph G = Graph();
	G.addNodes(4);
	G.addEdge(0, 1);
	G.addEdge(0, 2);
	G.addEdge(1, 2);
	G.addEdge(1, 3);
	G.addEdge(2, 3);

	
	auto diagonal = approxLaplacianPseudoinverseDiagonal(G);

	assert(std::abs(diagonal(0) - 0.3125) < 0.05);
	assert(std::abs(diagonal(1) - 0.1875) < 0.05);
	assert(std::abs(diagonal(2) - 0.1876) < 0.05);
	assert(std::abs(diagonal(3) - 0.3125) < 0.05);
	

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

	/*
	Graph largeG = NetworKit::BarabasiAlbertGenerator(10, 1000, 2).generate();
	int pivot;
	diagonal = approxLaplacianPseudoinverseDiagonal(largeG, 0.5);
	std::cout << diagonal(5);
	lpinv = laplacianPseudoinverse(largeG);
	std::cout << " " << lpinv(5, 5);
	*/
}

/*
void testSampleUSTWithEdge() {
	omp_set_num_threads(1);
	Graph G = Graph();
	G.addNodes(4);
	G.addEdge(0, 1);
	G.addEdge(0, 2);
	G.addEdge(1, 2);
	G.addEdge(1, 3);
	G.addEdge(2, 3);
	NetworKit::ApproxElectricalCloseness apx(G);
	apx.run();
	node a = 1;
	node b = 2;
	apx.testSampleUSTWithEdge(a, b);
}
*/

void testDynamicColumnApprox() {
    const double eps = 0.1;
    
    //for (int seed : {1, 2, 3}) {
		int seed = 1;
		count n = 30;
        Aux::Random::setSeed(seed, true);
		
        auto G = HyperbolicGenerator(n, 4, 3).generate();
        G = ConnectedComponents::extractLargestConnectedComponent(G, true);
		n = G.numberOfNodes();

        // Create a biconnected component with size 2.
        G.addNodes(2);
        G.addEdge(n - 1, n);
        G.addEdge(n, n + 1);
		
		/*
		Graph G = Graph();
		G.addNodes(4);
		G.addEdge(0, 1);
		G.addEdge(1, 2);
		G.addEdge(2, 3);
		*/
        ApproxElectricalCloseness apx(G);
        apx.run();
        auto diag = apx.getDiagonal();
        auto gt = apx.computeExactDiagonal(1e-12);
        G.forNodes([&](node u) { assert(std::abs(diag[u] - gt[u]) < eps); });
        assert(apx.scores().size() == G.numberOfNodes());
		n = G.numberOfNodes();

        G.forNodes([&] (node u) {
            auto col = apx.approxColumn(u);
            auto exactCol = apx.computeExactColumn(u);
            G.forNodes([&](node v) { 
				if (std::abs(col[v] - exactCol[v]) > 2*eps)
					std::cout << "C " << u << ", " << v << ": " << col[v] - exactCol[v] << std::endl;
            });
        });

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> distN(0, n - 1);

        node a; 
        node b;

		for (int i = 0; i < 20; i++) {
			do {
				a = distN(rng);
				b = distN(rng);
			} while (G.hasEdge(a, b) || a == b);

			G.addEdge(a, b);
			apx.edgeAdded(a, b);

			std::cout << "Edge added " << a << ", " << b << std::endl;

			diag = apx.getDiagonal();
			gt = apx.computeExactDiagonal(1e-12);
			G.forNodes([&](node u) { 
				if (std::abs(diag[u] - gt[u]) > eps) {
					std::cout << "Diagonal off: " << u << "err: " << diag[u] - gt[u] << " rnd: " << i << " seed: " << seed << std::endl;
				}
			});
			assert(apx.scores().size() == G.numberOfNodes());

			G.forNodes([&] (node u) {
				auto col = apx.approxColumn(u);
				auto exactCol = apx.computeExactColumn(u);
				G.forNodes([&](node v) { 
					if (std::abs(col[v] - exactCol[v]) > eps) {
						std::cout << "Column off: " << u << ", " << v << ": " << col[v] - exactCol[v] << " rnd: " << i << " seed:" << seed << std::endl;
					}
				});
			});
		}
    //}
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
	if (n*(n-1)/2 - G.numberOfEdges() < k) {
		throw std::logic_error("Graph does not allow requested number of edges.");
	}
	for (int i = 0; i < k; i++) {
		do {
			int u = std::rand() % n;
			int v = std::rand() % n;
			if (u > v) {
				std::swap(u, v);
			}
			auto e = std::pair<unsigned int, unsigned int> (u, v);
			if (u != v && !G.hasEdge(u,v) && !G.hasEdge(v,u) && es.count(std::pair<unsigned int, unsigned int>(u, v)) == 0) {
				es.insert(e);
				result.push_back(NetworKit::Edge(u, v));
				break;
			}
		} while (true);
	}
	return result;
}

int main(int argc, char* argv[])
{
	omp_set_num_threads(4);

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
	bool run_a5 = false; 
	bool run_a6 = false;

	bool heuristic_0 = false;
	bool heuristic_1 = false;
	bool heuristic_2 = false;

	bool verbose = false;
	bool vv = false;
	bool verify_result = false;

	double k_factor = 1.0;
	bool km_sqrt = false;
	bool km_linear = false;
	bool km_crt = false;

	double eps = 0.1;
	double eps2 = 0.1;

	Graph g;
	std::string instance_filename = "";
	int instance_first_node = 0;
	char instance_sep_char = ' ';
	bool instance_directed = false;
	std::string instance_comment_prefix = "%";

	double roundFactor = 1.0;
	int seed = 1;

	std::string helpstring = "EXAMPLE CALL\n"
    	"\trobustness -a1 -a2 -i graph.gml\n"
		"OPTIONS:\n" 
		"\t-i <path to instance>\n"
		"\t-v --verbose\n\t\tAlso output edge lists\n"
		"\t-km\t\t k is a function of n, possible values: linear, sqrt, crt, const"
		"\t-k 5.3\n\t multiplicative factor to influence k.\n\n"
		"\t-a0\t\tRandom Edges\n"
		"\t-a1\t\tSubmodular Greedy\n"
		"\t-a2\t\tStochastic submodular greedy\n"
		//"\t-a3\t\tSimulated Annealing\n"
		//"\t-a4\t\tHill Climbing\n"
		"\t-a5\t\tGreedy Sq\n"
		"\t-a6\t\tTree Greedy\n"
		"\t-tr\t\tTest Results for correctness (correct length, correct resistance value, no edges from original graph). \n"
		"\t-eps\n\tEpsilon value for approximative algorithms. Default: 0.1\n"
		"\t-j 4\t number of threads"
		"\n";


	if (argv == 0) { std::cout << "Error!"; return 1; }
	if (argc < 2) { std::cout << helpstring; return 1; }

	// Return next arg and increment i
	auto nextArg = [&](int &i) {
		if (i+1 >= argc || argv[i+1] == 0) {
			std::cout << "Error!";
			throw std::exception();
		}
		i++;
		return argv[i];
	};


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

		if (arg == "-tr") { verify_result = true; }

		if (arg == "-k" || arg == "--override-k" || arg == "--k-factor") {
			std::string k_string = nextArg(i);
			k_factor = std::stod(k_string);
			continue;
		}
		if (arg == "-km") {
			std::string km_string = nextArg(i);
			if (km_string == "sqrt") {
				km_sqrt = true;
			} else if (km_string == "linear") {
				km_linear = true;
			} else if (km_string == "crt") {
				km_crt = true;
			} else {
				std::cout << "Error: bad argument to -km. Possible Values: 'sqrt', 'linear', 'crt'. Default is const.\n";
				return 1;
			}
			continue;
		}

		if (arg == "-j") {
			auto arg_str = nextArg(i);
			omp_set_num_threads(atoi(arg_str));
		}

		if (arg == "--round-factor" || arg == "-r") {
			auto rf_string = nextArg(i);
			roundFactor = std::stod(rf_string);
			continue;
		}
		if (arg == "-eps" || arg == "--eps") {
			eps = std::stod(nextArg(i));
			continue;
		}
		if (arg == "-eps2" || arg == "--eps2") {
			eps2 = std::stod(nextArg(i));
			continue;
		}

		
		if (arg == "--seed") {
			auto s_string = nextArg(i);
			seed = atoi(s_string);
			continue;
		}
		if (arg == "-a1" || arg == "--submodular-greedy") { run_submodular = true; continue; }
		if (arg == "-a11") { run_submodular2 = true; continue; }
		if (arg == "-a2" || arg == "--stochastic-submodular-greedy") { run_stochastic = true; continue; }
		//if (arg == "-a3" || arg == "--simulated-annealing") { run_simulated_annealing = true; continue; }
		if (arg == "-a0" || arg == "--random-avg") { run_random_avg = true; continue; }
		if (arg == "-a00" || arg == "--random") { run_random = true; continue; }
		//if (arg == "-a4" || arg == "--combined") { run_combined = true; continue; }
		if (arg == "-a5") { run_a5 = true; continue; }
		if (arg == "-a6") { run_a6 = true; continue; }

		if (arg == "-h0") { heuristic_0 = true; continue; }
		if (arg == "-h1") { heuristic_1 = true; continue; }
		if (arg == "-h2") { heuristic_2 = true; continue; }
		if (arg == "-t") { run_tests = true; run_experiments = false; continue; }


		if (arg == "-i" || arg == "--instance") {
			instance_filename = nextArg(i);
			continue;
		}
		if (arg == "-in") {
			instance_first_node = atoi(nextArg(i));
		}
		if (arg == "-isep") {
			instance_sep_char = nextArg(i)[0];
		}
		if (arg == "--isepspace") {
			instance_sep_char = ' ';
			std::cout << "OY";
		}
		if (arg == "-id") {
			instance_directed = true;
		}
		if (arg == "-ic") {
			instance_comment_prefix = nextArg(i);
		}
		
	}


	if (!heuristic_1 && !heuristic_2) {
		heuristic_0 = true;
	}

	std::string call = "";
	for (int i = 0; i < argc; i++) {
		call += argv[i]; call += " ";
	}
	//std::cout << "  Call: " << call << "\n";


	// Read graph file

	auto hasEnding = [&] (std::string const &fullString, std::string const &ending) {
		if (fullString.length() >= ending.length()) {
			return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
		} else {
			return false;
		}
	};
	auto hasStart = [&] (std::string const &fullString, std::string const &start) {
		if (fullString.length() >= start.length()) {
			return (0 == fullString.compare (0, start.length(), start));
		} else {
			return false;
		}
	};
	if (instance_filename != "") {
		if (hasEnding(instance_filename, ".gml")) {
			GMLGraphReader reader;
			try { g = reader.read(instance_filename); }
			catch(const std::exception& e) { std::cout << "Failed to open or parse gml file " + instance_filename << '\n'; return 1; }
		} else if (hasEnding(instance_filename, "nkb")) {
			NetworkitBinaryReader reader;
			try {g = reader.read(instance_filename); }
			catch(const std::exception& e) { std::cout << "Failed to open or parse networkit binary file " << instance_filename << '\n'; return 1; }
		} else {
			try {
				NetworKit::EdgeListReader reader(instance_sep_char, NetworKit::node(instance_first_node), instance_comment_prefix, true, instance_directed);
				g = reader.read(instance_filename);
			}
			catch (const std::exception& e) { 
				std::cout << "Failed to open or parse edge list file " + instance_filename << '\n' << e.what() << "\n"; 
				return 1;
			}
		}
		g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(g, true);
		g.removeSelfLoops();
	}



	if (run_experiments) {
		int k;
		int n = g.numberOfNodes();
		if (km_sqrt) {
			k = std::sqrt(n) * k_factor;
		} else if (km_linear) {
			k = n * k_factor;
		} else if (km_crt) {
			k = std::pow(n, 0.33333) * k_factor;
		} else {
			k = k_factor;
		}
		k = std::min(k, (int)(n*(n-1)/2 - g.numberOfEdges()));
		k = std::max(k, 1);

		std::cout << "Runs: \n";


		NetworKit::ConnectedComponents comp {g};
		comp.run();
		if (comp.numberOfComponents() != 1) {
			std::cout << "Error: Instance " << instance_filename << " is not connected!\n";
			return 1;
		}

		auto write_result = [&](std::string name, double value, double original_value, std::chrono::nanoseconds duration, std::vector<NetworKit::Edge> edges, std::string variant_name="") {
			std::cout << "- Instance: '" << instance_filename << "'\n";
			std::cout << "  Nodes: " << n << "\n";
			std::cout << "  Edges: " << g.numberOfEdges() << "\n";
			std::cout << "  k: " << k << "\n";
			std::string call = "";
			for (int i = 0; i < argc; i++) {
				call += argv[i]; call += " ";
			}
			std::cout << "  Call: " << call << "\n";
			if (verbose) {
				std::cout << "  EdgeList: [";
				g.forEdges([](NetworKit::node u, NetworKit::node v) { std::cout << "(" << u << ", " << v << "), "; });
				std::cout << "]\n" << std::endl;
			}

			std::cout << "  Algorithm:  " << "'" << name << "'" << "\n";
			if (variant_name != "") {
				std::cout << "  Variant:  '" << variant_name << "'\n";
			}
			std::cout << "  Value:  " << value << "\n";
			std::cout << "  Original Value:  " << original_value << "\n";
			std::cout << "  Gain:  " << original_value - value << "\n";
			using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
			std::cout << "  Time:    " << std::chrono::duration_cast<scnds>(duration).count() << "\n";
			if (verbose) {
				std::cout << "  AddedEdgeList:  [";
				for (auto e: edges) { std::cout << "(" << e.u << ", " << e.v << "), "; }
				std::cout << "]\n" << std::endl;
			}
		};

		auto result_correct = [&](double gain, std::vector<NetworKit::Edge> edges, double epsilon=0.00001) {
			if (edges.size() != k) { 
				std::cout << "Error: Output size != k\n"; 
				return false;
			}
			for (auto e : edges) { 
				if (g.hasEdge(e.u, e.v)) { 
					std::cout << "Error: Edge from result already in original graph! (" << e.u << ", " << e.v << ")\n";
					return false;
				}
			}
			auto g_ = g;
			double v0 = static_cast<double>(g.numberOfNodes()) * laplacianPseudoinverse(g).trace();
			for (auto e: edges) { g_.addEdge(e.u, e.v); }
			double v = static_cast<double>(g_.numberOfNodes()) * laplacianPseudoinverse(g_).trace();
			if (std::abs(gain - std::abs(v0 - v))/std::abs(gain) > epsilon) {
				std::cout << "Error: Gain Test failed. Algorithm output: " << gain << ", computed: " << v0 - v<< "\n";
				return false;
			}
			return true;
		};

		if (run_random) {
			Aux::Random::setSeed(seed, true);
			auto t1 = std::chrono::high_resolution_clock::now();
			auto edges = randomEdges(g, k);
			auto t2 = std::chrono::high_resolution_clock::now();

			auto G_copy = g;
			double original_resistance = G_copy.numberOfNodes() * laplacianPseudoinverse(G_copy).trace();

			for (auto e : edges) {
				G_copy.addEdge(e.u, e.v);
			}
			double resistance = G_copy.numberOfNodes() * laplacianPseudoinverse(G_copy).trace();

			write_result("Random Edges", resistance, original_resistance, t2 - t1, edges, "");
			if (verify_result && !result_correct(original_resistance - resistance, edges)) { return 1; }
		}

		if (run_random_avg) {
			Aux::Random::setSeed(seed, true);
			double resistance = 0.0;
			std::chrono::nanoseconds duration;
			int rnds = 10;
			double original_resistance = g.numberOfNodes() * laplacianPseudoinverse(g).trace();

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
			write_result("Random Edges averaged", resistance / rnds, original_resistance, duration / rnds, {}, "");
		}


		if (run_submodular) {
			Aux::Random::setSeed(seed, true);
			RobustnessGreedy rg;
			auto t1 = std::chrono::high_resolution_clock::now();
			rg.init(g, k);
			rg.addAllEdges();
			rg.run();
			auto t2 = std::chrono::high_resolution_clock::now();

			auto resistance = rg.getResultResistance();
			auto edges = rg.getResultEdges();

			double original_resistance = rg.getOriginalResistance();

			write_result("Submodular Greedy", rg.getResultResistance(), original_resistance, t2 - t1, rg.getResultEdges(), "");
			if (verify_result && !result_correct(original_resistance - resistance, edges)) { return 1; }
		}
		if (run_a5) {
			Aux::Random::setSeed(seed, true);
			RobustnessSqGreedy rg;
			auto t1 = std::chrono::high_resolution_clock::now();
			rg.init(g, k);
			rg.run();
			auto t2 = std::chrono::high_resolution_clock::now();
			if (rg.isValidSolution()) {
				auto resistance = rg.getResultResistance();
				double original_resistance = rg.getOriginalResistance();
				auto edges = rg.getResultEdges();
				write_result("Greedy Sq", resistance, original_resistance, t2 - t1, edges, "");
				if (verify_result && !result_correct(original_resistance - resistance, edges)) { return 1; }
			} else {
				std::cout << "A5 failed!";
				return 1;
			}
		}

		if (run_a6) {
			Aux::Random::setSeed(seed, true);
			Graph g_cpy = g;
			RobustnessTreeGreedy rg(g_cpy, k, eps, eps2);
			auto t1 = std::chrono::high_resolution_clock::now();
			rg.init();
			rg.run();
			auto t2 = std::chrono::high_resolution_clock::now();
			if (rg.isValidSolution()) {
				auto resistance = rg.getResultResistance();
				auto edges = rg.getResultEdges();
				double original_resistance = rg.getOriginalResistance();
				write_result("UST Greedy", resistance, original_resistance, t2 - t1, edges, "");
				if (verify_result && !result_correct(original_resistance - resistance, edges)) { return 1; }
			} else {
				std::cout << "A6 failed!";
				return 1;
			}
		}
		if (run_submodular2) {
			Aux::Random::setSeed(seed, true);
			RobustnessGreedy2 rg;
			auto t1 = std::chrono::high_resolution_clock::now();
			rg.init(g, k);
			rg.addAllEdges();
			rg.run();
			auto t2 = std::chrono::high_resolution_clock::now();

			auto resistance = rg.getResultResistance();
			auto edges = rg.getResultEdges();
			double original_resistance = rg.getOriginalResistance();
			write_result("Submodular Greedy", resistance, original_resistance, t2 - t1, edges, "Lpinv Updates On Demand");
			if (verify_result && !result_correct(original_resistance - resistance, edges)) { return 1; }
		}
		if (run_stochastic) {
			Aux::Random::setSeed(seed, true);
			RobustnessStochasticGreedy rs;
			auto t1 = std::chrono::high_resolution_clock::now();
			rs.init(g, k, eps);
			rs.addAllEdges();
			rs.run();
			auto resistance = rs.getResultResistance();
			auto edges = rs.getResultEdges();
			auto t2 = std::chrono::high_resolution_clock::now();
			double original_resistance = rs.getOriginalResistance();
			write_result("Stochastic Submodular Greedy", resistance, original_resistance, t2 - t1, edges, "");
			if (verify_result && !result_correct(original_resistance - resistance, edges)) { return 1; }

		}

	}

	
	if (run_tests) {
		testDynamicColumnApprox();
	}
	

	//testRobustnessGreedy();
	//experiment(seed);
	return 0;
}
