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
#include <dynamicLaplacianSolver.hpp>
#include <robustnessGreedy.hpp>
#include <robustnessUSTGreedy.hpp>

//#include <robustnessSimulatedAnnealing.hpp>



#include <algorithm>


using namespace NetworKit;


typedef decltype(std::chrono::high_resolution_clock::now()) Time;

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
	auto Lmat = laplacianMatrix<Eigen::MatrixXd>(G);
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

void testRobustnessSubmodularGreedy() {
	
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

	GreedyParams args(G2, 2);
	RobustnessSubmodularGreedy rg2(args);
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

	GreedyParams args2(G3, 4);
	RobustnessSubmodularGreedy rg3(args2);
	rg3.run();
	assert(std::abs(rg3.getTotalValue() - 76.789) < 0.01);
	//rg3.summarize();

	//rgd3.summarize();
}



void testRobustnessStochasticGreedySpectral() {
	
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

	GreedyParams args(G2, 2);
	RobustnessStochasticGreedySpectral rg2(args);
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

	GreedyParams args2(G3, 4);
        RobustnessStochasticGreedySpectral rg3(args2);
	rg3.run();
	assert(std::abs(rg3.getTotalValue() - 76.789) < 0.01);
	//rg3.summarize();

	//rgd3.summarize();
}



void testJLT(NetworKit::Graph g, std::string instanceFile, int k) {
	int n = g.numberOfNodes();

	JLTLUSolver jlt_solver;
	SparseLUSolver reference_solver;

	double epsilon = 0.5;
	jlt_solver.setup(g, epsilon, n);
	reference_solver.setup(g, 0.1);


	
    std::mt19937 gen(1);
    std::uniform_int_distribution<> distrib_n(0, n - 1);


	/*
	for (int i = 0 ; i < 5; i++) {
		node u = distrib(gen);
		node v = distrib(gen);
		if (!g.hasEdge(u, v)) {
			g.addEdge(u, v);
			jlt_solver.addEdge(u, v);
			reference_solver.addEdge(u, v);
		}
	}
	*/

	std::cout << "Runs: \n";
	std::cout << "- Instance: '" << instanceFile << "'\n";
	std::cout << "  Nodes: " << n << "\n";
	std::cout << "  Edges: " << g.numberOfEdges() << "\n";

	std::cout << "  JLT-Test: true \n";
	std::cout << "  Rel-Errors: [";

	for (int i = 0; i < 1000; i++) {
		node u = distrib_n(gen);
		node v = distrib_n(gen);
		if (!g.hasEdge(u, v) && u != v) {
			auto approx_gain = jlt_solver.totalResistanceDifferenceApprox(u, v);
			auto reference_gain = reference_solver.totalResistanceDifferenceExact(u, v);
			auto rel_error = std::abs(reference_gain - approx_gain) / reference_gain;
			std::cout << rel_error << ", ";
		}
	}
	std::cout << "] \n";


	std::cout << "  Rel-Errors-2: [";

	JLTLUSolver jlt_solver2;
	SparseLUSolver reference_solver2;


	epsilon = 0.75;
	int c = std::max(n / std::sqrt(2) / k * std::log(1. / 0.9), 3.);
	jlt_solver2.setup(g, epsilon, c);
    std::uniform_int_distribution<> distrib_c(0, c - 1);
	std::set<int> node_subset;
	while (node_subset.size() < c) {
		node_subset.insert(distrib_n(gen));
	}
	std::vector<NetworKit::node> node_subset_vec(node_subset.begin(), node_subset.end());
	assert(node_subset_vec.size() == c);

	for (int i = 0; i < 1000; i++) {
		node u = node_subset_vec[distrib_c(gen)];
		node v = node_subset_vec[distrib_c(gen)];
		if (!g.hasEdge(u, v) && u != v) {
			auto approx_gain = jlt_solver.totalResistanceDifferenceApprox(u, v);
			auto reference_gain = reference_solver.totalResistanceDifferenceExact(u, v);
			auto rel_error = std::abs(reference_gain - approx_gain) / reference_gain;
			std::cout << rel_error << ", ";
		}
	}

	std::cout << "] \n";
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

enum class LinAlgType {
	none,
	cg,
	least_squares,
	qr,
	ldlt, 
	lu,
	lamg,
	dense_ldlt,
	jlt_lu_sparse,
	jlt_lamg
};

enum class AlgorithmType {
	none,
	submodular,
	submodular2,
	stochastic,
	stochastic_spectral,
	stochastic_dyn,
	trees,
	random,
	random_avg,
	a5
};

class RobustnessExperiment {
public:
	NetworKit::count k;
	NetworKit::count n;
	double epsilon;
	double epsilon2;
	NetworKit::Graph g;
	NetworKit::Graph g_cpy;
	LinAlgType linalg;
	AlgorithmType alg;
	bool verify_result = false;
	bool verbose = false;
	bool vv = false;
	Time beforeInit;
	int seed;
	std::string name;
	unsigned int threads = 1;
	std::unique_ptr<GreedyParams> params;
	HeuristicType heuristic;
	bool always_use_known_columns_as_candidates = false;

	std::vector<NetworKit::Edge> edges;
	double resultResistance;
	double originalResistance;

	std::string algorithmName;
	std::string call;
	std::string instanceFile;

	std::unique_ptr<AbstractOptimizer<NetworKit::Edge>> greedy;


	void createGreedy() {
		if (alg == AlgorithmType::submodular) {
			algorithmName = "Submodular";
			createSpecific<RobustnessSubmodularGreedy>();
		} else if (alg == AlgorithmType::submodular2) {
			algorithmName = "Submodular 2";
			createSpecific<RobustnessSubmodularGreedy2>();
		} else if (alg == AlgorithmType::stochastic) {
			algorithmName = "Stochastic";
			createSpecific<RobustnessStochasticGreedy>();
		} else if (alg == AlgorithmType::a5) {
			algorithmName = "A5";
			createSpecific<RobustnessSqGreedy>();
		} else if (alg == AlgorithmType::trees) {
			algorithmName = "UST";
			createLinAlgGreedy<RobustnessTreeGreedy>();
		} else if (alg == AlgorithmType::random_avg) {
			algorithmName = "Random Averaged";
			createSpecific<RobustnessRandomAveraged<SparseLUSolver>>();
		} else if (alg == AlgorithmType::stochastic_dyn) {
			algorithmName = "Stochastic Dyn";
			createLinAlgGreedy<RobustnessStochasticGreedyDyn>();
		}
		else if(alg == AlgorithmType::stochastic_spectral) {
		        algorithmName = "Stochastic Spectral";
			createSpecific<RobustnessStochasticGreedySpectral>();
		} else {
			throw std::logic_error("Algorithm not implemented!");
		}
	}

	template <template <typename __Solver> class Greedy> 
	void createLinAlgGreedy() {
		if (linalg == LinAlgType::ldlt) {
			createSpecific<Greedy<SparseLDLTSolver>>();
		} else if (linalg == LinAlgType::least_squares) {
			createSpecific<Greedy<SparseLDLTSolver>>();
		} else if (linalg == LinAlgType::qr) {
			createSpecific<Greedy<SparseQRSolver>>();
		} else if (linalg == LinAlgType::lu) {
			createSpecific<Greedy<SparseLUSolver>>();
		} else if (linalg == LinAlgType::cg) {
			createSpecific<Greedy<SparseCGSolver>>();
		} else if (linalg == LinAlgType::dense_ldlt) {
			createSpecific<Greedy<DenseLDLTSolver>>();
		} else if (linalg == LinAlgType::lamg) {
			createSpecific<Greedy<LamgDynamicLaplacianSolver>>();
		} else if (linalg == LinAlgType::none) {
			createSpecific<Greedy<DenseCGSolver>>();
		} else if (linalg == LinAlgType::jlt_lu_sparse) {
			createSpecific<Greedy<JLTLUSolver>>();
		} else if (linalg == LinAlgType::jlt_lamg) {
			createSpecific<Greedy<JLTLamgSolver>>();
		} else {
			throw std::logic_error("Solver not implemented!");
		}
	}

	template <class Greedy> 
	void createSpecific() {
		auto grdy = new Greedy(*params);
		greedy = std::unique_ptr<Greedy>(grdy);
	}


	void run() {
		// Initialize
		n = g.numberOfNodes();
		omp_set_num_threads(threads);
		Aux::Random::setSeed(seed, true);
		g_cpy = g;


		if (alg == AlgorithmType::none) {
			throw std::logic_error("Not implemented!");
		}

		params = std::make_unique<GreedyParams>(g_cpy, k);
		params->epsilon = epsilon;
		params->epsilon2 = epsilon2;
		params->threads = threads;
		params->heuristic = heuristic;
		if (linalg == LinAlgType::jlt_lu_sparse || linalg == LinAlgType::jlt_lamg) {
			params->solverEpsilon = 0.75;
		}
		params->always_use_known_columns_as_candidates = this->always_use_known_columns_as_candidates;

		std::cout << "Runs: \n";
		std::cout << "- Instance: '" << instanceFile << "'\n";
		std::cout << "  Nodes: " << n << "\n";
		std::cout << "  Edges: " << g.numberOfEdges() << "\n";
		std::cout << "  k: " << k << "\n";
		std::cout << "  Call: " << call << "\n";
		std::cout << "  Threads:  " << threads << "\n";
		std::cout << "  All-Columns: ";
		if (params->always_use_known_columns_as_candidates) { std::cout << "True\n"; } else { std::cout << "False\n"; }


		if (alg == AlgorithmType::trees || alg == AlgorithmType::stochastic_dyn) {
			std::string linalgName = "";
			if (linalg == LinAlgType::cg) {
				linalgName = "CG";
			} else if (linalg == LinAlgType::qr) {
				linalgName = "QR";
			} else if (linalg == LinAlgType::ldlt) {
				linalgName = "LDLT";
			} else if (linalg == LinAlgType::lamg) {
				linalgName = "LAMG";
			} else if (linalg == LinAlgType::least_squares) {
				linalgName = "Least Squares";
			} else if (linalg == LinAlgType::lu) {
				linalgName = "LU";
			} else if (linalg == LinAlgType::dense_ldlt) {
				linalgName = "Dense LDLT";
			} else if (linalg == LinAlgType::none) {
				linalgName = "Dense CG";
			} else if (linalg == LinAlgType::jlt_lu_sparse) {
				linalgName = "JLT via Sparse LU";
			} else if (linalg == LinAlgType::jlt_lamg) {
				linalgName = "JLT via LAMG";
			}

			if (linalgName != "") {
				std::cout << "  Linalg: " << linalgName << "\n";
			}
		}

		if (alg == AlgorithmType::trees) {
			std::string heuristicName;
			if (heuristic == HeuristicType::lpinvDiag) {
				heuristicName = "Lpinv Diagonal";
			}
			if (heuristic == HeuristicType::similarity) {
				heuristicName = "Similarity";
			}
			if (heuristic == HeuristicType::random) {
				heuristicName = "Random";
			}
			std::cout << "  Heuristic: " << heuristicName << "\n";
			std::cout << "  Epsilon2: " << epsilon2 << "\n";
		}

		if (alg == AlgorithmType::a5 || alg == AlgorithmType::stochastic || alg == AlgorithmType::stochastic_dyn || alg == AlgorithmType::trees) {
			std::cout << "  Epsilon: " << epsilon << "\n";
		}

		beforeInit = std::chrono::high_resolution_clock::now();

		createGreedy();

		// Run greedy

		greedy->run();
		auto t = std::chrono::high_resolution_clock::now();
		auto duration = t - beforeInit;
		
		// Verify Results
		if (!greedy->isValidSolution()) {
			std::cout << name << " failed!\n";
			throw std::logic_error(std::string("Algorithm") + name + "failed!");
		}
		edges = greedy->getResultItems();
		resultResistance = greedy->getResultValue();
		originalResistance = greedy->getOriginalValue();		

		// Output Results
		std::cout << "  Algorithm:  " << "'" << algorithmName << "'" << "\n";

		if (vv) {
			std::cout << "  EdgeList: [";
			g.forEdges([](NetworKit::node u, NetworKit::node v) { std::cout << "(" << u << ", " << v << "), "; });
			std::cout << "]\n" << std::endl;
		}
		std::cout << "  Value:  " << resultResistance << "\n";
		std::cout << "  Original Value:  " << originalResistance << "\n";
		std::cout << "  Gain:  " << originalResistance - resultResistance << "\n";

		using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
		std::cout << "  Time:    " << std::chrono::duration_cast<scnds>(duration).count() << "\n";

		if (verbose) {
			std::cout << "  AddedEdgeList:  [";
			for (auto e: edges) { std::cout << "(" << e.u << ", " << e.v << "), "; }
			std::cout << "]\n" << std::endl;
		}
		std::cout.flush();

		if (verify_result) {
			if (edges.size() != k) {
				std::ostringstream stringStream;
  				stringStream << "Result Error: Output does contains " << edges.size() << " edges, not k = " << k << " edges!";
				std::string error_msg = stringStream.str();
				std::cout << error_msg << "\n";
				throw std::logic_error(error_msg);
			}

			LamgDynamicLaplacianSolver solver;
			double gain = originalResistance - resultResistance;
			double gain_exact = 0.;
			auto g_ = g;
			solver.setup(g_, 0.0001, 2);
			for (auto e : edges) {
				if (g.hasEdge(e.u, e.v)) {
					std::cout << "Error: Edge in result is already in original graph!\n";
					throw std::logic_error("Error: Edge from result already in original graph!");
				}

				gain_exact += solver.totalResistanceDifferenceExact(e.u, e.v);
				solver.addEdge(e.u, e.v);
				g_.addEdge(e.u, e.v);
			}

			if (std::abs((gain - gain_exact) / gain_exact / k) > 0.01) {
				std::ostringstream stringStream;
				stringStream << "Error: Gain Test failed. Output: " << gain << ", exact: " << gain_exact << ".";
				std::string error_msg = stringStream.str();
				std::cout << error_msg << "\n";
				throw std::logic_error(error_msg);
			}
		}
	}
};

int main(int argc, char* argv[])
{
	omp_set_num_threads(1);


	RobustnessExperiment experiment;

	for (int i = 0; i < argc; i++) {
		experiment.call += argv[i]; experiment.call += " ";
	}


	bool run_tests = false;
	bool run_experiments = true;

	bool verbose = false;

	double k_factor = 1.0;
	bool km_sqrt = false;
	bool km_linear = false;
	bool km_crt = false;

	bool test_jlt = false;

	LinAlgType linalg;

	Graph g;
	std::string instance_filename = "";
	int instance_first_node = 0;
	char instance_sep_char = ' ';
	bool instance_directed = false;
	std::string instance_comment_prefix = "%";

	std::string helpstring = "EXAMPLE CALL\n"
    	"\trobustness -a1 -i graph.gml\n"
		"OPTIONS:\n" 
		"\t-i <path to instance>\n"
		"\t\t\tAccepted Formats: .gml, Networkit Binary via .nkb, edge list.\n"
		"\t-v\t\tAlso output result edges \n"
		"\t-km\t\tk as a function of n, possible values: linear, sqrt, crt, const (default: const).\n"
		"\t-k 20\t\tmultiplicative factor to influence k (double).\n"
		"\t-a[0-6]\tAlgorithm Selection. 0: Random Edges, 1: Submodular Greedy, 2: Stochastic Submodular Greedy, via entire pseudoinverse, 3: Stochastic Submodular Greedy, via on demand column computation, (4: Unassigned), 5: Main Algorithm Prototype, via entire pseudoinverse, 6: Main Algorithm.\n"
		"\t-tr\t\tTest Results for correctness (correct number of edges, correct total resistance value, no edges from original graph). \n"
		"\t-vv\t\tAlso output edges of input graph. \n"
		"\t-eps\t\tEpsilon value for approximative algorithms. Default: 0.1\n"
		"\t-eps2\t\tAbsolute Error bound for UST based approximation. \n"
		"\t-j 12\t\tNumber of threads\n"
		"\t--lamg\t\tUse NetworKit LAMG solver (currently only with a6 and a3)\n"
		"\t--lu\t\tUse Eigen LU solver (currently only with a3 and a6)\n"
		"\t--jlt-lamg\tUse NetworKit LAMG solver in combination with JLT (only with a6)\n"
		"\t--jlt-lu\tUse Eigen LU solver in combination with JLT (only with a6)\n"
		"\t-h[0-2]\t\tHeuristics for a6. 0: random, 1: resistance, 2: similarity\n"
		"\t--seed\t\tSeed for NetworKit random generators.\n"
		"\t-in\t\tFirst node of edge list instances\n"
		"\t-isep\t\tSeparating character of edge list instance file\n"
		"\t-ic\t\tComment character of edge list instance file\n"
		"\n";


	if (argc < 2) {
		std::cout << "Error: Call without arguments.\n" << helpstring;
		return 1;
	}

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
			experiment.verbose = true;
			verbose = true;
			continue;
		}
		if (arg == "-vv" || arg == "--very-verbose") {
			verbose = true;
			experiment.vv = true;
			continue;
		}

		if (arg == "--test-jlt") {
			test_jlt = true;
		}

		if (arg == "-tr") { 
			experiment.verify_result = true;
		}

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
			experiment.threads = atoi(arg_str);
			omp_set_num_threads(atoi(arg_str));
		}

		if (arg == "-eps" || arg == "--eps") {
			experiment.epsilon = std::stod(nextArg(i));
			continue;
		}
		if (arg == "-eps2" || arg == "--eps2") {
			experiment.epsilon2 = std::stod(nextArg(i));
			continue;
		}

		if (arg == "--all-columns") {
			experiment.always_use_known_columns_as_candidates = true;
		}

		
		if (arg == "--seed") {
			auto seed_string = nextArg(i);
			experiment.seed = atoi(seed_string);
			continue;
		}
		if (arg == "-a1" || arg == "--submodular-greedy") { 
			experiment.alg = AlgorithmType::submodular;
			continue;
		}
		if (arg == "-a11") { 
			experiment.alg = AlgorithmType::submodular2;
			continue; 
		}
		if (arg == "-a2" || arg == "--stochastic-submodular-greedy") {
			experiment.alg = AlgorithmType::stochastic;
			continue; 
		}

		if (arg == "-a3") {
			experiment.alg = AlgorithmType::stochastic_dyn;
			continue;
		}
		if (arg == "-a0" || arg == "--random-avg") { 
			experiment.alg = AlgorithmType::random_avg;
			continue; 
		}
		//if (arg == "-a00" || arg == "--random") { run_random = true; continue; }

		if (arg == "-a5") { experiment.alg = AlgorithmType::a5; continue; }
		if (arg == "-a6") { experiment.alg = AlgorithmType::trees; continue; }
		if (arg == "-a7") { experiment.alg = AlgorithmType::stochastic_spectral; continue; }

		if (arg == "-h1") { experiment.heuristic = HeuristicType::lpinvDiag; }
		if (arg == "-h2") { experiment.heuristic = HeuristicType::similarity; }

		if (arg == "-t") { run_tests = true; run_experiments = false; continue; }


		if (arg == "-i" || arg == "--instance") {
			instance_filename = nextArg(i);
			experiment.instanceFile = instance_filename;
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


		if (arg == "--lamg") {
			linalg = LinAlgType::lamg;
		}
		if (arg == "--cg") {
			linalg = LinAlgType::cg;
		}
		if (arg == "--least_squares") {
			linalg = LinAlgType::least_squares;
		}
		if (arg == "--lu") {
			linalg = LinAlgType::lu;
		}
		if (arg == "--qr") {
			linalg = LinAlgType::qr;
		}
		if (arg == "--ldlt") {
			linalg = LinAlgType::ldlt;
		}
		if (arg == "--dense-ldlt") {
			linalg = LinAlgType::dense_ldlt;
		}
		if (arg == "--jlt-lu") {
			linalg = LinAlgType::jlt_lu_sparse;
		}
		if (arg == "--jlt-lamg") {
			linalg = LinAlgType::jlt_lamg;
		}
		experiment.linalg = linalg;
		
	}


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
		assert(g.checkConsistency());
	}

	int k = 1;
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
	k = std::max(k, 1);
	experiment.k = k;



	if (test_jlt) {
		run_experiments = false;
		testJLT(g, instance_filename, k);
	}

	if (run_experiments) {
		NetworKit::ConnectedComponents comp {g};
		comp.run();
		if (comp.numberOfComponents() != 1) {
			std::cout << "Error: Instance " << instance_filename << " is not connected!\n";
			return 1;
		}
		experiment.g = g;
		experiment.run();

	}

	
	if (run_tests) {
	  testRobustnessStochasticGreedySpectral();
	  //testDynamicColumnApprox();
	  //testRobustnessSubmodularGreedy();

	}
	

	return 0;
}
