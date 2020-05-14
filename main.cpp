#include <iostream>
#include <Eigen/Dense>

#include <networkit/graph/Graph.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/auxiliary/Random.hpp>
 
using Eigen::MatrixXd;
using namespace NetworKit;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  Aux::Random::setSeed(42, false);
  auto G = NetworKit::ErdosRenyiGenerator(
    100, 0.15, false
  ).generate();

  Graph G2 = NetworKit::ErdosRenyiGenerator(10, 0.15, false).generate();
  auto G3 = NetworKit::ErdosRenyiGenerator(10, 0.15, false).generate();
  G2.append(G3);
  for(int i=0; i<20; i++) {
    auto v = G2.randomNode();
    auto w = G2.randomNode();
    if (v != w) {
      G2.addEdge(v, w);
    }
  }
  std::cout << G2.toString();
}
