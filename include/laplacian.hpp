#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <vector>
#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/centrality/ApproxEffectiveResistance.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/algebraic/Vector.hpp>

using namespace NetworKit;


std::vector<double> approxEffectiveResistances(Graph const &G, int &pivot);

Vector computeLaplacianPseudoinverseColumn(CSRMatrix const &laplacian, int index, bool connected = false);

// Update formula for the pseudoinverse of the Laplacian as an edge is added to the graph.
// Add edge (i, j) and compute difference between the k-th column of the pseudoinverse.
// Takes the columns of the pseudoinverse that correspond to the vertices i and j.
// Add this to the old to get the new.
Vector laplacianPseudoinverseColumnDifference(Vector const & column_i, int i, Vector const & column_j, int j, int k, double conductance = 1.0);

// Update formula for the trace of the lap pinv as an edge is added to the graph.
// Add edge (i, j) and compute the difference of the traces of the pseudoinverses.
// Add this to the old to get the new.
double laplacianPseudoinverseTraceDifference(Vector const & column_i, int i, Vector const &column_j, int j, double conductance = 1.0);


// Compute a stochastic approximation of the diagonal of the Moore-Penrose pseudoinverse of the laplacian matrix of a graph.
std::vector<double> approxLaplacianPseudoinverseDiagonal(Graph const &G, double epsilon = 0.1);


#endif // LAPLACIAN_H