#ifndef SLEPC_ADAPTER_H
#define SLEPC_ADAPTER_H

// TODO: add path to petsc that is required by slec
// following ex11.c from slepc/src/eps/tutorials/

#include <vector>
#include <iostream>
#include <networkit/graph/Graph.hpp>
#include <slepceps.h>

static char help[] = "My example with slepc following ex11.c in tutorials.\n";

class SlepcAdapter {
public:
    void setup(NetworKit::Graph const & g, NetworKit::count offset)  {
      
        int arg_c = 0;
	char ** v = NULL;
	ierr = SlepcInitialize(&arg_c,NULL,(char*)0,help);
	if (ierr) {
	  throw std::runtime_error("SlepcInitialize not working!");
	}
	
	n = (PetscInt)g.numberOfNodes();
	NetworKit::count m = (PetscInt)g.numberOfEdges();

	// TODO: ADJUST FOR DYNAMIC BY ALLOCATING MORE SPACE ! INSTEAD OF DEGREE(V), ALLOCATE DEGREE(V) + k. k IS EXPECTED TO BE SMALL COMPARED TO N!!
	PetscInt * nnz = (PetscInt *) malloc( n * sizeof( PetscInt ) ); // nnz per row
	// g.forNodes([&](NetworKit::node v) {
	// 	     assert(v < n);
	// 	     nnz[v] = (PetscInt) g.degree(v) + offset;
	// 	   });
	
	g.forNodes([&](NetworKit::node v) {
		     assert(v < n);
		     nnz[v] = (PetscInt) g.degree(v); // + offset;
		   });
	
	// FIRST APPROACH: Insert elements into PETSc Matrix one by one.
	// To do so, I need to store edges in COO format
	// SEE CSRMatrix.cpp, function laplacianMatrix().
	// ================================================
	// int * row  = (int *) malloc( (2*m - n) * sizeof( int ) );
	// int * col  = (int *) malloc( (2*m - n) * sizeof( int ) );
	// int * val  = (int *) malloc( (2*m - n) * sizeof( int ) );
	// unsigned int cc = 0;
	// g.forNodes([&](const NetworKit::node v){
	// 	     double weightedDegree = 0.0;
	// 	     g.forNeighborsOf(v, [&](const NetworKit::node u, double weight) { // - adj  mat
	// 				   if (u != v) { // exclude diagonal since this would be subtracted by the adjacency weight
	// 				     weightedDegree += weight;
	// 				   }
	// 				   row[cc] = v;
	// 				   col[cc] = u;
	// 				   val[cc] = -weight;
	// 				   cc++;
	// 				 });
	// 	     row[cc] = v;
	// 	     col[cc] = v;
	// 	     val[cc] = weightedDegree;
	// 	     cc++;
	// 	   });
	
	// ================================================
	
	// std::vector<std::vector<node> > edges(g.upperNodeIdBound());
	// g.balancedParallelForNodes([&](NetworKit::node v) {
	// 			     nnz[v] = (PetscInt) g.degree(v) + offset;
	// 			     edges[v].reserve(nnz[v]);
	// 			     g.forEdgesOf(v, [&](node, node v, edgeid) {
	// 						 //if (v > u)
	// 					       edges[v].emplace_back(u);
	// 					     });
	// 			   });


	
	
	// MatCreate(PETSC_COMM_WORLD,&A); // For general mat?
	// To create a sequential AIJ sparse matrix, A, with m rows and n columns:	
	MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A);
	MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
	MatSetType(A,MATAIJ);

	MatSetFromOptions(A);
	MatSetUp(A);
	//MatSetValues(A, 1, row[i], 1, col[i], val[i], INSERT_VALUES);

	// for (PetscInt i = 0; i < 3; i++) {
	//   for (PetscInt j = 0; j < 3; j++) {
	//     PetscScalar v = 1.0;
	//     MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);
	//   }
	// }

	
	// FIRST APPROACH: Insert elements into PETSc Matrix one by one.
	// To do so, I need to store edges in COO format
	// SEE CSRMatrix.cpp, function laplacianMatrix().
	// ================================================	
	g.forEdges([&](NetworKit::node u, NetworKit::node v, double w) {
		     if (u == v) {
		       std::cout << "Warning: Graph has edge with equal target and destination!";
		     }
        	     double neg_w = -w;

		     PetscInt pu = (PetscInt) u;
		     PetscInt pv = (PetscInt) v; 
		     PetscScalar pw = (PetscScalar) w;
		     PetscScalar n_pw = (PetscScalar) -w;
		     MatSetValues(A, 1, &pu, 1, &pu, &pw, ADD_VALUES);
		     MatSetValues(A, 1, &pv, 1, &pv, &pw, ADD_VALUES);
		     MatSetValues(A, 1, &pu, 1, &pv, &n_pw, INSERT_VALUES);
		     MatSetValues(A, 1, &pv, 1, &pu, &n_pw, INSERT_VALUES);
		   });

	
	
	MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

	//ierr = MatView(A, PETSC_VIEWER_STDOUT_SELF);

	free(nnz);
    }

  
    ~SlepcAdapter() {
      free(e_vectors);
      free(e_values);
      EPSDestroy(&eps);
      MatDestroy(&A);
      SlepcFinalize();
    }

    // Slepc Interface

    /* ==========================================================================================
    /* Create eigensolver context and set operators. Our case is a standard eigenvalue problem.
    /* ==========================================================================================
    */  
    PetscErrorCode set_eigensolver(unsigned int numberOfEigenpairs) {
      
      if ( !numberOfEigenpairs ) {
	std::cout << "Warning: No eigenpairs will be computed.";
	return 0;
      }
      assert(numberOfEigenpairs <= n);
      c = (PetscInt) numberOfEigenpairs;
      // storage for eigenpairs
      e_vectors = (double *) calloc(1, n * c * sizeof(double));
      e_values = (double *) calloc(1, (c + 1) * sizeof(double));
      
      ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
      ierr = EPSSetOperators(eps, A, NULL); CHKERRQ(ierr);
      ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
      // request # of eigenpairs
      ierr = EPSSetDimensions(eps, c, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
      ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
      ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
	
      /* From tutorials/ex11.c: attach deflation space exploiting the constant nullspace (e_vector = [1 1 ... 1]^T of e_value = 0). */
      Vec x;
      ierr = MatCreateVecs(A, &x, NULL); CHKERRQ(ierr);
      ierr = VecSet(x, 1.0); CHKERRQ(ierr);
      ierr = EPSSetDeflationSpace(eps, 1, &x); CHKERRQ(ierr);
      ierr = VecDestroy(&x); CHKERRQ(ierr);
      std::cout << "INFO: SET EIGENSOLVER SUCCESSFULLY! \n";
      return ierr;
    }
    /* ========================================================================================== */ 


    /* ==========================================================================================
    /* ================================ Run the eigensolver. ====================================
    /* ==========================================================================================    
    */
    PetscErrorCode run_eigensolver() {
      ierr = EPSSolve(eps); CHKERRQ(ierr);
      std::cout << "INFO: RUN EIGENSOLVER SUCCESSFULLY! \n";
      return ierr;
    }
    /* ========================================================================================== */

    /* ==========================================================================================
    /* ======================= Get info from eigensolver and display it. ========================
    /* ==========================================================================================
    */  
    PetscErrorCode info_eigensolver() {

        ierr = EPSGetType(eps, &type); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type); CHKERRQ(ierr);
        EPSGetIterationNumber(eps, &its);
        PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n", its);
	EPSGetTolerances(eps, &tol, &maxit);
	PetscPrintf(PETSC_COMM_WORLD," Stopping cond: tol=%.4g, maxit=%D\n", (double)tol, maxit);
	ierr = EPSGetDimensions(eps, &nev, NULL, NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n", c);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Number of computed eigenvalues: %D\n", nev);
	EPSGetConverged(eps, &nconv);
        PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n", nconv);
	ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL); CHKERRQ(ierr);
	ierr = EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	std::cout << "INFO: INFO EIGENSOLVER SUCCESSFULLY! \n";
      return ierr;
    }
  /* ========================================================================================== */
    

  
    void set_eigenpairs() {
      PetscScalar val;
      Vec vec;
      // create once and overwrite in loop
      MatCreateVecs(A, NULL, &vec);
      PetscInt i;
      for (i = 0 ; i < nconv; i++) {
      	EPSGetEigenpair(eps, i, &val, NULL, vec, NULL);
      	//Compute relative error associated to each eigenpair
      	EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error);
      	PetscPrintf(PETSC_COMM_WORLD,"   %12f      %12g\n", (double)val, (double)error);
      	PetscPrintf(PETSC_COMM_WORLD,"\n");
      	e_values[i] = (double) val;
      	for(PetscInt j = 0; j < n; j++) {
      	  PetscScalar w;
      	  //VecGetValues(Vec x,PetscInt ni,const PetscInt ix[],PetscScalar y[])
      	  VecGetValues(vec, 1, &j, &w);
      	  *(e_vectors + i*c + j ) = (double) w; 
      	}
      }
      e_values[i+1] = c * e_values[i];
      // TODO: IMPORTANT I HAVENT COMPUTED THE LARGEST EIGENVALUE YET, ONLY APPROXIMATE IT TO BE c TIMES LARGER THAN THE CURRENTLY LARGEST ONE (FROM THE SET OF COMPUTED EVALUES).
      VecDestroy(&vec);
      
    }


  double * get_eigenpairs() const {return e_vectors;}

  double * get_eigenvalues() const {return e_values;}
  
  // TODO: rename to totalResistanceDifferenceExact
  // input solver, a , b --- require c, e_vectors, e_values 
  double SpectralApproximationGainDifference(NetworKit::node a, NetworKit::node b) {
    double * vectors = get_eigenpairs();
    double * values = get_eigenvalues();
    double g = 0.0;
    double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;

    double constant_n = 1.0/(values[c] * values[c]);
    double constant_c = 1.0/(values[c-1] * values[c-1]);

    double sq_diff;
    
    for (int i = 0 ; i < c; i++) {
      sq_diff = *(vectors+a*c+i) - *(vectors+b*c+i);
      sq_diff *= sq_diff;
      dlow += (1.0/(values[i] * values[i]) - constant_n) * sq_diff;
      dup += (1.0/(values[i] * values[i]) - constant_c) * sq_diff;
      rlow += (1.0/values[i] - 1.0/values[c]) * sq_diff;
      rup += (1.0/values[i] - 1.0/values[c-1]) * sq_diff;      
    }

    g = (constant_c + dlow)/ (1.0 + 2.0/values[c] + rlow) + (constant_n + dup)/ (1.0 + 2.0/values[c-1] + rup);
    return (g / 2.0);
  }
  
  //TODO: supposing unweighted here!
  void addEdge(NetworKit::node u, NetworKit::node v) {
    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
      return;
    }
    PetscScalar w = 1.0;
    // const PetscInt * u_adr = (PetscInt *)&u;
    // const PetscInt * v_adr = (PetscInt *)&v;
    // MatSetValues(A, 1, u_adr, 1, u_adr, &w, ADD_VALUES);
    // MatSetValues(A, 1, v_adr, 1, v_adr, &w, ADD_VALUES);
    // w = -w;
    // MatSetValues(A, 1, u_adr, 1, v_adr, &w, INSERT_VALUES);
    // MatSetValues(A, 1, v_adr, 1, u_adr, &w, INSERT_VALUES);

  }
    
  

private:
  EPS            eps;             /* eigenproblem solver context */
  Mat            A;               /* operator matrix */
  PetscInt       n, Istart, Iend;
  
  EPSType        type; // diagnostic
  PetscReal      error, tol; // diagnostic
  PetscInt       maxit, its; // diagnostic
  PetscErrorCode ierr; // diagnostic
  
  PetscInt       c, nev, nconv; // c: # of eigenpairs (input), nev: # of eigenpairs (computed via slepc), nconv: # of converged eigenpairs
  double * e_vectors;  // stores the c eigenvalues (of size c+1) TODO: THE LARGEST EIGENVALUE IS CURRENTLY MISSING
  double * e_values; // stores the c eigenvectors (of size n*c)

};





#endif // SLEPC_ADAPTER_H
