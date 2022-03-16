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
    SlepcAdapter(NetworKit::Graph const & g, unsigned int approx_c)  {
      
        int i = 0;
	char ** v = NULL;
	ierr = SlepcInitialize(&i,&v,(char*)0,help);
	if (ierr) {
            throw std::runtime_error("SlepcInitialize not working!");
	}
	
	n = (PetscInt)g.numberOfNodes();
	
	PetscInt * nnz = (PetscInt *) malloc( n * sizeof( PetscInt ) ); // nnz per row
	g.forNodes([&](NetworKit::node v) {
		     assert(v < n);
		     nnz[v] = (PetscInt) g.degree(v);
		   });
	

	
	// MatCreate(PETSC_COMM_WORLD,&A); // For general mat?
	// To create a sequential AIJ sparse matrix, A, with m rows and n columns:
	
	MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
	MatSetType(A,MATAIJ);

	// TODO: DO I NEED THE FOLLOWING?
	MatSetFromOptions(A);
	MatSetUp(A);
	MatGetOwnershipRange(A,&Istart,&Iend);

	// MatSetValues(Mat A,PetscInt m,const PetscInt idxm[],PetscInt n,
	//              const PetscInt idxn[],const PetscScalar values[],INSERT_VALUES);

	g.forEdges([&](NetworKit::node u, NetworKit::node v, double w) {
		     if (u == v) {
		       std::cout << "Warning: Graph has edge with equal target and destination!";
		     }
		     
		     double neg_w = -w;
		     const PetscInt * u_adr = (PetscInt *)&u;
		     const PetscInt * v_adr = (PetscInt *)&v;
		     MatSetValues(A, 1, u_adr, 1, u_adr, &w, ADD_VALUES);
		     MatSetValues(A, 1, v_adr, 1, v_adr, &w, ADD_VALUES);
		     MatSetValues(A, 1, u_adr, 1, v_adr, &neg_w, INSERT_VALUES);
		     MatSetValues(A, 1, v_adr, 1, u_adr, &neg_w, INSERT_VALUES);
		   });

	MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);


	//c = (PetscInt)approx_c;
	// TODO: create vectors to store eigenpairs
	// TODO: find best way to store if access is row-wise.
	//MatCreateVecs(A,NULL,&e_v);
    }

  
    ~SlepcAdapter() {
        EPSDestroy(&eps);
        MatDestroy(&A);
	free(e_vectors);
	free(e_values);
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
	
    }
    /* ========================================================================================== */ 


    /* ==========================================================================================
    /* ================================ Run the eigensolver. ====================================
    /* ==========================================================================================    
    */
    PetscErrorCode run_eigensolver() {
      ierr = EPSSolve(eps); CHKERRQ(ierr);
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

    }
  /* ========================================================================================== */
    

  
    void set_eigenpairs() {
      PetscScalar val;
      Vec vec;
      // create once and overwrite in loop
      MatCreateVecs(A, NULL, &vec);
      for (PetscInt i = 0 ; i < nconv; i++) {
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
      VecDestroy(&vec);
    }


  double * get_eigenpairs() {
    if (e_vectors)
      return e_vectors;
  }

  double * get_eigenvalues() {
    if (e_vectors)
      return e_values;
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
