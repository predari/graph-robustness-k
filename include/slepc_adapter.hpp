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
	//unsigned int nz = (PetscInt)g.numberOfEdges();

	PetscInt * nnz = (PetscInt *) malloc( n * sizeof( PetscInt ) );
	g.forNodes([&](NetworKit::node v) {
		     assert(v < n);
		     nnz[v] = (PetscInt) g.degree(v);
		   });


	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	   Compute the operator matrix that defines the eigensystem, Ax=kx
	   In this example, A = L(G), where L is the Laplacian of graph G, i.e.
	   Lii = degree of node i, Lij = -1 if edge (i,j) exists in G
	   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	
	// ierr = MatCreate(PETSC_COMM_WORLD,&A); //CHKERRXX(ierr);
	// To create a sequential AIJ sparse matrix, A, with m rows and n columns:
	// MatCreateSeqAIJ(PETSC_COMM_SELF,PetscInt m,PetscInt n,PetscInt nz,PetscInt *nnz,Mat *A);

	ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A);
	ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n); //CHKERRXX(ierr);
	ierr = MatSetType(A,MATAIJ);

	// TODO: DO I NEED THE FOLLOWING?
	ierr = MatSetFromOptions(A); // CHKERRQ(ierr); // ?
	ierr = MatSetUp(A); // CHKERRQ(ierr); // ?
	ierr = MatGetOwnershipRange(A,&Istart,&Iend); // CHKERRQ(ierr); // ?

	// MatSetValues(Mat A,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar values[],INSERT_VALUES);
	// MatSetValues(Mat A,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar values[],ADD_VALUES);
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

	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); // CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); // CHKERRQ(ierr);

	assert(approx_c <= n);
	c = (PetscInt)approx_c;
	//TODO: create vectors to store eigenpairs
	//TODO: what is the best way to store if I know I will access them row-wise ?
	MatCreateVecs(A,NULL,&e_v);
    }

  
    ~SlepcAdapter() {

        ierr = EPSDestroy(&eps);
        ierr = MatDestroy(&A);
	ierr = SlepcFinalize();
	VecDestroy(&e_v);

    }

    // ============================
    // Slepc Interface
    /*
      Create eigensolver context and set operators. 
      In this case, it is a standard eigenvalue problem
    */
  
    PetscErrorCode create_eigensolver() {
        // TODO: number of eigenpairs to be computed: pairl 
	ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
	ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
	ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
	// set the number of eigenpairs
	ierr = EPSSetDimensions(eps,c,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
	ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
	ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);


	/*
	  Attach deflation space: in this case, the matrix has a constant
	  nullspace, [1 1 ... 1]^T is the eigenvector of the zero eigenvalue
	*/
	ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);
	ierr = VecSet(x,1.0);CHKERRQ(ierr);
	ierr = EPSSetDeflationSpace(eps,1,&x);CHKERRQ(ierr);
	ierr = VecDestroy(&x);CHKERRQ(ierr);
    }

    PetscErrorCode run_eigensolver() {
        ierr = EPSSolve(eps);CHKERRQ(ierr);
    }
  
    /*
      Optional: Get some information from the solver and display it
    */
    PetscErrorCode info_eigensolver() {

        EPSGetIterationNumber(eps,&its);
        PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);
	EPSGetTolerances(eps,&tol,&maxit);
	PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);
        ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
	ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
	EPSGetConverged(eps,&nconv);
        PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);
    }
    // TODO: should include access functions get_eigenvalues, get_eigenvectors, get_eigenvector 
    /* show detailed info */
    PetscErrorCode get_eigensolution() {
     
        ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
	ierr = EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    }


    void get_eigenpairs() {
      PetscScalar l;
      Vec e;
      MatCreateVecs(A,NULL,&e);
      for (PetscInt i = 0 ; i < nconv; i++) {
	EPSGetEigenpair(eps, i, &l, NULL, e, NULL);
	//Compute the relative error associated to each eigenpair
	EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);
	PetscPrintf(PETSC_COMM_WORLD,"   %12f      %12g\n",(double)l,(double)error);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
      }
    }



  

private:
    EPS            eps;             /* eigenproblem solver context */
    Mat            A;               /* operator matrix */
    Vec            x; // diagnostic ? not yet sure 
    EPSType        type; // diagnostic
    PetscReal      error,tol; // diagnostic
    PetscInt       n=10, Istart, Iend, nconv;
    PetscInt       nev,maxit,its; // diagnostic
    PetscErrorCode ierr; // diagnostic
    PetscInt       c; // number of e_pairs for the approximation -- input
    PetscScalar    lambda; // one eigenvalue (TODO: store all eigenvalues in array)
    Vec            e_v; // stores the eigenvector (one here, TODO: store all eigenvectors)
                        // e_v needs to be created before use (constructor) and destroyed afterwards (destructor) 


};

#endif // SLEPC_ADAPTER_H
