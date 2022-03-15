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
    SlepcAdapter(NetworKit::Graph const & g)  {
    
        int c = 0;
	char ** v = NULL;
	ierr = SlepcInitialize(&c,&v,(char*)0,help);
	if (ierr) {
            throw std::runtime_error("SlepcInitialize not working!");
	}
	n = (PetscInt)g.numberOfNodes();
	//unsigned int nz = (PetscInt)g.numberOfEdges();
	m = n;
	N = n * m;

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
	
    }

    ~SlepcAdapter() {

        ierr = EPSDestroy(&eps); // CHKERRQ(ierr);
        ierr = MatDestroy(&A); //CHKERRQ(ierr);
	ierr = SlepcFinalize();
    }

    // ============================
    // Slepc Interface
    /*
      Create eigensolver context and set operators. 
      In this case, it is a standard eigenvalue problem
    */
    PetscErrorCode create_eigensolver(unsigned int pairl) {
        // TODO: number of eigenpairs to be computed: pairl 
	ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
	ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
	ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
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

        ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
	ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
    }
    // TODO: should include access functions get_eigenvalues, get_eigenvectors, get_eigenvector 
    /* show detailed info */
    PetscErrorCode get_eigensolution() {
     
        ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
	ierr = EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    }

private:
    EPS            eps;             /* eigenproblem solver context */
    Mat            A;               /* operator matrix */
    Vec            x;
    EPSType        type;
    PetscInt       N,n=10,m,i,j,II,Istart,Iend,nev;
    PetscScalar    w;
    PetscErrorCode ierr;

};

#endif // SLEPC_ADAPTER_H
