#ifndef SLEPC_ADAPTER_H
#define SLEPC_ADAPTER_H

// TODO: add path to petsc that is required by slec
// following ex11.c from slepc/src/eps/tutorials/

#include <vector>
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
	    return ierr;
	}
	n = (PetscInt)g.numberOfNodes();
	m = n;
	N = n * m;

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	   Compute the operator matrix that defines the eigensystem, Ax=kx
	   In this example, A = L(G), where L is the Laplacian of graph G, i.e.
	   Lii = degree of node i, Lij = -1 if edge (i,j) exists in G
	   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	
	ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
	ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	ierr = MatSetUp(A);CHKERRQ(ierr);

	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
	for (II=Istart;II<Iend;II++) {
	  i = II/n; j = II-i*n;
	  w = 0.0;
	  if (i>0) { ierr = MatSetValue(A,II,II-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); w=w+1.0; }
	  if (i<m-1) { ierr = MatSetValue(A,II,II+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); w=w+1.0; }
	  if (j>0) { ierr = MatSetValue(A,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); w=w+1.0; }
	  if (j<n-1) { ierr = MatSetValue(A,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); w=w+1.0; }
	  ierr = MatSetValue(A,II,II,w,INSERT_VALUES);CHKERRQ(ierr);
	}
	
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	
    }

    ~SlepcAdapter() {

        ierr = EPSDestroy(&eps);CHKERRQ(ierr);
	ierr = MatDestroy(&A);CHKERRQ(ierr);
	ierr = SlepcFinalize();
    }

    // ============================
    // Slepc Interface
    /*
      Create eigensolver context and set operators. 
      In this case, it is a standard eigenvalue problem
    */
    void create_eigensolver(unsigned int pairl) {
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

    void run_eigensolver() {
        ierr = EPSSolve(eps);CHKERRQ(ierr);
    }

    /*
      Optional: Get some information from the solver and display it
    */
    void info_eigensolver() {

        ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
	ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
    }
    // TODO: should include access functions get_eigenvalues, get_eigenvectors, get_eigenvector 
    /* show detailed info */
    void get_eigensolution() {
     
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
