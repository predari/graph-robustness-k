#ifndef SLEPC_ADAPTER_H
#define SLEPC_ADAPTER_H

// TODO: add path to petsc that is required by slec
// following ex11.c from slepc/src/eps/tutorials/


#include <vector>
#include <iostream>
#include <networkit/graph/Graph.hpp>
#include <slepceps.h>


#define EIGENVALUE_MULTIPLIER 4000

static char help[] = "INTERFACE TO EIGENSOLVER \n";

class SlepcAdapter {
public:
  void setup(NetworKit::Graph const & g, NetworKit::count offset, unsigned int numberOfEigenpairs)  {
    
    int arg_c = 0;
    char ** v = NULL;
    ierr = SlepcInitialize(&arg_c,NULL,(char*)0,help);
    if (ierr) {
      throw std::runtime_error("SlepcInitialize() not working!");
    }
    if ( !numberOfEigenpairs ) {
      throw std::runtime_error("Requesting no eigenpairs!");
    }
    
    k = offset;
    n = (PetscInt)g.numberOfNodes();
    NetworKit::count m = (PetscInt) g.numberOfEdges();
    DEBUG("GRAPH INPUT: (n = ", n, " m = ", m, ")\n");
    
    // TODO: ADJUST FOR ALLOCATING MORE SPACE BASED ON k!
    // INSTEAD OF DEGREE(V) + 1, ALLOCATE DEGREE(V) + 1 + k
    // TO AVOID ANOTHER MALLOC (k IS SMALL COMPARED TO AVG DEGREE).	
    
    PetscInt * nnz = (PetscInt *) malloc( n * sizeof( PetscInt ) );	
    g.forNodes([&](NetworKit::node v) {
		 nnz[v] = (PetscInt) g.degree(v) + 1; //+ offset;
	       });
    
    MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A); // includes preallocation
    MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    // TODO: TEMP. PLEASE REMOVE FOLLOWING LINE!! // INGORES NEW MALLOC ERROR!
    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); 
    MatSetType(A, MATSEQAIJ);
    //MatSetOption(A, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE); // for sbaij only created by MatCreateSeqBAIJ().
    MatSetUp(A);
    
    // SETTING MATRIX ELEMENTS
    MatSetValuesROW(g, nnz, &A);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    DEBUG("MATRIX IS CREATED SUCCESSFULLY.");
    DEBUG("VIEW MATRIX:");
    //MatView(A,PETSC_VIEWER_STDOUT_WORLD);
    free(nnz);
    
    c = (PetscInt) numberOfEigenpairs;
    // storage for eigenpairs
    e_vectors = (double *) calloc(1, n * c * sizeof(double));
    e_values = (double *) calloc(1, (c + 1) * sizeof(double));
    // Vec x for deflation eigenvector;
    ierr = MatCreateVecs(A, &x, NULL);
    ierr = VecSet(x, 1.0);
    
    // create eps environment
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, A, NULL);
    EPSSetProblemType(eps, EPS_HEP);
    EPSSetFromOptions(eps);
    // set deflation space
    EPSSetDeflationSpace(eps, 1, &x);

    }

  
  ~SlepcAdapter() {
    free(e_vectors);
    free(e_values);
    EPSDestroy(&eps);
    MatDestroy(&A);
    VecDestroy(&x);
    VecDestroyVecs(nconv+1,&Q);
    SlepcFinalize();    
  }


  PetscErrorCode update_eigensolver() {

    // RESET UPDATED MATRIX
    ierr = EPSSetOperators(eps, A, NULL);
    ierr = EPSSetDeflationSpace(eps, 1, &x);
    // DO I NEED TO: EPSReset(eps); 
    EPSSetInitialSpace(eps, nconv+1, Q);
    DEBUG("VIEW MATRIX:");
    //MatView(A,PETSC_VIEWER_STDOUT_WORLD);    
    run_eigensolver();	
    DEBUG("RERUN EIGENSOLVER SUCCESSFULLY.");
    return ierr;
    }

  

  PetscErrorCode run_eigensolver() {

    // PetscReal error;
    // PetscReal norm;

    PetscScalar val;
    Vec vec;
    PetscInt i;
    // allocate for eigenvector
    MatCreateVecs(A, NULL, &vec);

    EPSSetDimensions(eps, c, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
    ierr = EPSSolve(eps);
    EPSGetConverged(eps, &nconv);
    if (nconv > c) nconv = c;
    // allocation eigenvectors (one more for top)
    VecDuplicateVecs(vec,nconv+1,&Q);

    for (i = 0 ; i < nconv; i++) {
      EPSGetEigenpair(eps, i, &val, NULL, vec, NULL);
      // compute relative error associated to each eigenpair
      // EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error);
      // PetscPrintf(PETSC_COMM_WORLD,"   %12f      %12g \n", (double)val, (double)error);
      // PetscPrintf(PETSC_COMM_WORLD,"\n");
      // VecNorm(vec, NORM_2, &norm);
      // DEBUG("Norm of evector %d : %g \n", i, norm);
      // VecView(vec, PETSC_VIEWER_STDOUT_WORLD);
      e_values[i] = (double) val;
      VecCopy(vec,Q[i]);
      for(PetscInt j = 0; j < n; j++) {
	PetscScalar w;
	VecGetValues(vec, 1, &j, &w);
	*(e_vectors + i + j*c ) = (double) w;
      }
    }
    
    //info_eigensolver();
    
    // reset for largest eigenpair
    EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
    ierr = EPSSolve(eps);
    EPSGetConverged(eps,&nconv_l);
    //PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs FOR LARGE EIGENVALUE: %D\n\n",nconv_l);
    if ( !nconv_l ) {
      throw std::runtime_error("Largest eigenvalue has not converged!");
    }
    EPSGetEigenpair(eps, 0, &val, NULL, vec, NULL);
    //DEBUG("           k          ||Ax-kx||/||kx||\n"
    //"   ----------------- ------------------\n");
    //EPSComputeError(eps, 0, EPS_ERROR_RELATIVE, &error);
    //DEBUG("   %12f       %12g\n",(double)val,(double)error, "\n");
    // store top eigenvector at the end of Q.
    VecCopy(vec,Q[i]);
    e_values[i] = val; 
    VecDestroy(&vec);
    //info_eigensolver();
    return ierr;
  }

  void info_eigensolver() {
    
    EPSType        type;            /* diagnostic: type of solver */
    PetscReal      error, tol;      /* diagnostic: error and tolerance of the solver */
    PetscInt       maxit, its;      /* diagnostic: max iterations, actual iterations */
    PetscInt       nev, nc;         /* diagnostic: # of computed eigenvalues, # of converged eigenvalues */  
    DEBUG(" -------------- INFO ----------------- "); 
    EPSGetType(eps, &type);
    DEBUG(" SOLUTION METHOD: ", type); 
    EPSGetIterationNumber(eps, &its);
    DEBUG(" ITERATION COUNT: ", its);
    EPSGetTolerances(eps, &tol, &maxit);
    DEBUG(" STOP COND: tol= ",(double)tol , " maxit= ", maxit);
    EPSGetDimensions(eps, &nev, NULL, NULL);
    DEBUG(" COMPUT EVALUES: ", nev);
    EPSGetConverged(eps, &nc);
    DEBUG(" CONVRG EVALUES: ", nc);
    PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
    EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);
    EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);
    PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD); 
    EPSView(eps,PETSC_VIEWER_STDOUT_WORLD);
  }

  double * get_eigenpairs() const {return e_vectors;}
  
  double * get_eigenvalues() const {return e_values;}

  double get_max_eigenvalue() const {return e_values[nconv];}

  PetscInt get_nconv() const {return nconv;}



  // compute gain(u,v) using middle points for lambdas l_m = (l_c + l_n)/2.
  // l_m^2 = (l_c^2 + l_n^2)/2.
  double SpectralApproximationGainDifference3(NetworKit::node a, NetworKit::node b) {

    double g = 0.0;
    double d = 0.0, r = 0.0;
    
    //assert(e_values[nconv] > 0);
    double constant_avg = 2.0/(e_values[nconv] + e_values[nconv-1]);
    double constant_avg2 = 2.0/(e_values[nconv-1] * e_values[nconv-1] + e_values[nconv] * e_values[nconv]);
    double sq_diff;
    
    for (int i = 0 ; i < nconv; i++) {

      sq_diff = *(e_vectors+a*c+i) - *(e_vectors+b*c+i);
      sq_diff *= sq_diff;

      d += (1.0/(e_values[i] * e_values[i]) - constant_avg2) * sq_diff;
      r += (1.0/e_values[i] - constant_avg) * sq_diff;
      
    }
    g = ( 2*constant_avg2 + d  ) / (1 + 2 * constant_avg + r);
    return g;
  }


  

  // compute gain(u,v) using upper and low bounds.
  // gain() = (U_bound + L_bound)/2.

  double SpectralApproximationGainDifference1(NetworKit::node a, NetworKit::node b) {

    double g = 0.0;
    double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;
    
    //assert(e_values[nconv] > 0);
    double constant_n = 1.0/(e_values[nconv] * e_values[nconv]);
    double constant_c = 1.0/(e_values[nconv-1] * e_values[nconv-1]);
    double sq_diff;
    
    for (int i = 0 ; i < nconv; i++) {

      sq_diff = *(e_vectors+a*c+i) - *(e_vectors+b*c+i);
      sq_diff *= sq_diff;
      dup += (1.0/(e_values[i] * e_values[i]) - constant_n) * sq_diff;
      rup += (1.0/e_values[i] - 1.0/e_values[nconv-1]) * sq_diff;
      
      dlow += (1.0/(e_values[i] * e_values[i]) - constant_c) * sq_diff;
      rlow += (1.0/e_values[i] - 1.0/e_values[nconv]) * sq_diff;
    }
    g = ( (constant_c + dlow)/ (1.0 + 1.0/e_values[nconv] + rlow)  ) +
        ( (constant_n + dup) / (1.0 + 1.0/e_values[nconv-1] + rup) );

    return (g / 2.0);
  }

  // gain =  D_appx / (1.0 + R_approx).
  // D_appx = (upD + lowD)/2,
  // R_appx = (upR + lowR)/2.
  double SpectralApproximationGainDifference2(NetworKit::node a, NetworKit::node b) {

    double upD = 0.0, upR = 0.0, lowD = 0.0, lowR = 0.0;
    double g = 0.0;
    double lambda_n = 1.0/(e_values[nconv] * e_values[nconv]);
    double lambda_c = 1.0/(e_values[nconv-1] * e_values[nconv-1]);    
    double sq_diff;

    for (int i = 0 ; i < nconv; i++) {
      sq_diff = *(e_vectors+a*c+i) - *(e_vectors+b*c+i);
      sq_diff *= sq_diff;
      
      lowD += (1.0/(e_values[i] * e_values[i]) - lambda_n) * sq_diff;      
      upR += (1.0/e_values[i] - 1.0/e_values[nconv-1]) * sq_diff;      
      
      upD += (1.0/(e_values[i] * e_values[i]) - lambda_c) * sq_diff;
      lowR += (1.0/e_values[i] - 1.0/e_values[nconv]) * sq_diff;
    }

    // upD += 2.0*lambda_c;
    // lowD += 2.0*lambda_n;    
    // upR += 2.0/e_values[nconv-1];
    // lowR +=  2.0/e_values[nconv];
    // return (upD + lowD)/(2.0 + upR + lowR);

    return (upD + 2.0*lambda_c + lowD + 2.0*lambda_n)/(2.0 + 2.0/e_values[nconv] + 2.0/e_values[nconv-1] + upR + lowR);
  }

  // gain =  D_appx / (1.0 + R_approx).
  // D_appx = (upD + lowD)/2,
  // R_appx = (upR + lowR)/2.

  double SpectralApproximationGainDifference2(NetworKit::node a, NetworKit::node b,
					      double *vectors, double *values, int s) {
    std::cout << " Spectral Diff 2. \n";    
   
    double upD = 0.0, upR = 0.0, lowD = 0.0, lowR = 0.0;
    double g = 0.0;
    double lambda_n = 1.0/(values[nconv] * values[nconv]);
    double lambda_c = 1.0/(values[s-1] * values[s-1]);    
    double sq_diff;
    
    for (int i = 0 ; i < s; i++) {
      sq_diff = *(vectors+a*c+i) - *(vectors+b*c+i);
      sq_diff *= sq_diff;

      lowD += (1.0/(values[i] * values[i]) - lambda_n) * sq_diff;      
      upR += (1.0/values[i] - 1.0/values[s-1]) * sq_diff;      
      
      upD += (1.0/(values[i] * values[i]) - lambda_c) * sq_diff;
      lowR += (1.0/values[i] - 1.0/values[nconv]) * sq_diff; 
    }
    std::cout << "lower1 :" << 2.0/values[nconv] + lowR << "  lower2 :" << lowD + 2.0*lambda_n << "  upper1 :" << 2.0/values[s-1] + upR << "  upper2 :" << upD + 2.0*lambda_c << "\n";    
    return (upD + 2.0*lambda_c + lowD + 2.0*lambda_n)/(2.0 + 2.0/values[nconv] + 2.0/values[s-1] + upR + lowR);
  }

  double SpectralApproximationGainDifference1(NetworKit::node a, NetworKit::node b,
					      double *vectors, double *values, int s) {
    std::cout << " Spectral Diff 1. \n";    
    double g = 0.0;
    double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;
    
    double lambda_n = 1.0/(values[nconv] * values[nconv]);
    double lambda_c = 1.0/(values[s-1] * values[s-1]);    
    double sq_diff;
    
    for (int i = 0 ; i < s; i++) {
      sq_diff = *(vectors+a*c+i) - *(vectors+b*c+i);
      sq_diff *= sq_diff;
      dup += (1.0/(values[i] * values[i]) - lambda_n) * sq_diff;
      rup += (1.0/values[i] - 1.0/values[s-1]) * sq_diff;
      
      dlow += (1.0/(values[i] * values[i]) - lambda_c) * sq_diff;
      rlow += (1.0/values[i] - 1.0/values[nconv]) * sq_diff;
    }
    
    std::cout << "lower1 :" << 2.0/values[nconv] + rlow << "  lower2 :" << dup + 2.0*lambda_n <<  "  upper1 :" << 2.0/values[s-1] + rup << "  upper2 :" << dlow + 2.0*lambda_c << "\n";

    g = ( (2.0*lambda_c + dlow)/ (1.0 + 2.0/values[nconv] + rlow)  ) +
        ( (2.0*lambda_n + dup) / (1.0 + 2.0/values[s-1] + rup) );

    return (g / 2.0);
  }


  // compute gain(u,v) using middle points for lambdas l_m = (l_c + l_n)/2.
  // l_m^2 = (l_c^2 + l_n^2)/2.
  double SpectralApproximationGainDifference3(NetworKit::node a, NetworKit::node b,
					      double *vectors, double *values, int s) {

    double g = 0.0;
    double d = 0.0, r = 0.0;
    
    //assert(e_values[nconv] > 0);
    double constant_avg = 2.0/(values[nconv] + values[s-1]);
    double constant_avg2 = 2.0/(values[s-1] * values[s-1] + values[nconv] * values[nconv]);
    double sq_diff;
    
    for (int i = 0 ; i < nconv; i++) {

      sq_diff = *(vectors+a*c+i) - *(vectors+b*c+i);
      sq_diff *= sq_diff;

      d += (1.0/(values[i] * values[i]) - constant_avg2) * sq_diff;
      r += (1.0/values[i] - constant_avg) * sq_diff;
      
    }
    g = ( 2*constant_avg2 + d  ) / (1 + 2 * constant_avg + r);
    return g;
  }


  
  
  // // slower than SpectralApproximationGainDifference1
  // double SpectralApproximationGainDifference1a(NetworKit::node a, NetworKit::node b) {

  //   double g = 0.0;
  //   double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;
    
  //   double constant_n = 1.0/(e_values[nconv] * e_values[nconv]);
  //   double constant_c = 1.0/(e_values[nconv-1] * e_values[nconv-1]);
  //   double sq_diff;
    
  //   for (int i = 0 ; i < nconv; i++) {
  //     PetscScalar w_a;
  //     PetscScalar w_b;
  //     VecGetValues(Q[i], 1, (PetscInt*)&a, &w_a);
  //     VecGetValues(Q[i], 1, (PetscInt*)&b, &w_b);
  //     sq_diff = w_a - w_b;
  //     sq_diff *= sq_diff;
  //     dup += (1.0/(e_values[i] * e_values[i]) - constant_n) * sq_diff;
  //     rup += (1.0/e_values[i] - 1.0/e_values[nconv-1]) * sq_diff;
      
  //     dlow += (1.0/(e_values[i] * e_values[i]) - constant_c) * sq_diff;
  //     rlow += (1.0/e_values[i] - 1.0/e_values[nconv]) * sq_diff;
  //   }
  //   g = ( (constant_c + dlow)/ (1.0 + 1.0/e_values[nconv] + rlow)  ) +
  //       ( (constant_n + dup) / (1.0 + 1.0/e_values[nconv-1] + rup) );

  //   return (g / 2.0);
  // }
  

  double SpectralToTalEffectiveResistance() {
    double Sum = 1.0/e_values[nconv];
    for (int i = 0 ; i < nconv; i++)
      Sum += 1.0/e_values[i];
    return n*Sum;
  }
  

  double SpectralToTalEffectiveResistance(int s) {
    assert( s <= nconv);
    double Sum = 1.0/e_values[nconv];
    for (int i = 0 ; i < s; i++)
      Sum += 1.0/e_values[i];
    return n*Sum;
  }

  
  //TODO: supposing unweighted here!
  void addEdge(NetworKit::node u, NetworKit::node v) {

    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
      return;
    }
    
    PetscInt a = (PetscInt) u;
    PetscInt b = (PetscInt) v; 
    PetscScalar w = 1.0;
    PetscScalar nw = -1.0;
		 
    MatSetValues(A, 1, &a, 1, &a, &w, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &b, &w, ADD_VALUES);
    MatSetValues(A, 1, &a, 1, &b, &nw, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &a, &nw, ADD_VALUES);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    DEBUG("ADDING NEW EDGE: (", u, ", ", v, ") SUCCESSFULLY.");    

  }
    
  

private:

  // MatCreateSeqAIJMP(PETSC_COMM_WORLD, #rows, #cols, 0, nnz, &A);
  PetscErrorCode  MatCreateSeqAIJMP(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
  {
    PetscErrorCode ierr; 
    PetscFunctionBegin;
    ierr = MatCreate(comm, A); 
    ierr = MatSetSizes(*A, m, n, m, n);
    // TODO: TEMP. PLEASE REMOVE FOLLOWING LINE!!
    ierr = MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    ierr = MatSetFromOptions(*A);
    ierr = MatSetType(*A, MATSEQAIJ); // MATAIJ
    //ierr = MatSetType(*A, MATSEQSBAIJ); // MATSBAIJ
    ierr = MatSeqAIJSetPreallocation(*A, nz, (PetscInt*)nnz);
    PetscFunctionReturn(0);
  }

  // FIRST APPROACH: INSERT ELEMENTS TO PETSc MATRIX (ONE BY ONE).
  // MatSetValuesELM(g, &A);
  void  MatSetValuesELM(NetworKit::Graph const & g, Mat *A) {
    g.forEdges([&](NetworKit::node u, NetworKit::node v, double w) {
		 if (u == v) {
		   std::cout << "Warning: Graph has edge with equal target and destination!";
		 }        	     
		 PetscInt a = (PetscInt) u;
		 PetscInt b = (PetscInt) v; 
		 PetscScalar vv = (PetscScalar) w;
		 PetscScalar nv = (PetscScalar) -w;
		 
		 //std::cout<< " edge (" << a << ", " << b << ") w = " << w << "or " << vv << "\n";
		 
		 MatSetValues(*A, 1, &a, 1, &a, &vv, ADD_VALUES);
		 MatSetValues(*A, 1, &b, 1, &b, &vv, ADD_VALUES);
		 MatSetValues(*A, 1, &a, 1, &b, &nv, ADD_VALUES); // DONT MIX ADD AND INSERT_VALUES
		 MatSetValues(*A, 1, &b, 1, &a, &nv, ADD_VALUES); // DONT MIX ADD AND INSERT_VALUES
	       });
    

  }

  // SECOND APPROACH: INSERT ROW BY ROW.
  // MatSetValuesROW(g, nnz, &A);
  void  MatSetValuesROW(NetworKit::Graph const & g, PetscInt * nnz, Mat *A)
  {
    // g.balancedParallelForNodes([&](NetworKit::node v) {    
    g.forNodes([&](const NetworKit::node v){
  		 double weightedDegree = 0.0;
  		 PetscInt * col  = (PetscInt *) malloc(nnz[v] * sizeof(PetscInt));
  		 PetscScalar * val  = (PetscScalar *) malloc(nnz[v] * sizeof(PetscScalar));
  		 unsigned int idx = 0;
  		 g.forNeighborsOf(v, [&](const NetworKit::node u, double w) { // - adj  mat
  				       if (u != v) { // exclude diagonal since this would be subtracted by the adjacency weight
  					 weightedDegree += w;
  				       }
  				       col[idx] = (PetscInt)u;
  				       val[idx] = -(PetscScalar)w;
  				       idx++;
  				     });
  		 col[idx] = v;
  		 val[idx] = weightedDegree;
  		 PetscInt a = (PetscInt) v;
		 // std::cout<< " node " << a << " : [";
		 // for(int i = 0; i < nnz[v]; i++)
		 //   std::cout << col[i] << " (" << val[i] << ") ";
		 // std::cout << "] \n";
		 // std::cout << "idx =  " << idx << "\n";
		 // std::cout << "nnz[v] =  " << nnz[v] << "\n";
		 
		 MatSetValues(*A, 1, &a, nnz[v] , col, val, INSERT_VALUES);
  	       });	
  }
  
  


  
  EPS            eps;             /* eigenproblem solver context*/
  Mat            A;               /* operator matrix */
  PetscInt       n;               /* size of matrix */
  Vec            x;               /* vector representing the nullspace */
  PetscErrorCode ierr;            /* diagnostic: petscerror */
  PetscInt       c, nconv, nconv_l = 0; /* # requested eigenvalues, # converged eigenvalues (small), # of converged top eigenvalue */
  double *       e_vectors;       /* stores the eigenvectors (of size n*nconv) */
  double *       e_values;        /* stores eigenvalues (of size nconv + 1) */
  Vec            *Q;
  NetworKit::count k;
  
};



#endif // SLEPC_ADAPTER_H
