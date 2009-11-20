/* logistic-lasso.c
 * coded by Dennis Chao
 * 3/2008
 * revised by Pei Wang
 * 4/09
 * C module called by an R wrapper function for Li Hsu's logistic
 * lasso algorithm.
 */

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#define SIGN(a) ((a)>0?1:((a)<0?-1:0))

double computerisk(double *X, double *beta, int n, int p);

void logistic_lasso_weighted_c(double *X, int *N, int *P, double *WEIGHTS, double *TOL, int *MAXITER, double *BETA, int *ITER);

void logistic_lasso_weighted_ini_c(double *X, int *N, int *P, double *WEIGHTS, double *TOL, int *MAXITER, double *BETAINI, double *BETA, int *ITER);

void unbias(double *test, int *N, int *P, double *weight, double *maxw, double *tol, int *maxiter, double *beta1, double *beta2);

void likelihood(double *x, int *N, int *P, double *beta, double *like);


void crossvalid(double *test, int *N, int *P, int *F, double *weight, double *tol, int *maxiter, double *beta_reg,
                double *beta_unbias, double *like_test, double *like_train)
{
// F: number of folds of cross-validation. Make sure N is dividable by F

	int i,j,k, n, n1, n2, p, f, count;
	double *test1, *test2, *beta1, *beta2, maxw, temp;
	//FILE *fp;

	//fp=fopen(fname,"w");

	n=*N;
	p=*P;
	f=*F;

	n1=n*(f-1)/f;
	n2=n/f;

	test1=(double *)malloc(p*n1*sizeof(double));
	test2=(double *)malloc(p*n2*sizeof(double));
	beta1=(double *)malloc(p*p*sizeof(double));
	beta2=(double *)malloc(p*p*sizeof(double));

	// find the largest element in weight matrix, may use other maxw.
	maxw=0.0;
	for(i=0;i<p*p;i++)
		maxw=(maxw>weight[i])?maxw:weight[i];


	for(i=0;i<f;i++)
	{
		count=0;
		// split test data into 2 part
		for(j=0;j<n;j++)
		{

			if(j>=n/f*i && j<n/f*(i+1))
				for(k=0;k<p;k++)
					test2[(j-n/f*i)*p+k]=test[j*p+k];
			else
			{
				for(k=0;k<p;k++)
				{
					test1[count]=test[j*p+k];
					count++;
				}
			}

		}

		// run cross-validation

		//1. get unbias estimate of beta on test1
		//unbias(double *test, int *N, int *P, double *weight, double *maxw, double *tol, int *maxiter, double *beta)
		unbias(test1, &n1, P, weight, &maxw, tol, maxiter, beta1, beta2);


		//2. cal likelihood on test2
		//likelihood(double *x, int *N, int *P, double *beta, double *like)
		likelihood(test2, &n2, P, beta2, &temp);
		like_test[i]=temp;

		likelihood(test1, &n1, P, beta2, &temp);
		like_train[i]=temp;

		//3. output beat1 and beta2
		for(j=0;j<p*p;j++)
		{
			beta_reg[i*p*p+j]=beta1[j];
			beta_unbias[i*p*p+j]=beta2[j];
		}
	}

	//fclose(fp);

	free(test1);
	free(test2);
	free(beta1);
	free(beta2);
}

// take in 4/5 of data, output unbiased beta
void unbias(double *test, int *N, int *P, double *weight, double *maxw, double *tol, int *maxiter, double *beta1, double *beta2)
{
	int i, n, p, iter;
	double *w;

	n=*N;
	p=*P;
	w=(double *)malloc(p*p*sizeof(double));

	logistic_lasso_weighted_c(test, N, P, weight, tol, maxiter, beta1, &iter);

	// unbiased estimate
	for(i=0;i<p*p;i++)
	{
		if(beta1[i]!=0)
			w[i]=0;
		else
			w[i]=*maxw;
	}

	logistic_lasso_weighted_c(test, N, P, w, tol, maxiter, beta2, &iter);

	free(w);
}

// take in 1/5 of data and unbiased beta, output likelihood
void likelihood(double *x, int *N, int *P, double *beta, double *like)
{

	int i, j, k, n, p;
	//double *x_h;
	double temp, temp2;

	n=*N;	//40
	p=*P;	//600

	//x_h=(double *)malloc(n*p*sizeof(double));

	temp2=0.0;
	for(i=0;i<n;i++)
		for(j=0;j<p;j++)
		{
			temp=0.0;

			for(k=0;k<p;k++)
			{
				if(k!=j)
					temp+=x[i*p+k]*beta[k*p+j];
				else
					temp+=beta[k*p+j];
			}

			//x_h[i*p+j]=exp(temp)/(1.0+exp(temp));
			temp2 += x[i*p+j]*temp-log(1.0+exp(temp));

		}

	//temp=0.0;
	//for(i=0;i<n;i++)
		//for(j=0;j<p;j++)
			//temp += x[i*p+j]*log(x_h[i*p+j])+(1-x[i*p+j])*log(1-x_h[i*p+j]);

	*like=temp2;
	//printf("temp2: %f.\n",temp2);
	//free(x_h);
}




////////////////////////////////////////////////////////////////////////////////////
/* logistic.lasso.weighted.c
    X - input data 0/1 vector of length N*P
    N - number of observations
    P - number of variables
    WEIGHTS - P*P matrix of weighted lambda (lasso penalty)
    TOL - tolerance for termination
    MAXITER - maximum number of iterations (unlimited if MAXITER<=0)
    BETA - output beta
    ITER - number of iterations performed before convergence */
//////////////////////////////////////////////////////////////////////////////
void logistic_lasso_weighted_c(double *X, int *N, int *P,
     double *WEIGHTS, double *TOL, int *MAXITER, double *BETA, int *ITER)
{
  int n, p, nWindowSize, nMaxIter;
  double tol;
  double *Y,     /* X, but with 0s replaced with -1s */
    	 *delta; /* step interval bounds */
  double dOldRisk,dNewRisk;
  double dStep;  /* \Delta v_rs */
  double dNumerator;   /* numerator of dStep */
  double dDenominator; /* denominator of dStep */
  int i,j,r,s;
  double diff, temp1, temp2, cur1, cur2;
  int flag, flag0;
  int index, i_index;
  int *pick;

  double *Xbeta,  /* Xbeta[ni,pi]=BETA[pi,pi] + sum_{j!=pi} X_[ni, j]*BETA[j,pi]*/
         *X2,     /* X2[ni,pi]=X[ni,pi]^2 */
         *Xacs;   /* absolute column sum of X/

  /* initialization */
  p = *P;
  n = *N;
  tol = *TOL;
  nMaxIter = (*MAXITER>0?*MAXITER:INT_MAX);
  *ITER = 0;

  Y = (double *) malloc(n*p*sizeof(double));
  delta = (double *) malloc(p*p*sizeof(double));
  pick=(int *) malloc(p*(p-1)*sizeof(int));
  Xbeta=(double *) malloc(n*p*sizeof(double));
  X2=(double *) malloc(n*p*sizeof(double));
  Xacs=(double *) malloc(p*sizeof(double));

 /// Initialization
 for (i=n*p-1; i>=0; i--)
    Y[i] = (X[i]==1?1:-1);

  for (i=p*p-1; i>=0; i--) {
    BETA[i]=0.0;
    delta[i]=1.0;
  }

 for(i=0; i<n*p; i++)
 { Xbeta[i]=0; }

 for(i=0; i<n*p; i++)
 { X2[i]=X[i]*X[i];}

 for(i=0; i<p; i++)
 { Xacs[i]=0;
   for(j=0;j<n;j++)
    Xacs[i]+=fabs(X[j*p+i]);	// Xacs[i]+=fabs(X[j*n+i]);
 }

  dNewRisk=0;

  ///////////////////////
  /* main loop */
  do
  {

/////////////////////////////////////////////////////
    /* Part 1: active set*/
      /* Part 1(1): derive active set*/
      index=0;
      for(r=0; r<p; r++)
       for(s=r; s<p; s++)
         if(fabs(BETA[s*p+r])>1e-6)
            {
				pick[index*2]=r;
				pick[index*2+1]=s;
				index++;
			}
      /* Part 1(2): update active set*/
     do{
		dOldRisk=dNewRisk;
		for(i_index=0; i_index<index; i_index++)
		{
			r=pick[i_index*2];
			s=pick[i_index*2+1];

    		/* compute tentative step */
    		dNumerator=0.0;
			dDenominator=0.0;

			if(r==s)
			     {
				   for (i=0; i<n; i++)
			          {
					  temp1 = Y[i*p+r] * Xbeta[i*p+r];
			          dNumerator += 2*Y[i*p+r]/(1+exp(temp1));
			          diff = fabs(temp1)-fabs(delta[r*p+s]);
			          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff) )));
			          dDenominator += 2*cur1;
				      }
				 }
			else
			     {
                    for (i=0; i<n; i++)
			          {
					  temp1 = Y[i*p+r] * Xbeta[i*p+r];
			          temp2 = Y[i*p+s] * Xbeta[i*p+s];
			          dNumerator += Y[i*p+r]*X[i*p+s]/(1+exp(temp1)) +
			                      Y[i*p+s]*X[i*p+r]/(1+exp(temp2));

			          diff = fabs(temp1)-fabs(delta[r*p+s] * X[i*p+s]);
			          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
			          diff = fabs(temp2)-fabs(delta[r*p+s] * X[i*p+r]);
		              cur2= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
			          dDenominator += X2[i*p+s]*cur1+X2[i*p+r]*cur2;
			         }
			     }
			     dStep=dNumerator/dDenominator;
			     if (BETA[r*p+s]!=0.0)
			     {
			        if (r!=s)
			           dStep -= SIGN(BETA[r*p+s])*WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
			        if (SIGN(BETA[r*p+s] + dStep) != SIGN(BETA[r*p+s]))
			           dStep = -BETA[r*p+s];
			      } else {
			         temp1 = WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
			         if (dStep-temp1 > 0)
			              dStep -= temp1;    /* positive direction */
			         else if (dStep+temp1 < 0)
			              dStep += temp1;    /* negative direction */
			         else
			              dStep = 0.0;
			      }
			      if (dStep < -delta[r*p+s])
			           dStep = -delta[r*p+s];
			      else if (dStep > delta[r*p+s])
			           dStep = delta[r*p+s];


			     /* update Xbeta */
			     if(r!=s)
			     {
			       for(i=0; i<n; i++)
			          {
			             Xbeta[i*p+r]+=X[i*p+s]*dStep;
			             Xbeta[i*p+s]+=X[i*p+r]*dStep;
					  }
			     } else {
					for(i=0; i<n; i++)
			          {
					 	 Xbeta[i*p+r]+=dStep;
					  }
				 }

				 /* update dNewRisk */
				 temp1=fabs(BETA[r*p+s]+dStep)-fabs(BETA[r*p+s]);
				 if(r!=s)
				    dNewRisk+=(Xacs[r]+Xacs[s])* temp1;
				 else
				    dNewRisk+=n* temp1;

			     /* update beta, delta */
			     BETA[r*p+s] += dStep;
			     BETA[s*p+r] = BETA[r*p+s];
			     temp1 = 2.0*fabs(dStep);
			     temp2 = delta[r*p+s]/2.0;
	             delta[r*p+s] = (temp1>temp2?temp1:temp2);
		}// end active for loop
	    flag0=(fabs(dNewRisk-dOldRisk)>1e-6) && (fabs(dNewRisk-dOldRisk)>= tol*dOldRisk) && (++*ITER<nMaxIter);
	  } while(flag0);
	  /* end of Part 1*/
    //} //end debug if

    ////////////////////////////////////////////////////////////
    /// Part 2: full loop ////
    dOldRisk=dNewRisk;
    for (r=0; r<p; r++)
    {
      for (s=r; s<p; s++)
      {
	    	/* compute tentative step */
		    		dNumerator=0.0;
					dDenominator=0.0;

					if(r==s)
					     {
						   for (i=0; i<n; i++)
					          {
							  temp1 = Y[i*p+r] * Xbeta[i*p+r];
					          dNumerator += 2*Y[i*p+r]/(1+exp(temp1));
					          diff = fabs(temp1)-fabs(delta[r*p+s]);
					          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff) )));
					          dDenominator += 2*cur1;
						      }
						 }
					else
					     {
		                    for (i=0; i<n; i++)
					          {
							  temp1 = Y[i*p+r] * Xbeta[i*p+r];
					          temp2 = Y[i*p+s] * Xbeta[i*p+s];
					          dNumerator += Y[i*p+r]*X[i*p+s]/(1+exp(temp1)) +
					                      Y[i*p+s]*X[i*p+r]/(1+exp(temp2));

					          diff = fabs(temp1)-fabs(delta[r*p+s] * X[i*p+s]);
					          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
					          diff = fabs(temp2)-fabs(delta[r*p+s] * X[i*p+r]);
				              cur2= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
					          dDenominator += X2[i*p+s]*cur1+X2[i*p+r]*cur2;
					         }
					     }
					     dStep=dNumerator/dDenominator;
					     if (BETA[r*p+s]!=0.0)
					     {
					        if (r!=s)
					           dStep -= SIGN(BETA[r*p+s])*WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
					        if (SIGN(BETA[r*p+s] + dStep) != SIGN(BETA[r*p+s]))
					           dStep = -BETA[r*p+s];
					      } else {
					         temp1 = WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
					         if (dStep-temp1 > 0)
					              dStep -= temp1;    /* positive direction */
					         else if (dStep+temp1 < 0)
					              dStep += temp1;    /* negative direction */
					         else
					              dStep = 0.0;
					      }
					      if (dStep < -delta[r*p+s])
					           dStep = -delta[r*p+s];
					      else if (dStep > delta[r*p+s])
					           dStep = delta[r*p+s];


					     /* update Xbeta */
					     if(r!=s)
					     {
					       for(i=0; i<n; i++)
					          {
					             Xbeta[i*p+r]+=X[i*p+s]*dStep;
					             Xbeta[i*p+s]+=X[i*p+r]*dStep;
							  }
					     } else {
							for(i=0; i<n; i++)
					          {
							 	 Xbeta[i*p+r]+=dStep;
							  }
						 }

						 /* update dNewRisk */
						 temp1=fabs(BETA[r*p+s]+dStep)-fabs(BETA[r*p+s]);
						 if(r!=s)
						    dNewRisk+=(Xacs[r]+Xacs[s])* temp1;
						 else
						    dNewRisk+=n* temp1;

					     /* update beta, delta */
					     BETA[r*p+s] += dStep;
					     BETA[s*p+r] = BETA[r*p+s];
					     temp1 = 2.0*fabs(dStep);
					     temp2 = delta[r*p+s]/2.0;
	             delta[r*p+s] = (temp1>temp2?temp1:temp2);
      } /* for s */
    } /* for r */
    flag=(fabs(dNewRisk-dOldRisk)>1e-6) && (fabs(dNewRisk-dOldRisk)>= tol*dOldRisk) && (++*ITER<nMaxIter);
  } while (flag);
  free(Y);
  free(delta);
  free(pick);
  free(Xbeta);
  free(X2);
  free(Xacs);
}


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/* logistic.lasso.weighted.c
    X - input data 0/1 vector of length N*P
    N - number of observations
    P - number of variables
    WEIGHTS - P*P matrix of weighted lambda (lasso penalty)
    TOL - tolerance for termination
    MAXITER - maximum number of iterations (unlimited if MAXITER<=0)
    BETAINI- initial beta
    BETA - output beta
    ITER - number of iterations performed before convergence */
//////////////////////////////////////////////////////////////////////////////

void logistic_lasso_weighted_ini_c(double *X, int *N, int *P,
     double *WEIGHTS, double *TOL, int *MAXITER, double *BETAINI, double *BETA, int *ITER)
{

  int n, p, nWindowSize, nMaxIter;
  double tol;
  double *Y,     /* X, but with 0s replaced with -1s */
    	 *delta; /* step interval bounds */
  double dOldRisk,dNewRisk;
  double dStep;  /* \Delta v_rs */
  double dNumerator;   /* numerator of dStep */
  double dDenominator; /* denominator of dStep */
  int i,j,r,s;
  double diff, temp1, temp2, cur1, cur2;
  int flag, flag0;
  int index, i_index;
  int *pick;

  double *Xbeta,  /* Xbeta[ni,pi]=BETA[pi,pi] + sum_{j!=pi} X_[ni, j]*BETA[j,pi]*/
         *X2,     /* X2[ni,pi]=X[ni,pi]^2 */
         *Xacs;   /* absolute column sum of X/

  /* initialization */
  p = *P;
  n = *N;
  tol = *TOL;
  nMaxIter = (*MAXITER>0?*MAXITER:INT_MAX);
  *ITER = 0;

  Y = (double *) malloc(n*p*sizeof(double));
  delta = (double *) malloc(p*p*sizeof(double));
  pick=(int *) malloc(p*(p-1)*sizeof(int));
  Xbeta=(double *) malloc(n*p*sizeof(double));
  X2=(double *) malloc(n*p*sizeof(double));
  Xacs=(double *) malloc(p*sizeof(double));

 /// Initialization
 for (i=n*p-1; i>=0; i--)
    Y[i] = (X[i]==1?1:-1);

  for (i=p*p-1; i>=0; i--) {
    BETA[i]=BETAINI[i];
    delta[i]=1.0;
  }

 for(i=0; i<n*p; i++)
 { Xbeta[i]=0; }

 for(i=0; i<n*p; i++)
 { X2[i]=X[i]*X[i];}

 for(i=0; i<p; i++)
 { Xacs[i]=0;
   for(j=0;j<n;j++)
    Xacs[i]+=fabs(X[j*p+i]);	//Xacs[i]+=fabs(X[j*n+i]);
 }

  dNewRisk=computerisk(X, BETA, n, p);

  ///////////////////////
  /* main loop */
  do
  {

/////////////////////////////////////////////////////
    /* Part 1: active set*/
      /* Part 1(1): derive active set*/
      index=0;
      for(r=0; r<p; r++)
       for(s=r; s<p; s++)
         if(fabs(BETA[s*p+r])>1e-6)
            {
				pick[index*2]=r;
				pick[index*2+1]=s;
				index++;
			}
      /* Part 1(2): update active set*/
    do{
		dOldRisk=dNewRisk;
		for(i_index=0; i_index<index; i_index++)
		{
			r=pick[i_index*2];
			s=pick[i_index*2+1];

    		/* compute tentative step */
    		dNumerator=0.0;
			dDenominator=0.0;

			if(r==s)
			     {
				   for (i=0; i<n; i++)
			          {
					  temp1 = Y[i*p+r] * Xbeta[i*p+r];
			          dNumerator += 2*Y[i*p+r]/(1+exp(temp1));
			          diff = fabs(temp1)-fabs(delta[r*p+s]);
			          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff) )));
			          dDenominator += 2*cur1;
				      }
				 }
			else
			     {
                    for (i=0; i<n; i++)
			          {
					  temp1 = Y[i*p+r] * Xbeta[i*p+r];
			          temp2 = Y[i*p+s] * Xbeta[i*p+s];
			          dNumerator += Y[i*p+r]*X[i*p+s]/(1+exp(temp1)) +
			                      Y[i*p+s]*X[i*p+r]/(1+exp(temp2));

			          diff = fabs(temp1)-fabs(delta[r*p+s] * X[i*p+s]);
			          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
			          diff = fabs(temp2)-fabs(delta[r*p+s] * X[i*p+r]);
		              cur2= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
			          dDenominator += X2[i*p+s]*cur1+X2[i*p+r]*cur2;
			         }
			     }
			     dStep=dNumerator/dDenominator;
			     if (BETA[r*p+s]!=0.0)
			     {
			        if (r!=s)
			           dStep -= SIGN(BETA[r*p+s])*WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
			        if (SIGN(BETA[r*p+s] + dStep) != SIGN(BETA[r*p+s]))
			           dStep = -BETA[r*p+s];
			      } else {
			         temp1 = WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
			         if (dStep-temp1 > 0)
			              dStep -= temp1;    /* positive direction */
			         else if (dStep+temp1 < 0)
			              dStep += temp1;    /* negative direction */
			         else
			              dStep = 0.0;
			      }
			      if (dStep < -delta[r*p+s])
			           dStep = -delta[r*p+s];
			      else if (dStep > delta[r*p+s])
			           dStep = delta[r*p+s];


			     /* update Xbeta */
			     if(r!=s)
			     {
			       for(i=0; i<n; i++)
			          {
			             Xbeta[i*p+r]+=X[i*p+s]*dStep;
			             Xbeta[i*p+s]+=X[i*p+r]*dStep;
					  }
			     } else {
					for(i=0; i<n; i++)
			          {
					 	 Xbeta[i*p+r]+=dStep;
					  }
				 }

				 /* update dNewRisk */
				 temp1=fabs(BETA[r*p+s]+dStep)-fabs(BETA[r*p+s]);
				 if(r!=s)
				    dNewRisk+=(Xacs[r]+Xacs[s])* temp1;
				 else
				    dNewRisk+=n* temp1;

			     /* update beta, delta */
			     BETA[r*p+s] += dStep;
			     BETA[s*p+r] = BETA[r*p+s];
			     temp1 = 2.0*fabs(dStep);
			     temp2 = delta[r*p+s]/2.0;
	             delta[r*p+s] = (temp1>temp2?temp1:temp2);
		}// end active for loop
	    flag0=(fabs(dNewRisk-dOldRisk)>1e-6) && (fabs(dNewRisk-dOldRisk)>= tol*dOldRisk) && (++*ITER<nMaxIter);
	  } while(flag0);
	  /* end of Part 1*/
    //} //end debug if

    ////////////////////////////////////////////////////////////
    /// Part 2: full loop ////
    dOldRisk=dNewRisk;
    for (r=0; r<p; r++)
    {
      for (s=r; s<p; s++)
      {
	    	/* compute tentative step */
		    		dNumerator=0.0;
					dDenominator=0.0;

					if(r==s)
					     {
						   for (i=0; i<n; i++)
					          {
							  temp1 = Y[i*p+r] * Xbeta[i*p+r];
					          dNumerator += 2*Y[i*p+r]/(1+exp(temp1));
					          diff = fabs(temp1)-fabs(delta[r*p+s]);
					          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff) )));
					          dDenominator += 2*cur1;
						      }
						 }
					else
					     {
		                    for (i=0; i<n; i++)
					          {
							  temp1 = Y[i*p+r] * Xbeta[i*p+r];
					          temp2 = Y[i*p+s] * Xbeta[i*p+s];
					          dNumerator += Y[i*p+r]*X[i*p+s]/(1+exp(temp1)) +
					                      Y[i*p+s]*X[i*p+r]/(1+exp(temp2));

					          diff = fabs(temp1)-fabs(delta[r*p+s] * X[i*p+s]);
					          cur1= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
					          diff = fabs(temp2)-fabs(delta[r*p+s] * X[i*p+r]);
				              cur2= (diff<=0 ? 0.25:(1.0/(2.0+exp(diff)+exp(-diff))));
					          dDenominator += X2[i*p+s]*cur1+X2[i*p+r]*cur2;
					         }
					     }
					     dStep=dNumerator/dDenominator;
					     if (BETA[r*p+s]!=0.0)
					     {
					        if (r!=s)
					           dStep -= SIGN(BETA[r*p+s])*WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
					        if (SIGN(BETA[r*p+s] + dStep) != SIGN(BETA[r*p+s]))
					           dStep = -BETA[r*p+s];
					      } else {
					         temp1 = WEIGHTS[r*p+s]/dDenominator; /* L1 penalty */
					         if (dStep-temp1 > 0)
					              dStep -= temp1;    /* positive direction */
					         else if (dStep+temp1 < 0)
					              dStep += temp1;    /* negative direction */
					         else
					              dStep = 0.0;
					      }
					      if (dStep < -delta[r*p+s])
					           dStep = -delta[r*p+s];
					      else if (dStep > delta[r*p+s])
					           dStep = delta[r*p+s];


					     /* update Xbeta */
					     if(r!=s)
					     {
					       for(i=0; i<n; i++)
					          {
					             Xbeta[i*p+r]+=X[i*p+s]*dStep;
					             Xbeta[i*p+s]+=X[i*p+r]*dStep;
							  }
					     } else {
							for(i=0; i<n; i++)
					          {
							 	 Xbeta[i*p+r]+=dStep;
							  }
						 }

						 /* update dNewRisk */
						 temp1=fabs(BETA[r*p+s]+dStep)-fabs(BETA[r*p+s]);
						 if(r!=s)
						    dNewRisk+=(Xacs[r]+Xacs[s])* temp1;
						 else
						    dNewRisk+=n* temp1;

					     /* update beta, delta */
					     BETA[r*p+s] += dStep;
					     BETA[s*p+r] = BETA[r*p+s];
					     temp1 = 2.0*fabs(dStep);
					     temp2 = delta[r*p+s]/2.0;
	             delta[r*p+s] = (temp1>temp2?temp1:temp2);
      } /* for s */
    } /* for r */
    flag=(fabs(dNewRisk-dOldRisk)>1e-6) && (fabs(dNewRisk-dOldRisk)>= tol*dOldRisk) && (++*ITER<nMaxIter);
  } while (flag);
  free(Y);
  free(delta);
  free(pick);
  free(Xbeta);
  free(X2);
  free(Xacs);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * computerisk
 * Xr needs to be allocated but not initialized
 */
double computerisk(double *X, double *beta, int n, int p) {
  double risk=0.0;
  double sum;
  int r,i,j;
  risk=0.0;
  for (r=0; r<p; r++)
  {
    for (i=0; i<n; i++)
    {
      sum = 0.0;
      for (j=0; j<p; j++)
        if(j!=r)
        {
        	//sum += X[i*p+j]*beta[r*p+j];
        	sum +=fabs(X[i*p+j]*beta[r*p+j]);
	    } else {
			//sum += beta[r*p+j];
			sum += fabs(beta[r*p+j]);
		}
      risk += fabs(sum);
    }
  }
  return risk;
}



