
/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"

/* Subroutine */ int dspmv_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *ap, doublereal *x, integer *incx, doublereal *beta, 
	doublereal *y, integer *incy)//dspmv_("U",&l,&one,Q,s,&inc,&zero,Qs,&inc);
{

int mv_n = *n;
double mv_alpha = *alpha;
int mv_incx = *incx;
int mv_incy = *incy;
double mv_beta = *beta;

    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp1, temp2;
    static integer i, j, k;
    extern logical lsame_(char *, char *);
    static integer kk, ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*  Purpose   
    =======   

    DSPMV  performs the matrix-vector operation   

       y := alpha*A*x + beta*y,   

    where alpha and beta are scalars, x and y are n element vectors and   
    A is an n by n symmetric matrix, supplied in packed form.   

    Parameters   
    ==========   

    UPLO   - CHARACTER*1.   
             On entry, UPLO specifies whether the upper or lower   
             triangular part of the matrix A is supplied in the packed   
             array AP as follows:   

                UPLO = 'U' or 'u'   The upper triangular part of A is   
                                    supplied in AP.   

                UPLO = 'L' or 'l'   The lower triangular part of A is   
                                    supplied in AP.   

             Unchanged on exit.   

    N      - INTEGER.   
             On entry, N specifies the order of the matrix A.   
             N must be at least zero.   
             Unchanged on exit.   

    ALPHA  - DOUBLE PRECISION.   
             On entry, ALPHA specifies the scalar alpha.   
             Unchanged on exit.   

    AP     - DOUBLE PRECISION array of DIMENSION at least   
             ( ( n*( n + 1 ) )/2 ).   
             Before entry with UPLO = 'U' or 'u', the array AP must   
             contain the upper triangular part of the symmetric matrix   
             packed sequentially, column by column, so that AP( 1 )   
             contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 )   
             and a( 2, 2 ) respectively, and so on.   
             Before entry with UPLO = 'L' or 'l', the array AP must   
             contain the lower triangular part of the symmetric matrix   
             packed sequentially, column by column, so that AP( 1 )   
             contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 )   
             and a( 3, 1 ) respectively, and so on.   
             Unchanged on exit.   

    X      - DOUBLE PRECISION array of dimension at least   
             ( 1 + ( n - 1 )*abs( INCX ) ).   
             Before entry, the incremented array X must contain the n   
             element vector x.   
             Unchanged on exit.   

    INCX   - INTEGER.   
             On entry, INCX specifies the increment for the elements of   
             X. INCX must not be zero.   
             Unchanged on exit.   

    BETA   - DOUBLE PRECISION.   
             On entry, BETA specifies the scalar beta. When BETA is   
             supplied as zero then Y need not be set on input.   
             Unchanged on exit.   

    Y      - DOUBLE PRECISION array of dimension at least   
             ( 1 + ( n - 1 )*abs( INCY ) ).   
             Before entry, the incremented array Y must contain the n   
             element vector y. On exit, Y is overwritten by the updated   
             vector y.   

    INCY   - INTEGER.   
             On entry, INCY specifies the increment for the elements of   
             Y. INCY must not be zero.   
             Unchanged on exit.   


    Level 2 Blas routine.   

    -- Written on 22-October-1986.   
       Jack Dongarra, Argonne National Lab.   
       Jeremy Du Croz, Nag Central Office.   
       Sven Hammarling, Nag Central Office.   
       Richard Hanson, Sandia National Labs.   



       Test the input parameters.   

    
   Parameter adjustments   
       Function Body */
#define Y(I) y[(I)-1]
#define X(I) x[(I)-1]
#define AP(I) ap[(I)-1]


    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (mv_n < 0) {
	info = 2;
    } else if (mv_incx == 0) {
	info = 6;
    } else if (mv_incy == 0) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("DSPMV ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (mv_n == 0 || mv_alpha == 0. && mv_beta == 1.) {
	return 0;
    }

/*     Set up the start points in  X  and  Y. */

    if (mv_incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (mv_n - 1) * mv_incx;
    }
    if (mv_incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (mv_n - 1) * mv_incy;
    }

/*     Start the operations. In this version the elements of the array AP 
  
       are accessed sequentially with one pass through AP.   

       First form  y := beta*y. */

    if (mv_beta != 1.) {
	if (mv_incy == 1) {
	    if (mv_beta == 0.) {
		i__1 = mv_n;
		for (i = 1; i <= mv_n; ++i) {
		    Y(i) = 0.;
/* L10: */
		}
	    } else {
		i__1 = mv_n;
		for (i = 1; i <= mv_n; ++i) {
		    Y(i) = mv_beta * Y(i);
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (mv_beta == 0.) {
		i__1 = mv_n;
		for (i = 1; i <= mv_n; ++i) {
		    Y(iy) = 0.;
		    iy += mv_incy;
/* L30: */
		}
	    } else {
		i__1 = mv_n;
		for (i = 1; i <= mv_n; ++i) {
		    Y(iy) = mv_beta * Y(iy);
		    iy += mv_incy;
/* L40: */
		}
	    }
	}
    }
    if (mv_alpha == 0.) {
	return 0;
    }
    kk = 1;
    if (lsame_(uplo, "U")) {

/*        Form  y  when AP contains the upper triangle. */

	if (mv_incx == 1 && mv_incy == 1) {
	    i__1 = mv_n;
	    for (j = 1; j <= mv_n; ++j) {
		temp1 = mv_alpha * X(j);
		temp2 = 0.;
		k = kk;
		i__2 = j - 1;
		for (i = 1; i <= j-1; ++i) {
		    Y(i) += temp1 * AP(k);
		    temp2 += AP(k) * X(i);
		    ++k;
/* L50: */
		}
		Y(j) = Y(j) + temp1 * AP(kk + j - 1) + mv_alpha * temp2;
		kk += j;
/* L60: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = mv_n;
	    for (j = 1; j <= mv_n; ++j) {
		temp1 = mv_alpha * X(jx);
		temp2 = 0.;
		ix = kx;
		iy = ky;
		i__2 = kk + j - 2;
		for (k = kk; k <= kk+j-2; ++k) {
		    Y(iy) += temp1 * AP(k);
		    temp2 += AP(k) * X(ix);
		    ix += mv_incx;
		    iy += mv_incy;
/* L70: */
		}
		Y(jy) = Y(jy) + temp1 * AP(kk + j - 1) + mv_alpha * temp2;
		jx += mv_incx;
		jy += mv_incy;
		kk += j;
/* L80: */
	    }
	}
    } else {

/*        Form  y  when AP contains the lower triangle. */

	if (mv_incx == 1 && mv_incy == 1) {
	    i__1 = mv_n;
	    for (j = 1; j <= mv_n; ++j) {
		temp1 = mv_alpha * X(j);
		temp2 = 0.;
		Y(j) += temp1 * AP(kk);
		k = kk + 1;
		i__2 = mv_n;
		for (i = j + 1; i <= mv_n; ++i) {
		    Y(i) += temp1 * AP(k);
		    temp2 += AP(k) * X(i);
		    ++k;
/* L90: */
		}
		Y(j) += mv_alpha * temp2;
		kk += mv_n - j + 1;
/* L100: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= mv_n; ++j) {
		temp1 = mv_alpha * X(jx);
		temp2 = 0.;
		Y(jy) += temp1 * AP(kk);
		ix = jx;
		iy = jy;
		i__2 = kk + mv_n - j;
		for (k = kk + 1; k <= kk+mv_n-j; ++k) {
		    ix += mv_incx;
		    iy += mv_incy;
		    Y(iy) += temp1 * AP(k);
		    temp2 += AP(k) * X(ix);
/* L110: */
		}
		Y(jy) += mv_alpha * temp2;
		jx += mv_incx;
		jy += mv_incy;
		kk += mv_n - j + 1;
/* L120: */
	    }
	}
    }

    return 0;

/*     End of DSPMV . */

} /* dspmv_ */

