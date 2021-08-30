/******************************************
Evaluate gaussian likelihood
May 18 2010
Luca Citi

% Copyright (C) Luca Citi and Riccardo Barbieri, 2010-2011.
% All Rights Reserved. See LICENSE.TXT for license details.
% {lciti,barbieri}@neurostat.mit.edu
% http://users.neurostat.mit.edu/barbieri/pphrv

******************************************/


/*
* To compile this mex file manually:
*
* Matlab for Linux, Octave for Linux, and Octave for Windows (with gcc/mingw compiler):
* mex 'LDFLAGS="\$LDFLAGS -Wl,-rpath=."' -L. -llikel_invnorm[32/64] likel_norm_mex.c
*
* Matlab for Windows (with the default LCC compiler):
* mex likel_norm_mex.c
*/


#include "mex.h"
#include <math.h>


#if defined(__GNUC__) || defined(STATIC)
# define FNORPTR
# define InitDLL()
#else
# define EXPLICIT
# define FNORPTR *
#endif


typedef double real;
int (FNORPTR maximize_loglikelihood_norm)(int nsamp, int nregr, real *xn, real *wn, real *eta, real *thetap, real *p_k, real *xt, real wt, int maxsteps);
int (FNORPTR maximize_loglikelihoodunc_norm)(int nsamp, int nregr, real *xn, real *wn, real *eta, real *thetap, real *k, int maxsteps);
real (FNORPTR loglikelihood_norm_cens)(real w, real m, real k, int, int *);
void (FNORPTR transpose)(real *src, real *dest, int M, int N);


#if defined(EXPLICIT)

#if defined(_WIN64) || defined(__amd64__)
#define DLLNAME "liblikel_invnorm64.dll"
#else
#define DLLNAME "liblikel_invnorm32.dll"
#endif

#include <windows.h>
#include <stdlib.h>

HINSTANCE lib_handle = NULL;

void FreeDLL()
{
    if (lib_handle != NULL)
        FreeLibrary(lib_handle);
}

void InitDLL()
{
    char dll_path[_MAX_PATH] = DLLNAME;
    HMODULE hm = NULL;
    if (lib_handle != NULL)
        return;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | 
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) &mexFunction, &hm)) {
        char drive[_MAX_DRIVE], dir[_MAX_DIR];
        GetModuleFileNameA(hm, dll_path, _MAX_PATH);
        _splitpath_s(dll_path, drive, _MAX_DRIVE, dir, _MAX_DIR, NULL, 0, NULL, 0);
        _makepath_s(dll_path, _MAX_PATH, drive, dir, DLLNAME, NULL);
    }
    lib_handle = LoadLibrary(dll_path);
    if (lib_handle == NULL)
        mexErrMsgIdAndTxt("likel_norm_mex:load_dll", "ERROR: unable to load '%s'.\n", dll_path);
    maximize_loglikelihood_norm = GetProcAddress(lib_handle, "maximize_loglikelihood_norm");
    maximize_loglikelihoodunc_norm = GetProcAddress(lib_handle, "maximize_loglikelihoodunc_norm");
    transpose = GetProcAddress(lib_handle, "transpose");
    loglikelihood_norm_cens = GetProcAddress(lib_handle, "loglikelihood_norm_cens");
    if (maximize_loglikelihood_norm == NULL || maximize_loglikelihoodunc_norm == NULL || transpose == NULL || loglikelihood_norm_cens == NULL) {
        FreeLibrary(lib_handle);
        mexErrMsgIdAndTxt("likel_norm_mex:dll_functions", "ERROR: unable to find DLL functions\n");
    }
    mexAtExit(FreeDLL);
}

#endif /*defined(EXPLICIT)*/


char syntax[] = "Syntax:\n[thetap,k,steps] = likel_norm_mex(xn, wn);\n[thetap,k,steps] = likel_norm_mex(xn, wn, eta);\n[thetap,k,steps] = likel_norm_mex(xn, wn, eta, max_steps);\n[thetap,k,steps,lambda] = likel_norm_mex(xn, wn, eta, thetap0, k0, xt, wt);\n[thetap,k,steps,lambda] = likel_norm_mex(xn, wn, eta, thetap0, k0, xt, wt, max_steps);";

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i, nsamp, nregr, steps=1000;
    double *xn, *wn, *eta;
    mxArray *m_thetap, *m_k;

    InitDLL();

    if ((nrhs != 2) && (nrhs != 3) && (nrhs != 4) && (nrhs != 7) && (nrhs != 8))
        mexErrMsgIdAndTxt("likel_norm_mex:syntax", syntax);
    for (i=0; i<nrhs; ++i)
        if (!mxIsDouble(prhs[i]))
            mexErrMsgIdAndTxt("likel_norm_mex:double", "All inputs should be of class double.");

/* xn, wn, eta */
    nregr = mxGetN(prhs[0]);
    nsamp = mxGetM(prhs[0]);
    if (mxGetNumberOfElements(prhs[1]) != nsamp || (nrhs != 2 && !mxIsEmpty(prhs[2]) && mxGetNumberOfElements(prhs[2]) != nsamp))
        mexErrMsgIdAndTxt("likel_norm_mex:nsamp", "The number of rows of 'xn', 'wn', and 'eta' should be the same.");

    m_thetap = mxCreateDoubleMatrix(nregr, 1, mxREAL);
    m_k = mxCreateDoubleMatrix(1, 1, mxREAL);
    xn = mxMalloc(nregr*nsamp*sizeof(double));
    transpose(mxGetPr(prhs[0]), xn, nsamp, nregr);
    wn = mxGetPr(prhs[1]);
    if (nrhs == 2 || mxIsEmpty(prhs[2])) {
        eta = mxMalloc(nsamp*sizeof(double));
        for (i=0; i<nsamp; ++i)
            eta[i] = 1.;        
    } else {
        eta = mxGetPr(prhs[2]);
    }

    if (nrhs == 4 || nrhs == 8)
        steps = (int)mxGetScalar(prhs[nrhs-1]);
    if (nrhs <= 4) {
        steps = maximize_loglikelihoodunc_norm(nsamp, nregr, xn, wn, eta, mxGetPr(m_thetap), mxGetPr(m_k), steps);
        if (nlhs >= 5)
            plhs[4] = mxCreateDoubleMatrix(0, 0, mxREAL);
    } else {
        double *thetap = mxGetPr(m_thetap);
        double *k = mxGetPr(m_k);
        double *xt = mxGetPr(prhs[5]);
        double wt = mxGetScalar(prhs[6]);
        double *thetap0 = mxGetPr(prhs[3]);
        for (i=0; i<nregr; ++i)
            thetap[i] = thetap0[i];
        *k = mxGetScalar(prhs[4]);
        steps = maximize_loglikelihood_norm(nsamp, nregr, xn, wn, eta, thetap, k, xt, wt, steps);
        if (nlhs >= 4) {
            double lambda, mu;
            for (i=0, mu=0; i<nregr; ++i)
                mu += thetap[i] * xt[i];
            lambda = loglikelihood_norm_cens(wt, mu, *k, 1, NULL);
            plhs[3] = mxCreateDoubleScalar(lambda);
            if (nlhs >= 5)
                plhs[4] = mxCreateDoubleMatrix(0, 0, mxREAL);
        }
    }

    if (nrhs == 2 || mxIsEmpty(prhs[2]))
        mxFree(eta);
    mxFree(xn);
    plhs[0] = m_thetap;
    if (nlhs >= 2)
        plhs[1] = m_k;
    else
        mxDestroyArray(m_k);
    if (nlhs >= 3)
        plhs[2] = mxCreateDoubleScalar((double)steps);
}
