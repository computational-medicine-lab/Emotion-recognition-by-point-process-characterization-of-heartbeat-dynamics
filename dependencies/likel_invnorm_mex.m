function varargout = likel_invnorm_mex(varargin)
% [thetap,k,steps,lambda,loglikel] = likel_invnorm_mex(xn, wn, eta, thetap0, k0, xt, wt, max_steps);
%
% Estimate the inverse Gaussian parameters by maximum likelihood.
%
% [thetap,k,steps,loglikel] = likel_invnorm_mex(xn, wn);
% [thetap,k,steps,loglikel] = likel_invnorm_mex(xn, wn, eta);
% [thetap,k,steps,loglikel] = likel_invnorm_mex(xn, wn, eta, max_steps);
%
% Estimate the parameters using a set of observations.
%    xn is a matrix MxN of regressors, each row of xn is associated to the
%            corresponding element of wn
%    wn is a vector Mx1 of observations
%    eta is a vector Mx1 of weights (when missing or empty, then a constant
%            weight is used)
%    max_steps is the maximum number of allowed newton-raphson iterations
%
%    thetap is a vector Nx1 of coefficients such that xn*thetap gives the mean
%            of the history-dependent inverse gaussian distribution for each
%            observation
%    k is the scale parameter of the inverse gaussian distribution
%            (sometimes called lambda)
%    steps is the number of newton-raphson iterations used
%    loglikel is the loglikelihood of the observations given the optimized parameters
%
%
% [thetap,k,steps,lambda,loglikel] = likel_invnorm_mex(xn, wn, eta, thetap0, k0, xt, wt);
% [thetap,k,steps,lambda,loglikel] = likel_invnorm_mex(xn, wn, eta, thetap0, k0, xt, wt, max_steps);
%
% Performs the estimation described above also considering a right censoring
% term. In addition to the inputs and outputs above:
%    thetap0 is a vector Nx1 of coefficients used as starting point for the
%            newton-rhapson optimization (found, e.g., using the uncensored
%            estimation above)
%    k0 is the starting point for the scale parameter
%    xt is a vector 1xN of regressors, for the censoring part
%    wt is the current value of the future observation
%
%    lambda is the value of the lambda function (the hazard-rate function)
%            corresponding to the right censoring term
%    loglikel is a vector of 2 elements, the first is the loglikelihood of
%            the observations given the optimized parameters, the second is
%            the likelihood of the censoring term
%
%
% Copyright (C) Luca Citi and Riccardo Barbieri, 2010-2011.
% All Rights Reserved. See LICENSE.TXT for license details.
% {lciti,barbieri}@neurostat.mit.edu
% http://users.neurostat.mit.edu/barbieri/pphrv

% if we get here it means the mex file does not exist,
% we need to build it

if exist('likel_invnorm_mex', 'file') ~= 3
    curpath = pwd();
    fpath = fileparts(which('likel_invnorm_mex'));
    cd(fpath);
    compiler = 'gnu';
    arch = '32';
    if ~isempty(strfind(computer(), '64'))
        arch = '64';
    end
    try
        CC  = mex.getCompilerConfigurations('C');
        compiler = CC.Manufacturer;
    end
    switch upper(compiler)
        case 'GNU'
            mex(['LDFLAGS="\$LDFLAGS -Wl,-rpath=' fpath '"'], '-L.', ['-llikel_invnorm' arch], 'likel_invnorm_mex.c');
        case {'LCC', 'MICROSOFT'}
            mex('likel_invnorm_mex.c');
        otherwise
            error('builderror:unknowncompiler', 'Unknown mex compiler %s.', compiler);
    end
    cd(curpath);

    rehash;
end

if exist('likel_invnorm_mex', 'file') == 3
    varargout = cell(max(nargout, 1), 1);
    [varargout{:}] = likel_invnorm_mex(varargin{:});
end

