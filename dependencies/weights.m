function p = weights(time, uk, weight)
% function p = weights(time, uk, weight)
%
% Find the weights of past events using an exponential decay
%
%
% Copyright (C) Luca Citi and Riccardo Barbieri, 2010-2011.
% All Rights Reserved. See LICENSE.TXT for license details.
% {lciti,barbieri}@neurostat.mit.edu
% http://users.neurostat.mit.edu/barbieri/pphrv

p = exp(log(weight) * (time-uk)); %weights
p = p/sum(p) * length(uk); %normalised weights

