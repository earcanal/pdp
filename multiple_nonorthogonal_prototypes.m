% Implementation of "Multiple, nonorthogonal prototypes"
% units x units matrix
% 16 "vision" elements: distinguish between cats, dogs and bagels
% 8 "name" elements: distinguish between 'cat', 'dog' and 'bagel'
%(McClelland & Rumelhart, 1985 p.168ff)

1; % not a function file

if 0
function ii=internal_input(u,i,e)
  % u = unit number
  % i = input vector
  % e = external input to unit
  a    = A .* W(i); % activations * weights
  a(i) = 0;         % unit doesn't activate itself
  ii   = sum(a);
endfunction
endif

function d=delta(u,i,e)
  % the delta rule (see McClelland & Rumelhart(1985), p.165)
  % u = unit number
  % i = input vector
  % e = external input to unit
  d = e(u) - net_internal_input(u,i,e)
endfunction

%% constants

precision = 1000;
units = 8;
% e = external
e = zeros(1,units);
e (2:2:units) = 1;
e = randn(1,units);
a = -1; b = 1;
e=(b-a).*rand(units,1)+a;
e = e';
e = [1 -1 1 -1 1 1 -1 -1];

A = zeros(1,units); % initial activations
W = ones(units);   % initial weights
E = .5; % excitation 
D = .5; % decay
S = .5; % global strength

% weight decay 

%Time is divided into discrete ticks.
%An input pattern is presented at some point in time over some or all of the input lines to the module and is
%then left on for several ticks, until the pattern of activation it produces settles down and stops changing.

% t-1 output vector
ticks = 0;
max_ticks = 50;

internal = ones(units) .* repmat(A,units,1);
internal = internal - diag(diag(internal));
weighted = internal .* W;
net      = sum(weighted,2) + e';

for trials = 1:10
e
A
do
  for i = 1:units
    % phase 1: determine net input
    %u    = a(i);                     % current activation for unit
    %n    = internal_input(i) + e(i); % net input
    % phase 2: update activations
    n = net(i);
    if (n > 0)
      d = E * n * (1 - A(i)) - D * A(i);
    else % n <= 0
      d = E * n * (A(i) - (-1)) - D * A(i);
    endif
    new_A(i) = A(i) + d;
  endfor
  ticks++;
  %ticks
  %e
  %A
  %new_A
  x = round(A * precision);
  y = round(new_A * precision);
  A = new_A;
until (isequal(x,y) | ticks == max_ticks) % stable activation
sprintf("%d ticks",ticks)
A

internal = ones(units) .* repmat(A,units,1);
internal = internal - diag(diag(internal));
weighted = internal .* W;
delta    = e' - sum(weighted,2);
deltas   = ones(units) .* repmat(delta',units,1);
deltas   = deltas - diag(diag(deltas));
deltas   = deltas * S;
sprintf("%d trials",trials)
W
W = deltas;
W
endfor

if 0
% apply delta rule
% FIXME: to all weights in one go
for i = 1:units
  W(i) + S * delta(i,a,e) * A(i);
endfor  
endif

%weight(1:end,1) = [1:1:units]'; % hack in some different values for each unit
