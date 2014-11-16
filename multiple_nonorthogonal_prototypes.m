% "Multiple, nonorthogonal prototypes" (McClelland & Rumelhart, 1985 p.168ff)
% 16 "vision" elements: distinguish between cats, dogs and bagels
% 8 "name" elements: distinguish between 'cat', 'dog' and 'bagel'

1; % not a function file

function A=activations(a,units)
  A = ones(units,"single") .* repmat(a,units,1);
  A = A - diag(diag(A)); % 0 -> diagonal
endfunction

function display (ticks, e, a)
  printf("\tinput  ");
  printf("%5.1f",e(:));
  printf("\n");
  printf("\toutput ");
  printf("%5.1f",a(:));
  printf(" (stable after %d ticks)\n",ticks);
endfunction

function newa = test (a,e,W,units,ddb)
  % FIXME: what should this constant be?
  E = D = single(.27);        % Excitation/Decay
  max_ticks = single(50);     % maximum iterations for activations to stabilise
  precision = 1000;           % overcome floating point comparison problem
  A = activations(a,units);   % activations as a matrix
  A = A .* W;                 % weighted activations
  n = sum(A,2) + e';          % phase 1: determine net activations
  if (ddb)
    e
    a
  endif
  ticks = 0;
  do                          % phase 2: update activations
    old_a = a;
    for i = 1:units
      ni = n(i);
      if (ni > 0)
        d = E * ni * (1 - a(i)) - D * a(i);
      else % ni <= 0
        d = E * ni * (a(i) + 1) - D * a(i);
      endif
      a(i) = a(i) + d;
    endfor
    ticks++;
    ra  = round(a * precision);
    roa = round(old_a * precision);
    if (ddb)
      ra
      roa
    endif
  until (isequal(ra,roa) | ticks == max_ticks) % stable activation
  newa = a;
  display(ticks,e,newa);
endfunction

%% constants
ddb   = 0;                       % DEBUG = 1 / NO_DEBUG = 0
units = single(8);               % number of units in the module
W     = ones(units,"single");    % initial weights
a     = zeros(1,units,"single"); % initial activations
S     = single(2.9);               % global strength
e = single([1 -1 1 -1 1 1 -1 -1]);      % external pattern
% FIXME: weight decay ???

a = test(a,e,W,units,ddb);

for trial = 1:10
  printf("learning trial %d\n",trial);
  a = test(a,e,W,units,ddb);

  %% delta rule
  if (ddb) printf("** Applying delta rule **\n\n"); endif
  A      = activations(a,units);    % new activations as a matrix
  A      = A .* W;                  % weighted activations
  delta  = e' - sum(A,2);
  deltas = activations(delta',units);
  W      = S .* deltas .* A;
  if (ddb)
    W
    deltas
  endif
endfor
