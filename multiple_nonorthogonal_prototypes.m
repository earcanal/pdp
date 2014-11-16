% "Multiple, nonorthogonal prototypes" (McClelland & Rumelhart, 1985 p.168ff)
% 16 "vision" elements: distinguish between cats, dogs and bagels
% 8 "name" elements: distinguish between 'cat', 'dog' and 'bagel'

1; % not a function file

function A=activations(a,units)
  A = ones(units,"single") .* repmat(a,units,1);
  A = A - diag(diag(A));                % 0 -> diagonal
endfunction

function display (ticks, e, a)
  printf("\tinput  ");
  printf("%5.1f",e(:));
  printf("\n");
  printf("\toutput ");
  printf("%5.1f",a(:));
  printf(" (stable after %d ticks)\n",ticks);
 endfunction
 
%% constants
ddb       = 0;                      % DEBUG = 1 / NO_DEBUG = 0
dp        = 1;                      % decimal places in output
precision = 1000;                   % overcome floating point comparison problem
units     = single(8);              % number of units in the module
max_ticks = single(50);             % maximum iterations for activations to stabilise
e = single([1 -1 1 -1 1 1 -1 -1]);  % external pattern
a = zeros(1,units,"single");        % initial activations
W = zeros(units,"single");          % initial weights

% FIXME: what should these constants be?
E = single(.1);                     % excitation 
D = single(.1);                     % activation decay
S = single(.3);                     % global strength
%E = D = S = single(.3);
% FIXME: weight decay ???

for trial = 1:10
  printf("learning trial %d\n",trial);
  A = activations(a,units);   % activations as a matrix
  w = A .* W;                 % weighted activations
  n = sum(w,2) + e';          % phase 1: determine net activations
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
        d = E * ni * (a(i) - (-1)) - D * a(i);
      endif
      a(i) = a(i) + d;
    endfor
    ticks++;
    ra  = round(a * precision);
    roa = round(old_a * precision);
  until (isequal(ra,roa) | ticks == max_ticks) % stable activation
  display(ticks,e,a);
  
  %% delta rule
  if (ddb) printf("** Applying delta rule **\n\n"); endif
  A = activations(a,units);    % new activations as a matrix
  w = A .* W;                  % weighted activations
  delta    = e' - sum(w,2);
  deltas   = activations(delta',units);
  deltas   = deltas * S;
  if (ddb)
    W
    deltas
  endif
  W = deltas;
endfor
