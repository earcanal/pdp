% 'Multiple, nonorthogonal prototypes' (McClelland & Rumelhart, 1985 p.168ff)
% 16 'vision' elements: distinguish between cats, dogs and bagels
% 8 'name' elements: distinguish between 'cat', 'dog' and 'bagel'

function multiple_nonorthoganal_prototypes
  %% constants
  ddb   = 0;                         % DEBUG = 1 / NO_DEBUG = 0
  units = single(8);                 % number of units in the module
  W     = ones(units,'single');      % initial weights
  a     = zeros(1,units,'single');   % initial activations
  S     = single(.1);               % global strength
  e = single([1 -1 1 -1 1 1 -1 -1]); % external pattern
  % FIXME: weight decay ???

  a = test(a,e,W,units,ddb);

  for trial = 1:10
    fprintf('learning trial %d\n',trial);
    %% delta rule
    if (ddb) fprintf('** Applying delta rule **\n\n'); end
    A      = activations(a,units);    % new activations as a matrix
    A      = A .* W;                  % weighted activations
    delta  = e' - sum(A,2);
    deltas = activations(delta',units);
    W      = S .* deltas .* A;
    if (ddb)
      fprintf('weights\n');
     disp(W);
      fprintf('deltas\n');
      disp (deltas);
    end
    a = test(a,e,W,units,ddb);
  end
end

function A=activations(a,units)
  A = ones(units,'single') .* repmat(a,units,1);
  A = A - diag(diag(A)); % 0 -> diagonal
end

function display (ticks, e, a)
  fprintf('\tinput  ');
  fprintf('%6.2f',e(:));
  fprintf('\n');
  fprintf('\toutput ');
  fprintf('%6.2f',a(:));
  fprintf(' (stable after %d ticks)\n',ticks);
end

function newa=test(a,e,W,units,ddb)
  E = single(.2);              % excitation
  D = single(.1);             % decay
  max_ticks = single(50);     % maximum iterations for activations to stabilise
  precision = 1000;           % overcome floating point comparison problem

  % phase 2: update activations
  ticks = 0;
  ra    = 1;
  roa   = 0;
  while (not(isequal(ra,roa)) && ticks < max_ticks) % stable activation
    A = activations(a,units);   % activations as a matrix
    A = A .* W;                 % weighted activations
    n = e' + sum(A,2);          % net activations
    if (ddb)
      fprintf('%s\n',mat2str(e,2));
      fprintf('%s\n',mat2str(a,2));
    end
  
    old_a = a;
    for i = 1:units
      ni = n(i);
      if (ni > 0)
        d = E * ni * (1 - a(i)) - D * a(i);
      else % ni <= 0
        d = E * ni * (a(i) + 1) - D * a(i);
      end
      a(i) = a(i) + d;
    end
    ticks = ticks + 1;
    ra  = round(a * precision);
    roa = round(old_a * precision);
    if (ddb > 1)
      fprintf('ra = %s\n',mat2str(ra));
      fprintf('roa = %s\n',mat2str(roa));
    end
  end % stable activation
  newa = a;
  display(ticks,e,newa);
end
