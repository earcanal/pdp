% 'Multiple, nonorthogonal prototypes' (McClelland & Rumelhart, 1985 p.168ff)
% 16 'vision' elements: distinguish between cats, dogs and bagels
% 8 'name' elements: distinguish between 'cat', 'dog' and 'bagel'

function multiple_nonorthogonal_prototypes
  %% constants
  ddb   = 0;                         % DEBUG level 0-2
  units = single(8);                 % number of units in the module
  W     = zeros(units,'single');     % initial weights
  a     = zeros(1,units,'single');   % initial activations
  S     = single(.1);                % global strength
  e = single([1 -1 1 -1 1 1 -1 -1]); % external pattern
  wd    = ones(units) * 20;          % weight decay

  a = test(a,e,W,units,ddb);

  for trial = 1:10
    fprintf('learning trial %d\n',trial);
    %% delta rule
    if (ddb) fprintf('** Applying delta rule **\n\n'); end
    A     = activations(a,units);    % activations as a matrix
    delta = e' - sum(A,2);
    if (ddb)
      fprintf('a(i)\n');
      disp(A);
      fprintf('e = %s\ni = %s\n',mat2str(e'),mat2str(sum(A,2),2));
      fprintf('e - i = %s\n',mat2str(delta,2));
    end
    deltas = repmat(delta,1,units);
    deltas = deltas - diag(diag(deltas));
    W      = S .* deltas .* A;
    if (ddb)
      fprintf('deltas\n');
      disp(deltas);
      fprintf('weights\n');
      disp(W);
    end
    W = W ./ wd;                     % weight decay

    a = test(a,e,W,units,ddb);
  end
  % FIXME: partial input
  a = test(a,[1 -1 1 -1 0 0 0 0],W,units,ddb);
  % FIXME: unit 8 distorted
  a = test(a,[1 -1 1 -1 1 1 -1 1],W,units,ddb);
end

function A=activations(a,units)
  A = ones(units,'single') .* repmat(a,units,1);
  A = A - diag(diag(A)); % 0 -> diagonal
end

function display (ticks, e, a)
  fprintf('\te ');
  fprintf('%6.2f',e(:));
  fprintf('\n');
  fprintf('\ta ');
  fprintf('%6.2f',a(:));
  fprintf('\n\n');
end

function newa=test(a,e,W,units,ddb)
  E = single(.99);             % excitation
  D = single(.99);             % decay
  max_ticks = single(50);     % maximum iterations for activations to stabilise

  for tick = 1:max_ticks      % ticks to stable activation
    % phase 1: calculate net inputs for units
    A = activations(a,units); % activations as a matrix
    A = A .* W;               % weighted activations
    n = sum(A,2) + e';        % net inputs 
    if (ddb > 1)
      %fprintf('e = %s\n',mat2str(e,2));
      fprintf('a = %s\n',mat2str(a,2));
      %fprintf('sum(A,2) = %s\n',mat2str(sum(A,2)));
      %fprintf('net activations = %s\n',mat2str(n));
    end
  
    % phase 2: update unit activations
    for i = 1:units
      if (n(i) > 0)
        d = E * n(i) * (1 - a(i)) - D * a(i);
      else % ni <= 0
        d = E * n(i) * (a(i) + 1) - D * a(i);
      end
      if (ddb > 1)
        fprintf('d = %.2f\n',d);
      end
      a(i) = a(i) + d;
    end
  end
  display(tick,e,a);
  newa = a;
end
