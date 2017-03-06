function wRR = RidgeRegression(X, y, lambda, varargin)

    n_features = size(X, 2);
    centering = false;
    standarize = false;
    
    if nargin > 5 || nargin < 2
        error('Incorrect number of inputs')
    end

    if nargin >= 4
        centering = varargin{1}
        if nargin == 5
            standarize = varargin{2}
        end

    end

    if centering
        y = y - mean(y);
        X = (X - mean(X));
    end

    if standarize
        X = X / var(X, 1);
    end

    I = eye(n_features);

    wRR = (lambda*I + X'*X)\X'*y;

endfunction
