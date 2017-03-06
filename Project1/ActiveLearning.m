function selected = ActiveLearning(X_train, X_test, lambda, sigma2, varargin)

    X = X_train;
    D = X_test;
    selected = zeros(1, 10);
    n_features = size(X, 2);

    I = eye(n_features);

    for i = 1:10
        idx_max = NaN;
        sigma02_max = -Inf;
        for idx = 1:size(D, 1)
            if ~ismember(idx, selected)
                x0 = D(idx, :);
                cov = pinv(lambda*I + 1.0/sigma2*x0*x0' + X'*X);
                sigma02 = sigma2 + x0* (cov* x0');
                if sigma02 > sigma02_max
                    sigma02_max = sigma02;
                    idx_max = idx;
                end
            end
        end

        selected(i) = idx_max;
        X = [X; D(idx_max, :)];
    end 
      
endfunction
