function [W, lambda_max] = priority_vector(A)
    [V, D] = eig(A);
    lambda_max_all = diag(D);
    [lambda_max_val, max_i] = max(real(lambda_max_all));
    lambda_max = real(lambda_max_val);
    W = V(:, max_i);
    if W(1) < 0
        W = -W;
    end
    W = W / sum(W);
end

