function [x_fast, p_fast, time] = fista(f, x0,A, b, AtA, Atb, lambda, gamma, beta, MAX_ITER, ABSTOL)
    tic;

    x = x0;
    xprev = x;
    for k = 1:MAX_ITER
        y = x + (k/(k+3))*(x - xprev);
        while 1
            grad_y = AtA*y - Atb;
            z = prox_l1(y - lambda*grad_y, lambda*gamma);
            if f(z) <= f(y) + grad_y'*(z - y) + (1/(2*lambda))*sum_square(z - y)
                break;
            end
            lambda = beta*lambda;
        end
        xprev = x;
        x = z;
    
        h.fast_optval(k) = objective(A, b, gamma, x, x);
        if k > 1 && abs(h.fast_optval(k) - h.fast_optval(k-1)) < ABSTOL
            break;
        end
    end
    
    x_fast = x;
    p_fast = h.fast_optval(end);
    time = toc;

end

function x = prox_l1(y, lambda)
    x = sign(y) .* max(abs(y) - lambda, 0);
end

function p = objective(A, b, gamma, x, z)
    p = 0.5*sum_square(A*x - b) + gamma*norm(z,1);
end