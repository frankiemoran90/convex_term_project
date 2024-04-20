function [x_prox,p_prox, time] = ista(f, x0,A, b, AtA, Atb, lambda, gamma, beta, MAX_ITER, ABSTOL)
    
    prox_optval = zeros(1, MAX_ITER);
    tic;
    
    x = x0;    
    for k = 1:MAX_ITER
        while 1
            grad_x = AtA*x - Atb;
            z = prox_l1(x - lambda*grad_x, lambda*gamma);
            if f(z) <= f(x) + grad_x'*(z - x) + (1/(2*lambda))*sum_square(z - x)
                break;
            end
            lambda = beta*lambda;
        end
        x = z;
    
        prox_optval(k) = objective(A, b, gamma, x, x);
        if k > 1 && abs(prox_optval(k) - prox_optval(k-1)) < ABSTOL
            break;
        end
    end
    
    x_prox = x;
    p_prox = prox_optval(end);
    time = toc;
end

function x = prox_l1(y, lambda)
    x = sign(y) .* max(abs(y) - lambda, 0);
end

function p = objective(A, b, gamma, x, z)
    p = 0.5*sum_square(A*x - b) + gamma*norm(z,1);
end