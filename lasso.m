function h = lasso

%% Problem data

% Read the image
image = imread('cameraman.jpg');


% Define the region of interest (ROI)
x_start = 100; % Starting x-coordinate of the ROI
y_start = 50;  % Starting y-coordinate of the ROI
width = 64;    % Width of the ROI
height = 64;   % Height of the ROI

% Get the patch
[patch, orig_patch] = noisy_patch(image, x_start, y_start, width, height, 0);


[rows, cols, numberOfColorChannels] = size(patch);
% Reshape the image matrix into a vector
b = reshape(patch, [], 1);

m = length(b);      % 
n = length(b);      % 

x0 = sprandn(n,1,0.05);
A = randn(m, n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns

fprintf('solving instance with %d examples, %d variables\n', m, n);

gamma_max = norm(A'*b,'inf');
gamma = 1e-20*gamma_max;

% cached computations for all methods
AtA = A'*A;
Atb = A'*b;

%% Global constants and defaults

MAX_ITER = 10000;
ABSTOL   = 1e-6;

% %% CVX
% 
% tic
% 
% cvx_begin quiet
%     cvx_precision low
%     variable x(n)
%     minimize(0.5*sum_square(A*x - b) + gamma*norm(x,1))
% cvx_end
% 
% h.x_cvx = x;
% h.p_cvx = cvx_optval;
% h.cvx_toc = toc;

%% Proximal gradient

f = @(u) 0.5*sum_square(A*u-b);
lambda = 0.1;
beta = 0.5;
n = 10;
ista_psnr = zeros(n);
fista_psnr = zeros(n);

    [x_prox, p_prox, time_ista] = ista(f, x0, A, b, AtA, Atb, lambda,gamma, beta, MAX_ITER, ABSTOL);
    


    [x_fast, p_fast, time_fista] = fista(f, x0, A, b, AtA, Atb, lambda,gamma, beta, MAX_ITER, ABSTOL);
    




%% Timing

%fprintf('Proximal gradient time elapsed: %.2f seconds.\n', time_ista);
%fprintf('Fast prox gradient time elapsed: %.2f seconds.\n', time_fista);

%% Plots

% Display the original and noisy images
recovered_ista = reshape(A * x_prox, rows, cols, numberOfColorChannels);
recovered_fista = reshape(A * x_fast, rows, cols, numberOfColorChannels);
%recovered_cvx = reshape(A * h.x_cvx, rows, cols);

% PSNRs
psnr_ista = psnr(recovered_ista, orig_patch);
psnr_fista = psnr(recovered_fista, orig_patch);
%psnr_cvx = psnr(recovered_cvx, image_double);

figure;
subplot(2, 2, 1);
imshow(orig_patch);
title('Original Patch');

subplot(2, 2, 2);
imshow(patch);
title('Noisy Patch');

subplot(2, 2, 3);
imshow(recovered_ista);
title(['Recovered Image (ISTA p). PSNR: ' num2str(psnr_ista)]);

subplot(2, 2, 4);
imshow(recovered_fista);
title(['Recovered Image (FISTA) PSNR: ' num2str(psnr_fista)]);
% x_axis = 1:n;
% figure;
% subplot(1, 2, 1);
% plot(x_axis, ista_psnr);
% title('ISTA');
% subplot(1, 2, 2);
% plot(x_axis, fista_psnr);
% title('FISTA');
end

function p = objective(A, b, gamma, x, z)
    p = 0.5*sum_square(A*x - b) + gamma*norm(z,1);
end


function x = prox_l1(y, lambda)
    x = sign(y) .* max(abs(y) - lambda, 0);
end
