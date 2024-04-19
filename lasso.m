function h = lasso

%% Problem data

% Read the image
image = imread('cameraman.jpg');


% Define the region of interest (ROI)
x_start = 50; % Starting x-coordinate of the ROI
y_start = 50;  % Starting y-coordinate of the ROI
width = 128;    % Width of the ROI
height = 128;   % Height of the ROI

% Extract the ROI from the image
patch = image(y_start:y_start+height-1, x_start:x_start+width-1, :);
[rows, cols, numberOfColorChannels] = size(patch);

if (numberOfColorChannels == 3)
    patch = rgb2gray(patch);
    numberOfColorChannels = 1;
end

% Add Gaussian noise
% Create the Gaussian filter kernel using fspecial\
image_double = im2double(patch);
gaussian_kernel = fspecial('gaussian', [3 3], 4);
blurred_patch = imfilter(image_double, gaussian_kernel, 'conv', 'replicate');

noisy_patch = imnoise(blurred_patch, 'gaussian', 0, (1e-6));

% Normalize the image
min_value = min(image_double(:));
max_value = max(image_double(:));
normalized_patch = (noisy_patch - min_value) / (max_value - min_value);
% Reshape the image matrix into a vector
b = reshape(normalized_patch, [], 1);



m = length(b);      % 
n = length(b);      % 

x0 = sprandn(n,1,0.05);
A = randn(m, n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns

fprintf('solving instance with %d examples, %d variables\n', m, n);

gamma_max = norm(A'*b,'inf');
gamma = 0.00001*gamma_max;

% cached computations for all methods
AtA = A'*A;
Atb = A'*b;

%% Global constants and defaults

MAX_ITER = 1000;
ABSTOL   = 1e-8;

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
beta = 0.1;

tic;

x = x0;
xprev = x;

for k = 1:MAX_ITER
    while 1
        grad_x = AtA*x - Atb;
        z = prox_l1(x - lambda*grad_x, lambda*gamma);
        if f(z) <= f(x) + grad_x'*(z - x) + (1/(2*lambda))*sum_square(z - x)
            break;
        end
        lambda = beta*lambda;
    end
    xprev = x;
    x = z;

    h.prox_optval(k) = objective(A, b, gamma, x, x);
    if k > 1 && abs(h.prox_optval(k) - h.prox_optval(k-1)) < ABSTOL
        break;
    end
end

h.x_prox = x;
h.p_prox = h.prox_optval(end);
h.prox_grad_toc = toc;

%% Fast proximal gradient

lambda = 0.1;

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

h.x_fast = x;
h.p_fast = h.fast_optval(end);
h.fast_toc = toc;

%% Timing

fprintf('Proximal gradient time elapsed: %.2f seconds.\n', h.prox_grad_toc);
fprintf('Fast prox gradient time elapsed: %.2f seconds.\n', h.fast_toc);

%% Plots

% Display the original and noisy images
recovered_ista = reshape(A * h.x_prox, rows, cols, numberOfColorChannels);
recovered_fista = reshape(A * h.x_fast, rows, cols, numberOfColorChannels);
%recovered_cvx = reshape(A * h.x_cvx, rows, cols);

% PSNRs
psnr_ista = psnr(recovered_ista, normalized_patch);
psnr_fista = psnr(recovered_fista, normalized_patch);
%psnr_cvx = psnr(recovered_cvx, image_double);


figure;
subplot(2, 2, 1);
imshow(patch);
title('Original Patch');

subplot(2, 2, 2);
imshow(normalized_patch);
title('Noisy Patch');

subplot(2, 2, 3);
imshow(recovered_ista);
title(['Recovered Image (ISTA). PSNR: ' num2str(psnr_ista)]);

subplot(2, 2, 4);
imshow(recovered_fista);
title(['Recovered Image (FISTA) PSNR: ' num2str(psnr_fista)]);


end

function p = objective(A, b, gamma, x, z)
    p = 0.5*sum_square(A*x - b) + gamma*norm(z,1);
end


function x = prox_l1(y, lambda)
    x = sign(y) .* max(abs(y) - lambda, 0);
end
