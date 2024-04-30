% deblurring of the camera image using ISTA, FISTA, and SpaRSA.
close all
clear
clf

f = double(imread('cameraman.tif'));
[m n] = size(f);

scrsz = get(0,'ScreenSize');

% create observation operator; in this case 
% it will be a blur function composed with an
% inverse weavelet transform
disp('Creating observation operator...');

middle = n/2 + 1;

sigma = sqrt(3);
h = zeros(size(f));
for i=-4:4
   for j=-4:4
      h(i+middle,j+middle)= (1/(1+i*i+j*j));
   end
end


% % center and normalize the blur
h = fftshift(h);   
h = h/sum(h(:));

% define the function handles that compute
% the blur and the conjugate blur.
R = @(x) real(ifft2(fft2(h).*fft2(x)));
RT = @(x) real(ifft2(conj(fft2(h)).*fft2(x)));

% define the function handles that compute 
% the products by W (inverse DWT) and W' (DWT)
wav = daubcqf(2);
W = @(x) midwt(x,wav,3);
WT = @(x) mdwt(x,wav,3);

%Finally define the function handles that compute 
% the products by A = RW  and A' =W'*R' 
A = @(x) R(W(x));
AT = @(x) WT(RT(x));

% generate noisy blurred observations
y = R(f) + sigma*randn(size(f));

% regularization parameter
tau = 0.035;

% set tolA
tolA = (120000);
max_iters = 200;
% Run IST until the relative change in objective function is no
% larger than tolA
[theta_ist,theta_debias,obj_IST,times_IST,debias_s,mses_IST]= ...
    IST(y,A,tau,max_iters,...
    'Debias',0,...
    'AT',AT,... 
    'True_x',WT(f),...
    'Initialization',AT(y),...
    'StopCriterion',4,...
    'ToleranceA',tolA);

% Run FISTA until the relative change in objective function is no
% larger than tolA
[theta_fista,theta_debias,obj_FISTA,times_FISTA,debias_s_fista,mses_FISTA]= ...
    FISTA(y,A,tau, max_iters,...
    'Debias',0,...
    'AT',AT,... 
    'True_x',WT(f),...
    'Initialization',AT(y),...
    'StopCriterion',4,...
    'ToleranceA',tolA);

% Now, run SpaRSA, until they reach the same value
% of objective function reached by IST.
[theta,theta_debias,obj_SpaRSA,times_SpaRSA,debias_start,mses_SpaRSA]= ...
    SpaRSA(y,A,tau, max_iters,...
    'AT', AT,...
    'Debias',0,...
    'Initialization',AT(y),...
    'True_x',WT(f),...
    'BB_variant',1,...
    'BB_cycle',3,...
    'Monotone',1,...
    'StopCriterion',4,...
    'ToleranceA',tolA);

% ================= Plotting results ==========
figure(1)
subplot(1, 2, 1)
imagesc(f)
colormap(gray(255))
axis off
axis equal
title('Original Image','FontName','Times','FontSize',14)

subplot(1, 2, 2)
imagesc(y)
colormap(gray(255))
axis off
axis equal
title('Blurred Image','FontName','Times','FontSize',14)


figure(2)
% ISTA
subplot('Position',[0.05 0.1 0.3 0.8]);
imagesc(W(theta))
colormap(gray)
axis off
axis equal
title('ISTA','FontName','Times','FontSize',14)

% FISTA
subplot('Position',[0.375 0.1 0.3 0.8]);
imagesc(W(theta_ist))
colormap(gray)
axis off
axis equal
title('FISTA','FontName','Times','FontSize',14)

% SpaRSA
subplot('Position',[0.7 0.1 0.3 0.8]);
imagesc(W(theta))
colormap(gray)
axis off
axis equal
title('SpaRSA','FontName','Times','FontSize',14)

figure(3)
hold on 
plot(obj_IST,'b.','LineWidth',1.8)
plot(obj_FISTA,'r--','LineWidth',1.8)
plot(obj_SpaRSA,'g','LineWidth',1.8)
leg = legend('ISTA', 'FISTA', 'SpaRSA');
yscale('log')
%title('Function Value vs Iteration Number','FontName','Times','FontSize',14)
ylabel('Function Value','FontName','Times','FontSize',12');
xlabel('Iteration Number','FontName','Times','FontSize',12')
grid('on')
ax = gca; % Get current axes
ax.Box = 'on'; % Turn on the box around the axes

