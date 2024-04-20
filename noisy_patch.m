function [patch, orig_patch] = noisy_patch(image,x_start, y_start, width, height, color)
    % Extract the ROI from the image
    patch = image(y_start:y_start+height-1, x_start:x_start+width-1, :);
    
    if (color == 0)
        patch = rgb2gray(patch);
    end
    
    orig_patch = im2double(patch);
    % Add Gaussian noise
    % Create the Gaussian filter kernel using fspecial\
    patch = im2double(patch);
    gaussian_kernel = fspecial('gaussian', [3 3], 4);
    patch = imfilter(patch, gaussian_kernel, 'conv', 'replicate');
    patch = imnoise(patch, 'gaussian', 0, (1e-6));
    
    % Normalize the image
    min_value = min(patch(:));
    max_value = max(patch(:));
    patch = (patch - min_value) / (max_value - min_value);
end

