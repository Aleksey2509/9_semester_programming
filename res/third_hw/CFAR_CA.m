function [targets_positions] = CFAR_CA(input_matrix)

    P_fa = 1e-6;

    window_size = 13;
    
    % Construct a convolution kernel:
    % We want just an averaging kernel, so something like matrix
    % window_size x windows_size with elements being equal to 1 /
    % window_size ^ 2. But we also want to not include closest to current
    % point values, to be able to differentiate several nearby targets
    midpoint = ceil(window_size / 2);
    averaging_kernel = ones(window_size);
    guard_range = 5; % for excluding nearest to current point from averaging
    averaging_kernel(midpoint - guard_range:midpoint + guard_range, ...
                     midpoint - guard_range:midpoint + guard_range) = 0;
    averaging_kernel_sum = sum(averaging_kernel(:));
    threshold = averaging_kernel_sum * (P_fa ^ (- 1 / averaging_kernel_sum) - 1)
    averaging_kernel = averaging_kernel ./ averaging_kernel_sum;

    % Do the convolution. Need to extend the matrix, but also, in newly
    % added symbols need to create similar noise, as in original. For this,
    % periodized extension is used
    extend_size = floor(window_size / 2);
    extended_matrix = wextend(2, 'ppd', input_matrix, extend_size, 'bb');
    convoluted_mat = conv2(extended_matrix, averaging_kernel, 'valid');

    bool_detected = input_matrix > (convoluted_mat * threshold);
    input_matrix(bool_detected <= 0) = 0;
    targets_positions = find_targets(input_matrix, guard_range);
end


function targets = find_targets(filtered_mat, guard_range)
    targets = [];
    column_amount = size(filtered_mat, 1);
    while any(filtered_mat(:))
            [~, max_index] = max(filtered_mat(:));
            column_index = ceil(max_index / column_amount);
            row_index = mod(max_index, column_amount);
            if row_index == 0
                row_index = column_amount;
            end
            targets = [targets; row_index, column_index];
            filtered_mat(row_index-guard_range:row_index+guard_range, ...
                         column_index-guard_range:column_index+guard_range) = 0;
    end
end
