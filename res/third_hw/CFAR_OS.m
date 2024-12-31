function [targets_positions] = CFAR_OS(input_matrix)

    window_size = 9;

    % Construct a convolution kernel:
    % We want just an averaging kernel, so something like matrix
    % window_size x windows_size with elements being equal to 1 /
    % window_size ^ 2. But we also want to not include closest to current
    % point values, to be able to differentiate several nearby targets
    midpoint = ceil(window_size / 2);
    averaging_kernel = ones(window_size);
    guard_range = 3; % for excluding nearest to current point from averaging
    averaging_kernel(midpoint - guard_range:midpoint + guard_range, ...
                     midpoint - guard_range:midpoint + guard_range) = 0;
    averaging_kernel_sum = sum(averaging_kernel(:));

    % Do the convolution. Need to extend the matrix, but also, in newly
    % added symbols need to create similar noise, as in original. For this,
    % periodized extension is used
    extend_size = floor(window_size / 2);
    extended_matrix = wextend(2, 'ppd', input_matrix, extend_size, 'bb');

    convoluted_mat = nlfilter(extended_matrix, [window_size, window_size], ...
                          @(x) window_fun(x, averaging_kernel));
    threshold = 10000;
    % Have to cutoff the extended parts by hand
    convoluted_mat = convoluted_mat(extend_size + 1:end - extend_size, ...
                                    extend_size + 1:end - extend_size);

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

function val = window_fun(mat_block, averaging_kernel)
    k = floor(3 / 4 * sum(averaging_kernel(:))); % such k value is advised in papers

    mat_block = mat_block .* averaging_kernel;
    maxes = maxk(mat_block(:), k);
    val = maxes(end);
end
