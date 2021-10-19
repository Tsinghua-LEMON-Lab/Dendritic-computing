function DenNet = DenNet_bp(DenNet, y)
% DenNet = DenNet_bp(DenNet) returns an neural network structure with updated weights, that is DenNet.synapse

    err = y - DenNet.soma{end};

    d = cell(DenNet.layers-1, 1);
    d{end} = - err .* (1 + DenNet.soma{end} .* (1 - DenNet.soma{end}));

    DenNet.dw = cell(DenNet.layers-1, 1);

    for i = DenNet.layers-1 : -1 : 1
        branch = max(DenNet.dendrites{i});
        for j = 1:branch
            mask = DenNet.dendrites{i} == j;
            dd = d{i} .* DenNet.dendrite{i}(:, :, j);
            DenNet.dw{i}(mask, :) = DenNet.soma{i}(:, mask)' * dd / size(y, 1);
            if i > 1
                d{i-1}(:, mask) = dd * DenNet.synapse{i}(mask, :)';
            end
        end
        if i > 1
            d{i-1} = d{i-1} .* (1 + DenNet.soma{i} .* (1 - DenNet.soma{i}));
        end
    end

    DenNet = DenNet_applygrads(DenNet);
end
