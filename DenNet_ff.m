function DenNet = DenNet_ff(DenNet, x)
% DenNet = DenNet_ff(DenNet, x) returns an network with updated layer activations, including DenNet.dendrite and DenNet.soma

    m = size(x, 1);

    alpha = 50 * DenNet.nonlinearity / (1 - DenNet.nonlinearity);

    DenNet.soma{1} = x;

    % feedforward pass

    for i = 2 : DenNet.layers
        branch = max(DenNet.dendrites{i-1});
        DenNet.dendrite{i-1} = zeros(m, DenNet.somas(i), branch);
        w = DenNet.synapse{i-1} .* (1 + 0.0*randn(size(DenNet.synapse{i-1})));
        for j = 1:branch
            mask = DenNet.dendrites{i-1} == j;
            DenNet.dendrite{i-1}(:, :, j) = dendrite_model(DenNet.soma{i-1}(:, mask) * w(mask, :), alpha);
        end
        DenNet.soma{i} = soma_model(sum(DenNet.dendrite{i-1} .* (1 + 0.00*randn(size(DenNet.dendrite{i-1}))), 3), DenNet.threshold);

    end
end



