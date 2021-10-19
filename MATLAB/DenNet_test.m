function [er, bad] = DenNet_test(DenNet, x, answer)
    DenNet = DenNet_ff(DenNet, x);
    [~, labels] = max(DenNet.soma{end}, [], 2);
    bad = find(labels ~= answer);
    er = numel(bad) / size(x, 1);
end
