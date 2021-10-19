function y = dendrite_model(x, alpha)
%     persistent maxValue;
%     if isempty(maxValue)
%         maxValue = 1e-8;
%     end
%     maxValue = max(maxValue, max(x(:)));
%     x = max(x, 0) / maxValue;
    x = max(x, 0);
    x = bsxfun(@rdivide, x, max(max(x, [], 2), 1));
    if alpha <= 0
        y = x;
    else
        y = (exp(alpha*x) - 1) / (exp(alpha) - 1);
    end
end
