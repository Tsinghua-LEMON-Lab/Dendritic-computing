function y = soma_model(x, threshold)
    y = max(2 * sigmoid(x - threshold) - 1, 0);
end
