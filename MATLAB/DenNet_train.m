function DenNet = DenNet_train(DenNet, x, y)
% DenNet = DenNet_train(DenNet, x, y) trains the neural network DenNet with input x and output y
% Returns a neural network DenNet with updated weights, activations, and loss

    [~, train_answer] = max(y, [], 2);
    m = size(x, 1);

    ep = DenNet.train_opts.numepochs;
    bs = DenNet.train_opts.batchsize;

    ba = floor(m / bs);

    DenNet.loss = zeros(ep * ba, 1);
    n = 1;
    tic;
    for i = 1 : ep
        r = randperm(m);
        for l = 1 : ba
            batch_x = x(r((l-1) * bs + 1 : l * bs), :);
            batch_y = y(r((l-1) * bs + 1 : l * bs), :);

            DenNet = DenNet_ff(DenNet, batch_x);
            DenNet = DenNet_bp(DenNet, batch_y);
            DenNet = DenNet_applygrads(DenNet);

            DenNet.loss(n) = .5 * sum(sum((batch_y - DenNet.soma{end}).^2)) / bs;

            n = n + 1;
        end

        if ~mod(i, min(10, ceil(ep/10)))
            t = toc;
            er = DenNet_test(DenNet, x, train_answer);
            message = ['epoch ' num2str(i) '/' num2str(ep) '.', ...
                       ' Took ', num2str(t), ' seconds', ...
                       '. Mini-batch mean loss is ', num2str(mean(DenNet.loss(n-ba:n-1))), ... 
                       '. Training accuracy is ', num2str(1 - er)];
            disp(message);
            tic;
        end

    end
end

