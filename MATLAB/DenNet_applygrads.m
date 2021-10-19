function DenNet = DenNet_applygrads(DenNet)
%DenNetAPPLYGRADS updates weights and biases with calculated gradients
% DenNet = DenNetapplygrads(DenNet) returns an neural network structure with updated weights with calculated gradients
% The strategy can be SGD, ADAM, Momentum, etc

    lr = 1e+1;
    for i = 1:DenNet.layers-1
        DenNet.synapse{i} = DenNet.synapse{i} - lr * DenNet.dw{i};
    end
end
