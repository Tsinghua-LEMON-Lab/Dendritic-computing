function DenNet = DenNet_setup(somas, dendrites)
% DenNet_SETUP creates a Feedforward Backpropagate Neural Network
% DenNet = DenNet_setup(somas, dendrites) returns an dendritic neural network structure with n=numel(somas) layers
% "somas" being a n x 1 vector of the number of somas in each layer, e.g. [784 100 10]
% "dendrites" being a n-1 x 1 cell, which defines the group of somas in each layer except the input layer

    DenNet.somas        = somas;
    DenNet.dendrites    = dendrites;
    DenNet.layers		= numel(somas);
    DenNet.threshold 	= 0.1;
    DenNet.nonlinearity = 0.1;

    DenNet.synapse = cell(DenNet.layers-1, 1);
    DenNet.dendrite = cell(DenNet.layers-1, 1);
    DenNet.soma = cell(DenNet.layers, 1);

    for l = 1 : DenNet.layers-1
        DenNet.synapse{l} = 8 * (rand(DenNet.somas(l), DenNet.somas(l+1)) - 0.5) * sqrt(6 / (DenNet.somas(l) + DenNet.somas(l+1)));
    end
end
