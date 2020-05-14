%% Data preprocess
dataDir = 'E:\Learn\Data\SVHN\';
if ~exist('train_x', 'var') || ~exist('train_y', 'var')
    load([dataDir, 'train_32x32.mat']);
    m = length(y);  n = max(y);
    train_x = double(permute(X, [4, 1, 2, 3]))/255;
    train_x = train_x(:, :);
    train_y = zeros(m, n);
    train_y(m * (y-1) + (1:m)') = 1;
    clear X y m n;
end

if ~exist('test_x', 'var') || ~exist('test_y', 'var')
    load([dataDir, 'test_32x32.mat']);
    test_x = double(permute(X, [4, 1, 2, 3]))/255;
    test_x = test_x(:, :);
    test_answer = y;
    clear X y m n;
end

%% Setup the network
inputSize = 32*32*3;
hiddenSize = 500;
outputSize = 10;
temp = zeros(32, 32, 3);
i1 = 10; 		i2 = 22;
for i = 1:3
    temp(:,    1:i1,  i) = 3*i - 2;
    temp(:, i1+1:i2,  i) = 3*i - 1;
    temp(:, i2+1:end, i) = 3*i;
end
dendrites{1} = temp(:);
dendrites{2} = kron(1:4, ones(1, hiddenSize/4))';

DenNet = DenNet_setup([3072 500 10], dendrites);

DenNet.threshold = 0.1;
DenNet.nonlinearity = 0.1;

%% train the network
DenNet.train_opts.numepochs = 5e+1;                %  Number of full sweeps through data
DenNet.train_opts.batchsize = 1e+3;                %  Take a mean gradient step over this many samples

DenNet = DenNet_train(DenNet, train_x, train_y);

%% test
[er, bad] = DenNet_test(DenNet, test_x, test_answer);
