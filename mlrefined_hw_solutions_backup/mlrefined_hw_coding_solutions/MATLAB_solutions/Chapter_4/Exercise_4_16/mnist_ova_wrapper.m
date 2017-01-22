function mnist_ova_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


[X,y] = load_data();        % load data

% find separators
W = learn_separators(X,y);  % learn separators
num = misclass(X,y,W)       % # of misclassifications

% determine how many misclassifications on the test set
data_test = csvread('MNIST_testing_data.csv');
X = data_test(:,1:end - 1);
y = data_test(:,end);
X = [ones(size(X,1),1), X];
X = X';
num = misclass(X,y,W)

%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%%
%%% newton's method for the softmax cost %%%
function w = softmax_newtons_method(X,y,max_its)
    % precomputations
    H = bsxfun(@times,X',y);
    w = randn(size(X,1),1)/size(X,1);
     
    % add a little to the eigenvalues of the Hessian to avoid machine
    % precision problems
    I = eye(size(X,1));
    lam = 0;
    
    %%% main %%%
    k = 1;
    while k < max_its
        
        % prep gradient for logistic objective
        r = sigmoid(-H*w);
        g = r.*(1 - r);
        grad = -H'*r;
        hess = bsxfun(@times,g,X')'*X' + lam*I;

        % take Newton step
        temp = hess*w - grad;
        w = pinv(hess)*temp;
   
        k = k + 1;
    end
    
    function t = sigmoid(z)
        t = 1./(1 + exp(-z));
    end
    
    function s = my_exp(input)
        a = find(input > 100);
        b = find(input < -100);
        input(a) = 0;
        input(b) = 0;
        s = exp(input);
        s(b) = 1;
    end
end

%%% learn separator for each class %%%
function W = learn_separators(X,y)  
    class_labels = unique(y);           % class labels
    num_classes = length(unique(y));
    W = [];         % container for all the weights to learn

    for i = 1:num_classes
        i
        % setup temporary labels for one-vs-all classification       
        y_temp = y;
        class = class_labels(i);
        ind = find(y_temp == class);
        ind2 = find(y_temp ~= class);
        y_temp(ind) = 1;
        y_temp(ind2) = -1;
        
        % run newton's method and store resulting weights
        max_its = 10;
        w = softmax_newtons_method(X,y_temp,max_its);
        W = [W, w];
    end
end

%%% count number of current misclassifications %%%
function num = misclass(X,y,W)
    [vals,y_predict] = max(X'*W,[],2);
    matches = y - y_predict;
    ind = find(matches ~= 0);
    num = length(ind);
end

%%% load data %%%
function [X,y] = load_data()
    % load data from file
    data = csvread('MNIST_training_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);
    
    X = [ones(size(X,1),1), X];
    X = X';
end
end