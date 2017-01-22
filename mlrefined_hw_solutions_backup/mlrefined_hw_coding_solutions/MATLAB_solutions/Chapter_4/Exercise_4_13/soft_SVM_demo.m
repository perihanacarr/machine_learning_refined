function soft_SVM_demo()
% soft_SVM_demo runs the perceptron model on a separable two 
% class dataset consisting of two dimensional data features. 
% The perceptron is run 3 times with 3 different values of lambda to show the 
% recovery of different separating boundaries.  All points and recovered
% as well as boundaries are then visualized.

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


%%% load data %%%
[X,y] = load_data();

%%% run perceptron for 3 lambda values %%%

% Calculate fixed steplength - via Lipschitz constant (see Chap 9 for 
% explanation) - for use in all four runs

lam = [1e-2 1e-1 1 10];   % regularization parameter 
w0 = randn(3,1);    % initial point
L = 2*norm(diag(y)*X')^2;
W = [];     % container for learned parameters in each case

for i = 1:length(lam)          
    alpha = 1/(L + 2*lam(i));        % step length
    W(:,i) = grad_descent_soft_SVM(X,y,w0,alpha,lam(i)); % Run perceptron 
    leg{i} = strcat('lam = ',num2str(lam(i)));
end

%%% plot everything, pts and lines %%%
plot_all(X',y,W);
legend('','',leg{1},leg{2},leg{3},leg{4});

%%% gradient descent function for perceptron %%%
    function w = grad_descent_soft_SVM(X,y,w0,alpha,lam)
    % Initializations 
    w = w0;
    H = diag(y)*X';
    l = ones(size(X,2),1);
    iter = 1;
    max_its = 3000;
    grad = 1;

    while  norm(grad) > 10^-6 && iter < max_its
        
        % form gradient and take step
        grad = 2*lam*[0;w(2:end)] - 2*H'*max(l - H*w,0);
        w = w - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(X,y,w1,w2,w3)
    
    % plot points 
    ind = find(y == 1);
    scatter(X(ind,2),X(ind,3),'Linewidth',2,'Markeredgecolor','b','markerFacecolor','none');
    hold on
    ind = find(y == -1);
    scatter(X(ind,2),X(ind,3),'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');
    hold on

    % plot separators
    colors = ['k','g','m','c'];
    s =[min(X(:,2)):.01:max(X(:,2))];
    for i = 1:size(W,2)
        plot (s,(-W(1,i)-W(2,i)*s)/W(3,i),colors(i),'linewidth',2);
        hold on
    end
   
    
    % graph info labels
    set(gcf,'color','w');
    box off
    xlabel('x_1','Fontsize',14)
    ylabel('x_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)

end

%%% loads data %%%
function [X,y] = load_data()
    data = load('imbalanced_2class.csv');
    X = data(:,1:end-1);
    X = [ones(size(X,1),1) X]';
    y = data(:,end);
end

end
