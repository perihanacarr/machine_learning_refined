function stochastic_vs_standard_gradient_descent()

% stochastic_vs_standard_gradient_descent makes a comparison of standard vs
% stochastic gradient descent

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data
[X,y] = load_data();

% run algorithms
run_grads(X,y);

%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%
%%% compare grad descents %%% 
function run_grads(X,y)
    % params for all algs
    max_its = 100;
    num_runs = 10;
    
    % containers for all runs
    MIS_stan = [];
    OBJ_stan = [];
    TIME_stan = [];
    MIS_iter = [];
    OBJ_iter = [];
    TIME_iter = [];
    
    for r = 1:num_runs
        w0 = randn(size(X,1),1);
        
        %%% standard gradient descent w/Lipschitz fixed step-size
        tic;
        [w,mis,obj] = standard_grad(X,y,w0,max_its);
        TIME_stan = [TIME_stan, toc];
        MIS_stan = [MIS_stan, mis];
        OBJ_stan = [OBJ_stan, obj];
        plot_it(mis,obj,[0.7 .7 .7],1)
 

        %%% stochastic gradient descent
        tic;
        [w,mis,obj] = iterative_grad(X,y,w0,max_its);
        TIME_iter = [TIME_iter, toc];
        MIS_iter = [MIS_iter, mis];
        OBJ_iter = [OBJ_iter, obj];
        plot_it(mis,obj,[0.5 1 0.5],1)
    end
    
    plot_it(mean(MIS_stan,2),mean(OBJ_stan,2),'k',2)
    plot_it(mean(MIS_iter,2),mean(OBJ_iter,2),'g',2)
    
    %%% print out ave times
    s = 'Ave time of standard grad descent = ';
    s = [s, num2str(mean(TIME_stan))];
    disp(s)     
    
    s = 'Ave time of iterative grad descent = ';
    s = [s, num2str(mean(TIME_iter))];
    disp(s)   
    
    mean(TIME_iter);
    % last adjustments to plot
    legend('standard','iterative')
    
    subplot(1,2,1)
    axis tight
    set(gca,'FontSize',14)
    xlabel('iteration','FontSize',18)
    ylabel('number of misclassifications','FontSize',18)
    
    subplot(1,2,2)
    axis tight
    set(gca,'FontSize',14)
    xlabel('iteration','FontSize',18)
    ylabel('objective value','FontSize',18)
end

%%% plots descent levels %%%
function plot_it(mis,obj,color,width)
    % plot all
    subplot(1,2,1)
    start = 1;
    hold on
    plot(start:length(mis),mis(start:end),'Color',color,'LineWidth',width);
    title('# of misclassifications')
    box off
    
    subplot(1,2,2)
    hold on
    plot(start:length(obj),obj(start:end),'Color',color,'LineWidth',width);
    title('objective value')
    set(gcf,'color','w');
    box off
end

%%% iterative grad descent algorithm %%%
function [w,mis,obj] = iterative_grad(X,y,w,max_its)
    % make random shuffle of data
    [K,N] = size(X);
    M = randperm(N);
    H = X*diag(y);

    % record stuff
    mis = [];
    obj = [];
    
    for k = 1:max_its
        for i = 1:N
            % take incremental grad step
            j = M(i);
            grad = - sigmoid(-y(j)*X(:,j)'*w)*y(j)*X(:,j);
            alpha = 1/k;
            w = w - alpha*grad;
        end
        mis = [mis; evaluate(X',y,w)];
        obj = [obj ;-sum(log(sigmoid(H'*w)))];  
    end
    
    function t = sigmoid(z)
        t = 1./(1+exp(-z));
    end
end

%%% standard gradient descent w/fixed step length %%%
function [w,mis,obj] = fast_grad(X,y,w,max_its)
    % Initializations 
    L = norm(X)^2;
    alpha = 4/L;
    z = w;
    H = X*diag(y);
    
    % containers
    obj = [];
    mis = [];
    mis = [mis; evaluate(X',y,w)];
    obj = [obj ;-sum(log(sigmoid(H'*w)))];  
    
    % main loop
    for iter = 1:max_its 
        % form gradient and take accelerated step
        w0 = w;
        grad = - (H*(sigmoid(-H'*z)));
        w = z - alpha*grad;
        z = w + iter/(iter+3)*(w - w0);

        mis = [mis; evaluate(X',y,w)];
        obj = [obj ;-sum(log(sigmoid(H'*w)))];  
    end

    % sigmoid function
    function y = sigmoid(z)
    y = 1./(1+exp(-z));
    end

end

%%% standard gradient descent w/fixed step length %%%
function [w,mis,obj] = standard_grad(X,y,w,max_its)
    % Initializations 
    L = norm(X)^2;
    alpha = 4/L;
    H = X*diag(y);
    
    % containers
    obj = [];
    mis = [];
    
    % main loop
    for iter = 1:max_its 
        % form gradient and take accelerated step
        grad = - (H*(sigmoid(-H'*w)));
        w = w - alpha*grad;

        mis = [mis; evaluate(X',y,w)];
        obj = [obj ;-sum(log(sigmoid(H'*w)))];  
    end

    % sigmoid function
    function y = sigmoid(z)
    y = 1./(1+exp(-z));
    end


end


%%% loads data and normalizes etc., %%%
function [X, y] = load_data()    
      
    %%% real face dataset %%%
    data = csvread('face_data.csv');

    X = data(:,1:end-1);
    X = X';
    y = data(:,end);

end

function score = evaluate(A,b,x)
% compute score of trained model on test data

    s = A*x;
    ind = find(s > 0);
    s(ind) = 1;
    ind = find(s <= 0);
    s(ind) = -1;
    t = s.*b;
    ind = find(t < 0);
    score = length(ind);

end

end