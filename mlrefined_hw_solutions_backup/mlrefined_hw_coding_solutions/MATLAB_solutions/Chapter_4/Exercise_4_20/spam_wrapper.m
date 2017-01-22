function spam_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% load data
[X,y] = load_data();

% run iterative grad
run_comparison(X,y);

%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%
%%% compare grad descents %%% 
function run_comparison(X_orig,y)
    X = X_orig;
    
    % containers for comparison
    MIS_soft = [];
    OBJ_soft = [];
    MIS_squa = [];
    OBJ_squa = [];
    
    % compare newtons' num_runs number of times
    max_its = 10;

    % remove the non BoW features
    X = X(:,1:48);
    X = [ ones(size(X,1),1), X];
    X = X';

    % initialize weights for both algorithms
    w0 = randn(size(X,1),1)/size(X,1);

    % run softmax
    [mis,obj,w] = softmax_newtons_method(X,y,w0,max_its);
    plot_it(mis,obj,'k',2)

    % keep char BoW features 
    X = X_orig;
    X = X(:,1:54);
    X = [ ones(size(X,1),1), X];
    X = X';

    % initialize weights for both algorithms
    w0 = randn(size(X,1),1)/size(X,1);
    [mis,obj,w] = softmax_newtons_method(X,y,w0,max_its);
    plot_it(mis,obj,'g',2)
    
    
    % keep char BoW features and spam features and take logs of big numbers
    X = X_orig;
    X(:,end) = log(X(:,end));
    X(:,end - 1) = log(X(:,end - 1));
    X = [ ones(size(X,1),1), X];
    X = X';

    % initialize weights for both algorithms
    w0 = randn(size(X,1),1)/size(X,1);
    [mis,obj,w] = softmax_newtons_method(X,y,w0,max_its);
    plot_it(mis,obj,'m',2)


    % plot mean result
    legend('BoW','BoW + chars', 'BoW + chars + spam feats')
end

function [mis,obj,w] = softmax_newtons_method(X,y,w,max_its)
    % precomputations
    H = bsxfun(@times,X',y);
    mis = [];
    obj = [];
    
    % caclulate number of misclassifications at current iteration
    s = X'*w;
    t = sum(max(0,sign(-y.*s)));
    mis = [mis; t];

    % calculate objective value at current iteration
    s = y.*s;
    s = sum(log(1 + exp(-s))); 
    obj = [obj; s];  
        
    k = 1;
    %%% main %%%
    while k < max_its
        
        % prep gradient for logistic objective
        r = sigmoid(-H*w);
        g = r.*(1 - r);
        grad = -H'*r;
        hess = bsxfun(@times,g,X')'*X';

        % take Newton step
        temp = hess*w - grad;
        w = pinv(hess)*temp;
        
        % caclulate number of misclassifications at current iteration
        s = X'*w;
        t = sum(max(0,sign(-y.*s)));
        mis = [mis; t];

        % calculate objective value at current iteration
        s = y.*s;
        s = sum(log(1 + exp(-s))); 
        obj = [obj; s];  
        k = k + 1;
    end
    
    function t = sigmoid(z)
        t = 1./(1 + my_exp(-z));
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

%%% gradient descent function for squared hinge loss %%%
function [mis,obj,w] = squared_hinge_newtons_method(X,y,w,max_its)
    % Initializations 
    l = ones(size(X,2),1);

    % precomputations
    H = bsxfun(@times,X',y);
    mis = [];
    obj = [];
    
    % caclulate number of misclassifications at current iteration
    s = X'*w;
    t = sum(max(0,sign(-y.*s)));
    mis = [mis; t];

    % calculate objective value at current iteration
    s = y.*s;
    s = sum(max(0,1 - s).^2); 
    obj = [obj; s];  
    k = 1;
    
    while k < max_its
        
        % form gradient
        temp = l-H*w;
        grad = -2*H'*max(l-H*w,0);
        
        % form Hessian
        s = find(temp > 0);
        X_temp = X(:,s);
        hess = 2*(X_temp*X_temp') + 10^-3*eye(size(X_temp,1));
        
        % take Newton step
        temp = hess*w - grad;
%         w = pinv(hess)*temp;
        w = linsolve(hess,temp);
        
        % caclulate number of misclassifications at current iteration
        s = X'*w;
        t = sum(max(0,sign(-y.*s)));
        mis = [mis; t];

        % calculate objective value at current iteration
        s = y.*s;
        s = sum(max(0,1 - s).^2); 
        obj = [obj; s];  
        k = k + 1;
    end
end

function [mis,obj,w] = adaptive_grad(X,y,w,max_its)
    H = X*diag(y);
    
    % containers
    s = X'*w;
    t = sum(max(0,sign(-y.*s)))
    mis = [ t];
    obj = [-sum(log(sigmoid(H'*w)))];   
    
    % main loop
    for iter = 1:max_its 
        iter
        % form gradient and take accelerated step
        grad = - (H*(sigmoid(-H'*w)));
        alpha = adaptive_step(w,grad,-sum(log(sigmoid(H'*w))));
        w = w - alpha*grad;

        % update containers
        s = X'*w;
        t = sum(max(0,sign(-y.*s)));
        mis = [mis; t];
        obj = [obj ;-sum(log(sigmoid(H'*w)))]; 
    end

    % adaptive step length selection function
    function p = adaptive_step(z,g,old_obj)
        g_n = norm(g)^2;
        step_l = 1;
        step_r = 0;
        u = 1;
        p = 1;
        while step_l > step_r && u < 20
            p = p*0.5;
            
            % left
            new_obj = -sum(log(sigmoid(H'*(z - p*g))));
            step_l = new_obj - old_obj;
            
            % right 
            step_r = -(p*g_n)/2;
            u = u + 1;
        end
    end

    % sigmoid function
    function y = sigmoid(z)
    y = 1./(1+exp(-z));
    end

end


function [mis,obj,w] = iterative_grad(X,y,w,max_its)
    % make random shuffle of data
    [K,N] = size(X);
    M = randperm(N);
    H = X*diag(y);

    % record stuff
    s = X'*w;
    t = sum(max(0,sign(-y.*s)))
    mis = [ t];
    obj = [-sum(log(sigmoid(H'*w)))];  
    
    for k = 1:max_its
        k
        alpha = 1/k;
        for i = 1:N
            % take incremental grad step
            j = M(i);
            grad = - sigmoid(-y(j)*X(:,j)'*w)*y(j)*X(:,j);
            w = w - alpha*grad;
%             w(2:end) = proj(w(2:end));
        end 
        % update containers
        s = X'*w;
        t = sum(max(0,sign(-y.*s)));
        mis = [mis; t];
        obj = [obj ;-sum(log(sigmoid(H'*w)))]; 
    end
    
    function t = sigmoid(z)
        t = 1./(1+exp(-z));
    end
end

%%% plots descent levels %%%
function plot_it(mis,obj,color,width)
    % plot all
    subplot(1,2,1)
    start = 2;
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
    
    % last adjustments to plot    
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

%%% load data %%%
function [X, y] = load_data()    
      
    %%% load spam database data %%%
    data = csvread('spambase_data.csv');
    X = data(:,1:end-1);  
    y = data(:,end);
    ind = find(y == 1);
    length(ind)
    ind = find(y == -1);
    length(ind)
end


end