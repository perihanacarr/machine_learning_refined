function raw_vs_feat()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% load data
[Xreal,Xfeat,y] = load_data();

% run iterative grad
run_comparison(Xreal,Xfeat,y);

%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%
%%% compare grad descents %%% 
function run_comparison(Xreal,Xfeat,y)

    % containers for comparison
    MIS_real = [];
    OBJ_real = [];
    MIS_feat = [];
    OBJ_feat = [];
    
    % compare newtons' num_runs number of times
    num_runs = 1;
    max_its = 10;

    for r = 1:num_runs
        % initialize weights
        w0 = randn(size(Xreal,1),1)/size(Xreal,1);

        % run with real feats
        [mis,obj,w] = softmax_newtons_method(Xreal,y,w0,max_its);
        MIS_real = [MIS_real, mis];
        OBJ_real = [OBJ_real, obj];
        
        plot_it(mis,obj,[0.7 .7 .7],1)

        % initialize weights
        w0 = randn(size(Xfeat,1),1)/size(Xfeat,1);
        
        % run with hog feats
        [mis,obj,w] = squared_hinge_newtons_method(Xfeat,y,w0,max_its);
        MIS_feat = [MIS_feat, mis];
        OBJ_feat = [OBJ_feat, obj];
        plot_it(mis,obj,[1 .8 1],1)

    end
    % plot mean result
    plot_it(mean(MIS_real,2),mean(OBJ_real,2),'k',2)
    plot_it(mean(MIS_feat,2),mean(OBJ_feat,2),'m',2)
    legend('pixels','hog')
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
        hess = 2*(X_temp*X_temp');
        
        % take Newton step
        temp = hess*w - grad;
        w = pinv(hess)*temp;
                
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
function [Xreal, Xfeat, yout] = load_data()    
      
    %%% real face dataset %%%
    data = csvread('raw_face_data.csv');
    X = data(:,1:end-1);
    y = data(:,end);
    
    % gather chosen number of faces/non-faces
    num_faces = 3000;
    num_gen = 7000;
    ind = find(y == 1);
    faces = X(ind(1:num_faces),:);
    face_labels = y(ind(1:num_faces));
    
    ind = find(y == -1);
    non_faces = X(ind(1:num_gen),:);
    gen_labels = y(ind(1:num_gen));
    
    Xreal = [faces',non_faces']';
    y = [face_labels; gen_labels];

    % update X
    Xreal = [ ones(size(Xreal,1),1), Xreal];
    Xreal = Xreal';
    yout = [face_labels; gen_labels];

    %%% feat face dataset %%%
    data = csvread('feat_face_data.csv');
    X = data(:,1:end-1);
    y = data(:,end);
    
    % gather chosen number of faces/non-faces
    ind = find(y == 1);
    faces = X(ind(1:num_faces),:);
    
    ind = find(y == -1);
    non_faces = X(ind(1:num_gen),:);
    
    Xfeat = [faces',non_faces']';

    % update X
    Xfeat = [ ones(size(Xfeat,1),1), Xfeat];
    Xfeat = Xfeat';
end


end