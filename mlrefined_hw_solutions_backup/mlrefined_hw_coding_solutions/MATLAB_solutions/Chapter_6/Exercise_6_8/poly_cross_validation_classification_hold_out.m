function poly_cross_validation_hold_out()

% cross validation on single fold for classification (with log-loss) with 
% polynomial features.  Here a 'model' is a set of poly features of some
% fixed degree

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


xmin = 0;   % viewing area minimum 
xmax = 1;   % viewing area maximum

% parameters to play with
k = 3;               % # of folds to make, we only use 1 here
poly_degs = 1:10;    % range of poly models to compare

% load data 
[X,y] = load_data();

% split points into k equal sized sets and plots each
c = split_data(X,y,k);

% do k-fold cross-validation
cross_validate(X,y,c,poly_degs,k);  


%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%
%%% splits data into k folds training/testing sets %%%
function c = split_data(X,y,k)
    % split data into k equal sized sets
    L = length(y);
    order = randperm(L);
    c = ones(L,1);
    L = round((1/k)*L);
    for s = 1:k-2
       c(order(s*L+1:(s+1)*L)) = s + 1;
    end
    c(order((k-1)*L+1:end)) = k;

    % plot train/test sets for first cross-validation instance
    subplot(1,3,1)
    box on
    plot_pts(X,y,c,1)
end

%%% performs k-fold cross validation %%%
function cross_validate(X,y,c,poly_degs,k)  

    test_errors = [];
    train_errors = [];
    % solve for weights and collect test errors
    for i = 1:length(poly_degs)
        % generate features
        poly_deg = poly_degs(i);         
        F = build_poly(X,poly_deg);
        size(F,2)-1
        test_resids = [];
        train_resids = [];
        for j = 1:1     % compute error on 1st fold
            F_1 = F(find(c ~= j),:);
            y_1 = y(find(c ~= j));
            F_2 = F(find(c==j),:);
            y_2 = y(find(c==j));
            
            % run logistic regression
            w = log_loss_newton(F_1',y_1);

            % calculate training/testing errors
            resid = evaluate(F_2,y_2,w);
            test_resids = [test_resids resid];
            resid = evaluate(F_1,y_1,w);
            train_resids = [train_resids resid];
            
        end
        test_errors = [test_errors; test_resids];
        train_errors = [train_errors; train_resids];
    end

    % find best parameter per data-split
    for i = 1:1
        [val,j] = min(test_errors(:,i));
        
        % build features
        poly_deg = poly_degs(j);       
        F = build_poly(X,poly_deg);
        F_1 = F(find(c ~= i),:);
        y_1 = y(find(c ~= i));
        w = log_loss_newton(F_1',y_1);
        
        % plot model learned on training set
        hold on
        subplot(1,3,1)
        plot_poly(w,poly_deg,'k')
    end
    [val,j] = min(test_errors);
    
    % plot best separator for all data
    poly_deg = poly_degs(j);       
    F = build_poly(X,poly_deg);
    w = log_loss_newton(F',y);
    s1 = 'Best cross-validated poly model has deg = ';
    s = [s1, num2str(j)];
    disp(s) 
    
    % plot fit
    hold on
    subplot(1,3,2)
    plot_poly(w,poly_deg,'k')

    % plot (mean) training/testing errors
    subplot(1,3,3)
    plot_errors(poly_degs, test_errors, train_errors)
end
    
%%% builds (poly) features based on input data %%%
function F = build_poly(data,deg)
    F = [];
    for n = 0:deg
        for m = 0:deg
            if n + m <= deg
               F = [F, data(:,1).^(n).*data(:,2).^(m)];
            end
        end
    end
end

%%% plots learned model %%%
function plot_poly(w,deg,color)
    % Generate poly seperator
    o = [xmin:0.01:xmax];
    [s,t] = meshgrid(o,o);
    s = reshape(s,numel(s),1);
    t = reshape(t,numel(t),1);
    f = zeros(length(s),1);
    count = 1;
    for n = 0:deg
        for m = 0:deg
            if n + m <= deg
                f = f + w(count)*s.^(n).*t.^(m);
                count = count + 1;
            end
        end
    end

    s = reshape(s,[length(o),length(o)]);
    t = reshape(t,[length(o),length(o)]);
    f = reshape(f,[length(o),length(o)]);  
    % plot contour in original space
    hold on
    contour(s,t,f,[0,0],'Color','k','LineWidth',2)
    axis([xmin xmax xmin xmax])
    axis square
    
    xlabel('x_1','Fontsize',18,'FontName','cmmi9')
    ylabel('x_2   ','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    box off
end

%%% plots points for each fold %%%
function plot_pts(X,y,c,j)
    
    % plot training set
    ind = find(c ~= j);
    ind2 = find(y(ind) == 1);
    ind3 = ind(ind2);
    red =  [ 1 0 .4];

    plot(X(ind3,1),X(ind3,2),'o','Linewidth',2.5,'MarkerEdgeColor',red,'MarkerFaceColor','none','MarkerSize',7)
    hold on
    ind2 = find(y(ind) == -1);
    ind3 = ind(ind2);
    blue =  [ 0 .4 1];
    plot(X(ind3,1),X(ind3,2),'o','Linewidth',2.5,'MarkerEdgeColor',blue,'MarkerFaceColor','none','MarkerSize',7)
    
    % plot test set?
    ind = find(c == j);
    ind2 = find(y(ind) == 1);
    ind3 = ind(ind2);
    red =  [ 1 0 .4];
    plot(X(ind3,1),X(ind3,2),'o','Linewidth',1,'MarkerEdgeColor',red,'MarkerFaceColor','none','MarkerSize',7)

    hold on
    ind2 = find(y(ind) == -1);
    ind3 = ind(ind2);
    blue =  [ 0 .4 1];
    plot(X(ind3,1),X(ind3,2),'o','Linewidth',1,'MarkerEdgeColor',blue,'MarkerFaceColor','none','MarkerSize',7)
end

%%% plots (mean) training/testing errors %%%
function plot_errors(poly_degs, test_errors, train_errors)
    % plot training and testing errors
    h1 = plot(1:max(poly_degs),test_errors,'--','Color',[1 0.7 0]);
    hold on
    plot(1:max(poly_degs),test_errors,'o','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0])
    hold on
    h2 = plot(1:max(poly_degs),train_errors,'--','Color',[0.1 0.8 1]);
    hold on
    plot(1:max(poly_degs),train_errors,'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1])
    legend([h1 h2],{'testing error','training error'}); 

    % clean up plot
    set(gcf,'color','w');
    box on
    xlabel('M','Fontsize',18,'FontName','cmr10')
    ylabel('error','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',90)
    set(gcf,'color','w');
    box off
    axis square
end

%%% loads and plots labeled data %%%
function [X,y] = load_data()
      
    % load data
    data = csvread('2eggs_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);

    subplot(1,3,2)
    ind = find(y == 1);
    red =  [ 1 0 .4];
    plot(X(ind,1),X(ind,2),'o','Linewidth',2.5,'MarkerEdgeColor',red,'MarkerFaceColor','none','MarkerSize',7)
    hold on
    ind = find(y == -1);
    blue =  [ 0 .4 1];
    plot(X(ind,1),X(ind,2),'o','Linewidth',2.5,'MarkerEdgeColor',blue,'MarkerFaceColor','none','MarkerSize',7)

end

%%% newton's method for log-loss classifier %%%
function w = log_loss_newton(D,b)
    % initialize
    w = randn(size(D,1),1);

    % precomputations
    H = diag(b)*D';
    grad = 1;
    n = 1;

    %%% main %%%
    while n <= 30 && norm(grad) > 10^-5

        % prep gradient for logistic objective
        r = sigmoid(-H*w);
        g = r.*(1 - r);
        grad = -H'*r;
        hess = D*diag(g)*D';

        % take Newton step
        s = hess*w - grad;
        w = pinv(hess)*s;
        n = n + 1;
    end
end

%%% sigmoid function for use with log_loss_newton %%%
function t = sigmoid(z)
    t = 1./(1+exp(-z));
end

%%% evaluates error of a learned model %%%
function score = evaluate(A,b,w)
    s = A*w;
    ind = find(s > 0);
    s(ind) = 1;
    ind = find(s <= 0);
    s(ind) = -1;
    t = s.*b;
    ind = find(t < 0);
    t(ind) = 0;
    score = 1 - sum(t)/numel(t);

end

end





