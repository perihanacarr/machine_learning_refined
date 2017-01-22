function k_fold_wrapper()

% k-fold cross validation for regression with polynomial features
% here a 'model' is a set of poly features of some fixed degree

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% parameters to play with
k = 6;                  % # of folds to use
poly_degs = 1:6;       % range of poly models to compare

% load data 
[x,y] = load_data(k);

% split points into k equal sized sets and plot
c = split_data(x,y,k);

% do k-fold cross-validation
cross_validate(x,y,c,poly_degs,k);  

%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%
%%% splits data into k training/testing sets %%%
function c = split_data(x,y,k)
   % split data into k equal (as possible) sized sets
    L = length(y);
    order = randperm(L);
    c = ones(L,1);
    L = round((1/k)*L); 
    for s = 1:k-2
       c(order(s*L+1:(s+1)*L)) = s + 1;
    end
    c(order((k-1)*L+1:end)) = k;

    % plot train/test sets for each cross-validation instance
    for j = 1:k
        % training data for jth fold
        x_r = x(find(c == j));
        y_r = y(find(c == j));
        
        % testing data for jth fold
        x_t = x(find(c~=j));
        y_t = y(find(c~=j));
        
        % plot each fold's data for visualization
        subplot(1,k+2,j)
        box on
        plot_pts(x_r,y_r,x_t,y_t)
    end
end
        
%%% performs k-fold cross validation %%%
function cross_validate(x,y,c,poly_degs,k)  

    % solve for weights and collect test errors
    test_errors = [];
    train_errors = [];
    for i = 1:length(poly_degs)
        % generate features
        deg = poly_degs(i);
        F = build_poly(x,deg);
        
        % compute training/testing errors
        train_resids = [];
        test_resids = [];
        for j = 1:k
            F_1 = F(find(c ~= j),:);
            y_1 = y(find(c ~= j));
            F_2 = F(find(c==j),:);
            y_2 = y(find(c==j));

            w = linsolve(F_1,y_1);
%             w = pinv(F_1'*F_1)*F_1'*y_1;
            resid = norm(F_2*w - y_2)/numel(y_2);
            test_resids = [test_resids resid];
            resid = norm(F_1*w - y_1)/numel(y_1);
            train_resids = [train_resids resid];
        end
        test_errors = [test_errors; test_resids];
        train_errors = [train_errors; train_resids];
    end

    % find best parameter per data-fold and plot fit to training data
    for i = 1:k
        [val,j] = min(test_errors(:,i));
        
        % rebuild features
        deg = poly_degs(j);
        F = build_poly(x,deg);

        % solve for best fit weights
        F_1= F(find(c ~= i),:);
        y_1 = y(find(c ~= i));
        w =linsolve(F_1,y_1);   
%         w = pinv(F_1'*F_1)*F_1'*y_1;
        
        % output model
        subplot(1,k+2,i) 
        hold on
        plot_poly(w,deg,'b')
    end
    
    % solve for overall best fit model - the one minimizing ave test error
    test_ave = mean(test_errors');
    [val,j] = min(test_ave);
    s = 'Best ';
    s1 = '-fold cross-validated poly model has deg = ';
    s = [s,num2str(k), s1, num2str(j)];
    disp(s)   
    
    % build features
    deg = poly_degs(j);
    F = build_poly(x,deg);
    w =linsolve(F,y);
    
    % plot learned model
    hold on
    subplot(1,k+2,k+2)
    plot_poly(w,deg,'b')
    
    % plot (mean) train/test errors for visualization
    subplot(1,k+2,k+1)
    plot_errors(poly_degs, test_errors, train_errors)
end
  
%%% buids polynomial features %%%
function F = build_poly(data,deg)
    F = [];
    for j = 0:deg
        F = [F  data.^j];
    end
end

%%% plots fit %%%
function plot_poly(w,deg,color)
    model = [min(x):0.01:max(x)]';
    out = [];
    for j = 1:deg;
        out = [out  w(j + 1)*model.^j];
    end
    out = sum(out,2) + w(1);
    plot(model,out,color,'LineWidth',1.25)

    % clean up plot
    axis([(min(x) - 0.1) (max(x) + 0.1) (min(y) - 0.1) (max(y) + 0.1)])
    axis square
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('y','Fontsize',18,'FontName','cmmi9')   
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    box off
end

%%% plots pts from each fold colored for visualization %%%
function plot_pts(x_r,y_r,x_e,y_e)
    % plot train
    hold on
    plot(x_e,y_e,'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1],'MarkerSize',7)
    % plot test
    hold on
    plot(x_r,y_r,'o','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0],'MarkerSize',7)
    set(gcf,'color','w');
    box on
end

%%% plot (mean) train/test errors %%%
function plot_errors(poly_degs, test_errors, train_errors)
    % plot training and testing mean errors
    h1 = plot(1:max(poly_degs),mean(test_errors'),'--','Color',[1 0.7 0]);
    hold on
    plot(1:max(poly_degs),mean(test_errors'),'o','MarkerEdgeColor',[1 0.7 0],'MarkerFaceColor',[1 0.7 0])
    hold on
    h2 = plot(1:max(poly_degs),mean(train_errors'),'--','Color',[0.1 0.8 1]);
    hold on
    plot(1:max(poly_degs),mean(train_errors'),'o','MarkerEdgeColor',[0.1 0.8 1],'MarkerFaceColor',[0.1 0.8 1])
    
    legend([h1 h2],{'ave test error','ave train error'}); 

    
    % clean up plot
    set(gcf,'color','w');
    set(gca,'FontSize',12); 
    set(gca,'xtick',0:max(poly_degs))
    s = mean(test_errors');
    axis([0.5 6 0 max(s(1:6))])
    axis square
    xlabel('M','Fontsize',18,'FontName','cmr10')
    ylabel('average error','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',90)
    box off
    set(gcf,'color','w');
        
end

%%% loads data %%%
function [x,y] = load_data(k)

   % load points and plot
%     data = load('discrete_sin_data.mat');
%     data = data.data;
%     x = data(:,1);
%     y = data(:,2);
    
    data = csvread('galileo_ramp_data.csv');
    x = data(:,1);
    y = data(:,2);
    
    % plot
    subplot(1,k+2,k+2)
    plot(x,y,'o','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',7)
    hold on
    set(gcf,'color','w');
end

end



