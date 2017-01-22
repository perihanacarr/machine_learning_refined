function hold_out_wrapper()

% cross validation on single fold for regression with polynomial features
% here a 'model' is a set of poly features of some fixed degree

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% parameters to play with
k = 3;                  % # of folds to use, we use only 1st here
degs = 1:8;       % range of poly models to compare

% load data 
[x,y] = load_data(k);

% split points into k equal sized sets 
c = split_data(x,y,k);

% do hold out cross-validation
cross_validate(x,y,c,degs,k); 

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

    % plot train/test sets for firs cross-validation instance
    j = 1;
    
    % training data for jth fold
    x_r = x(find(c == j));
    y_r = y(find(c == j));

    % testing data for jth fold
    x_t = x(find(c~=j));
    y_t = y(find(c~=j));

    % plot each fold's data for visualization
    subplot(1,4,2)
    box on
    plot_pts(x_r,y_r,x_t,y_t)
    
end
        
%%% performs k-fold cross validation %%%
function cross_validate(x,y,c,degs,k,iter)  

    % solve for weights and collect test errors
    test_errors = [];
    train_errors = [];
    for i = 1:length(degs)
        % generate features
        deg = degs(i);
        F = build_fourier(x,deg);
        
        % compute training/testing errors
        train_resids = [];
        test_resids = [];
        for j = 1:1
            F_1 = F(find(c ~= j),:);
            y_1 = y(find(c ~= j));
            F_2 = F(find(c==j),:);
            y_2 = y(find(c==j));

            w = linsolve(F_1,y_1);
            resid = norm(F_2*w - y_2)/numel(y_2);
            test_resids = [test_resids resid];
            resid = norm(F_1*w - y_1)/numel(y_1);
            train_resids = [train_resids resid];
        end
        test_errors = [test_errors; test_resids];
        train_errors = [train_errors; train_resids];
    end

    % find best parameter per data-fold and plot fit to training data
    for i = 1:1
        [val,j] = min(test_errors(:,i));
        
        % rebuild features
        deg = degs(j);
        F = build_fourier(x,deg);

        % solve for best fit weights
        F_1= F(find(c ~= i),:);
        y_1 = y(find(c ~= i));
        w =linsolve(F_1,y_1);   
        
        % output model
        subplot(1,4,2) 
        hold on
        plot_poly(w,deg,'b')
    end
    
    % solve for overall best fit model - the one minimizing ave test error
    test_ave = test_errors;
    [val,j] = min(test_ave);
    s1 = 'Best cross-validated poly model has deg = ';
    s = [s1, num2str(j)];
    disp(s)   
    
    % build features
    deg = degs(j);
    F = build_fourier(x,deg);
    w =linsolve(F,y);
    
    % plot learned model
    hold on
    subplot(1,4,4)
    plot_poly(w,deg,'r')
     
    % plot (mean) train/test errors for visualization
    subplot(1,4,3)
    plot_errors(degs, test_errors, train_errors)
end
  
%%% buids fourier features %%%
function F = build_fourier(data,deg)
    F = [];
    for n = 1:deg
        F = [F, cos(2*pi*n*data), sin(2*pi*n*data)];
    end
    F = [ones(size(F,1),1) F];
end

%%% plots fit %%%
function plot_poly(w,deg,color)
    model = [min(x):0.001:max(x)]';
    F = build_fourier(model,deg);
    out = sum(F*diag(w),2);
    plot(model,out,color,'LineWidth',1.25)

    % clean up plot
    axis([(min(x) - 0.1) (max(x) + 0.1) (min(y) - 0.1) (max(y) + 0.1)])
    axis square
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('y ','Fontsize',18,'FontName','cmmi9')   
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
end

%%% plot (mean) train/test errors %%%
function plot_errors(poly_degs, test_errors, train_errors)
    % plot training and testing mean errors
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
    set(gca,'xtick',0:max(poly_degs))
    box on
    axis([0.5 max(poly_degs) 0 max(test_errors)])
    axis square
    xlabel('D','Fontsize',18,'FontName','cmr10')
    ylabel('error','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',90)
    box off
    set(gcf,'color','w');
    set(gca,'FontSize',12); 
end

%%% loads data %%%
function [x,y] = load_data(k)

   % load points and plot
%     data = load('fourier_good_2.mat');
%     data = data.data;
%     x = data(:,1);
%     y = data(:,2);
    
    data = csvread('wavy_data.csv');
    x = data(:,1);
    y = data(:,2);
    
    
    x_true = 0:0.01:1;
    y_true = exp(3*x_true).*sinc(3*pi*(x_true - 0.5));
    
    % plot
    subplot(1,4,4)
    plot(x,y,'o','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',7)
    hold on
    plot(x_true,y_true,'--k')
    hold on
    subplot(1,4,2)
    plot(x_true,y_true,'--k')
    hold on
    
    subplot(1,4,1)
    plot(x,y,'o','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',7)
    hold on
    plot(x_true,y_true,'--k')
    hold on
    subplot(1,4,1)
    plot(x_true,y_true,'--k')
            % clean up plot
    axis([(min(x) - 0.1) (max(x) + 0.1) (min(y) - 0.1) (max(y) + 0.1)])
    axis square
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    ylabel('y ','Fontsize',18,'FontName','cmmi9')   
    set(get(gca,'YLabel'),'Rotation',0)
    set(gcf,'color','w');
    box off


end

end



