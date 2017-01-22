function squared_margin_grad_adaptive()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


%%% load data %%%
[X,y] = load_data();

w0 = randn(3,1);       % initial point
[w, obj] = squared_margin_grad(X,y,w0);

%%% plot everything, pts and lines %%%
plot_all(X(2:3,:)',y,w);

%%% plot the objective per iteration of gradient descent %%%
plot_obj(obj)


%%% gradient descent function for squared hinge loss %%%
function [w, obj] = squared_margin_grad(X,y,w)
    % Initializations 
    H = diag(y)*X';
    l = ones(size(X,2),1);
    iter = 1;
    max_its = 100;
    grad = 1;
    obj = [objective(X,y,w)];


    while  norm(grad) > 10^-8 && iter < max_its
        
        % form gradient
        grad = -2*H'*max(l-H*w,0);
        alpha = adaptive_step(w,grad,obj(iter),X,y);
        w = w - alpha*grad;
        
        % update iteration count
        obj = [obj, objective(X,y,w)];
        iter = iter + 1;
    end
end

function p = adaptive_step(w,g,o,X,y)
    g_n = norm(g,2)^2;
    step_l = 1;
    step_r = 0;
    u = 1;
    p = 1;
    while step_l > step_r && u < 20
        p = p*0.5;

        % left
        step_l = objective(X,y,w - p*g) - o;

        % right 
        step_r = -(p*g_n)/2;
        u = u + 1;
    end
end


function obj = objective(X,y,w)
    t = max(0,1-y.*(X'*w));
    obj = norm(t,2)^2;
end

%%% plots everything %%%
function plot_all(data,y,w)
    red = [1 0 .4];
    blue =  [ 0 .4 1];

    % plot points 
    ind = find(y == 1);
    scatter(data(ind,1),data(ind,2),'Linewidth',2,'Markeredgecolor',blue,'markerFacecolor','none');
    hold on
    ind = find(y == -1);
    scatter(data(ind,1),data(ind,2),'Linewidth',2,'Markeredgecolor',red,'markerFacecolor','none');
    hold on

    % plot separator
    s =[0:0.01:1];
    plot (s,(-w(1)-w(2)*s)/w(3),'m','linewidth',2);
    hold on

    % make plot nice looking
    set(gcf,'color','w');
    axis square
    box off
    
    % graph info labels
    xlabel('w_1','Fontsize',14)
    ylabel('w_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)
    axis([(min(X(2,:)) - 0.05) (max(X(2,:)) + 0.05) (min(X(3,:)) - 0.05) (max(X(3,:)) + 0.05)]);
end

%%% loads data %%%
function [A,b] = load_data()
    data = load('imbalanced_2class.csv');
    A = data(:,1:end-1);
    A = [ones(size(A,1),1) A]';
    b = data(:,end);
end

function plot_obj(obj)
    
    figure;
    plot(1:length(obj),obj,'r','linewidth',2)
    axis square
    xlabel('iteration');
    ylabel('objective value');
    box on
end

end
