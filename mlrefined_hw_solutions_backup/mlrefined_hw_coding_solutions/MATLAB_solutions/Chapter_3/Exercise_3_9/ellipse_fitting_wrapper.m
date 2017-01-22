function ellipse_fitting_wrapper()

% This wrapper fits an ellipse to noisy elliptical data, also
% illustrating a linear fit in the feature space [x1,x2] -> [(x1)^2,(x2)^2]

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data
X = csvread('asteroid_data.csv')';

% produce weights
y = ones(size(X,2),1);
F = X.^2;              % reassign values to feature space
w = pinv(F*F')*(F*y);

% plot
plot_all(X,y,w)

%%% functions %%%
function plot_all(X,y,w)
    
    subplot(1,2,1)
    % generate figure with linear transformation
    scatter(X(1,:),X(2,:),45,'fill','MarkerFaceColor','k');
    
    % plot elliptical fit in original space
    hold on
    a = linspace(min(min(X))-1,max(max(X))+1,500);
    y = a;
    [a,y] = meshgrid(a,y);
    z = w(1)*a.^2 + w(2)*y.^2;
    v = [1,1];
    hold on
    contour(a,y,z,v,'m','linewidth',2)

    % graph info labels
    xlabel('x_1','Fontsize',20,'FontName','cmmi9')
    ylabel('x_2   ','Fontsize',20,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gca,'XTickLabel',[])
    set(gca,'YTickLabel',[])
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    axis([-6 6 -1.2 1.2])
    set(gcf,'color','w');

    %%% plot transformed data
    subplot(1,2,2)
    scatter(X(1,:).^2,X(2,:).^2,45,'fill','MarkerFaceColor','k');
    
    % plot linear fit in feature space
    hold on
    s = linspace(min(X(1,:).^2),max(X(1,:).^2));
    plot(s,(1/w(2) - w(1)/w(2)*s),'m','linewidth',2);
    set(gcf,'color','w');
    
    % graph info labels
    xlabel('x_1^2','Fontsize',20,'FontName','cmmi9')
    ylabel('x_2^2    ','Fontsize',20,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    set(gca,'XTickLabel',[])
    set(gca,'YTickLabel',[])
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    axis([ (min(X(1,:).^2) - 0.5) (max(X(1,:).^2) + 0.5) (min(X(2,:).^2)-0.1) (max(X(2,:).^2)+0.1) ])

end

end









