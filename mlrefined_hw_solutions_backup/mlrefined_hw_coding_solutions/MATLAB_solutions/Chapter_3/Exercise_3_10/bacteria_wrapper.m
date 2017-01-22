function bacteria_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data and plot in each panel
[x,y] = load_data();

% fit models in respective panels
y_transformed = log(y./(1-y));
fit_sig_model(x, y_transformed)

%%%%%% functions %%%%%%%%
function [x,y] = load_data()
    % Variables
    data = csvread('bacteria_data.csv');
    x = data(:,1);
    y = data(:,2);
    
    % plot the data in each panel
    for i = 1:2
        subplot(1,3,i)
        scatter(x,y,'k','fill');
        ylabel('y  ','Fontsize',18,'FontName','cmmi9')
        xlabel('x','Fontsize',18,'FontName','cmmi9')
        set(get(gca,'YLabel'),'Rotation',0)
        axis square
        box off
    end
    
    subplot(1,3,3)
    scatter(x,log(y./(1-y)),'MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',0.5)
    ylabel('f(y)  ','Fontsize',18,'FontName','cmmi9')
    xlabel('x','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    axis square
    box off
    set(gcf,'color','w');

end

function fit_sig_model(x,y)

    % fit a linear model to the data in the transformed space 
    x_tilde = [ones(length(x),1) x];
    w_tilde = linsolve(x_tilde,y);
    b = w_tilde(1);
    w = w_tilde(2);
    
    % plot nonlinear model in the original space 
    subplot(1,3,2)
    hold on
    s = [0:0.1:25]';
    t = 1./(1 + exp(-(b + w*s)));
    plot(s,t,'m','LineWidth',1.4);
    
    % plot linear model in the transformed space 
    subplot(1,3,3)
    hold on
    t = b + w*s;
    plot(s,t,'m','LineWidth',1.4);
end

end





