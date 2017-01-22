function sinusoid_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% load data and plot in each panel
[x,y] = load_data();

% fit models in respective panel
subplot(1,3,2)
hold on
fit_sin_model(x,y)

subplot(1,3,3)
hold on
deg = 1;
fit_model(sin(x),y,deg)

%%%%%% functions %%%%%%%%
function [x,y] = load_data()
    % Variables
    data = csvread('sinusoid_example_data.csv');
    x = data(:,1);
    y = data(:,2);
    
    % plot in each panel
    for i = 1:3
        subplot(1,3,i)
        scatter(x,y,'k','fill');
        ylabel('y  ','Fontsize',18,'FontName','cmmi9')
        xlabel('x','Fontsize',18,'FontName','cmmi9')
        set(get(gca,'YLabel'),'Rotation',0)
        axis([0 1 -1.3 1.3])
        axis square
        box off
    end
    subplot(1,3,3)
    scatter(sin(2*pi*x),y,'MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',0.5)
    ylabel('y  ','Fontsize',18,'FontName','cmmi9')
    xlabel('f(x)','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    axis([-1.1 1.1 -1.3 1.3])
    axis square
    box off
    set(gcf,'color','w');

end

function fit_sin_model(x,y)
    % build transformation
    F = [];
    F = [ones(length(x),1) sin(2*pi*x)];
    w = linsolve(F,y);
    
    % plot transformation
    s = [-1:0.01:1]';
    t = w(1) + w(2)*sin(2*pi*s);
    plot(s,t,'m','LineWidth',1.4);
end


function fit_model(x,y,deg)
    % build transformation
    F = [ones(length(x),1), sin(2*pi*x)];
    w_tilde = linsolve(F,y);
    b = w_tilde(1);
    w = w_tilde(2);
    
    % plot transformation
    s = [-1:0.01:1]';
    t = b + s*w;
    plot(s,t,'m','LineWidth',1.4);
end
end





