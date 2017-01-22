function Galileo_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


% load data and plot in each panel
[x,y] = load_data();

% fit models in respective panel
subplot(1,3,2)
hold on
deg = 2;
fit_model(x,y,deg)

subplot(1,3,3)
hold on
deg = 1;
fit_model(x.^2,y,deg)

%%%%%% functions %%%%%%%%
function [x,y] = load_data()
    % Variables 
    data = csvread('Galileo_data.csv');
    x = data(:,1);
    y = data(:,2);


    % plot in each panel
    for i = 1:3
        subplot(1,3,i)
        scatter(x,y,'k','fill');
        ylabel('y  ','Fontsize',18,'FontName','cmmi9')
        xlabel('x','Fontsize',18,'FontName','cmmi9')
        set(get(gca,'YLabel'),'Rotation',0)
        axis([-0.25 (max(x) + 0.05) -0.05 1.05])
        axis square
        box off
    end
    subplot(1,3,3)
    scatter(x.^2,y,'k','fill')
    ylabel('y  ','Fontsize',18,'FontName','cmmi9')
    xlabel('x^2','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'YLabel'),'Rotation',0)
    axis([-2 (max(x.^2) + 0.05) -0.05 1.05])
    axis square
    box off
    set(gcf,'color','w');

end

function fit_model(x,y,deg)
    % build poly
    F = [];
    for i = 1:deg
        F = [F x.^i];
    end
    w = linsolve(F,y);
    
    % plot poly
    s = [min(x):0.01:max(x)]';
    t = 0;
    for i = 1:deg
        t = t + w(i)*s.^i;
    end
    plot(s,t,'m','LineWidth',1.4);
end
end





