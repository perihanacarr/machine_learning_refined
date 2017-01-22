function plot_extended_sinusoid_surface()


% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
data = csvread('extended_sinusoid_data.csv');
x = data(:,1);
y = data(:,2);

[w1 w2] = meshgrid(-3:.05:3);

% create surface
g = zeros(size(w1));

for i = 1:numel(w1)
    z = [w1(i);w2(i)];
    g(i) = norm(w1(i)*sin(2*pi*w2(i)*x)-y)^2;
end

% plot surface
surf(w1,w2,g)

% make labels for graph
xlabel('w_1','Fontsize',14,'FontName','cmr10')
ylabel('w_2','Fontsize',14,'FontName','cmr10')
zlabel('g','Fontsize',14,'FontName','cmr10')
set(get(gca,'ZLabel'),'Rotation',0)
set(gcf,'color','w'); % makes background white

end
