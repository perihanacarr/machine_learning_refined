function another_ramp_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% variables
data = csvread('another_ramp_experiment.csv',1);
x = data(:,1);
y = data(:,2);

% transform the data  
f = sin(pi*x/180);

% fit a linear model in the transformed data space
w = linsolve(f,y);
w = pinv(f'*f)*(f'*y);

% fit model to transformed data
figure(1)
plot(x,y,'o','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',8)
hold on
model = [min(x):0.1:max(x)];
out = w*sin(pi*model/180);

plot(model,out,'m','LineWidth',2)
box on
%axis([min(model) max(model) (min(y) - 1) (max(y) + 1)])
set(gca,'FontSize',12);

% define labels for subplot
xlabel('angle','Fontsize',14,'FontName','cmr10')
ylabel('traveled distance','Fontsize',14,'FontName','cmr10')
set(get(gca,'YLabel'),'Rotation',90)
set(gcf,'color','w');

end


