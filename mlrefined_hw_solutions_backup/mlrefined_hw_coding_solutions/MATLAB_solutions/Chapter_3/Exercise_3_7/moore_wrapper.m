function moore_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
data = csvread('transistor_counts.csv');
x = data(:,1);
y = data(:,2);

% fit model in transformed feature space
y_transformed = log(y);
x_tilde = [ones(length(x),1) x];
w_tilde = linsolve(x_tilde,y_transformed);
b = w_tilde(1);
w = w_tilde(2);

% plot model in the original space
subplot(1,2,1)
hold on
model = [1968:0.01:2015];
out = exp(b + w*model);
plot(model,out,'m','LineWidth',2)
plot(x,y,'o','MarkerEdgeColor','k','MarkerFaceColor','k')
box on
axis tight

% make labels correct
xlabel('year','Fontsize',14,'FontName','cmr10')
ylabel('number of transistors','Fontsize',14,'FontName','cmr10')
set(get(gca,'YLabel'),'Rotation',90)
set(gca,'FontSize',12); 

% plot model in the transformed space
subplot(1,2,2)
out = b + w*model;
hold on
plot(model,out,'m','LineWidth',2)
hold on
plot(x,y_transformed,'o','MarkerEdgeColor','k','MarkerFaceColor','k')
box on
axis tight

% define labels for subplot
xlabel('year','Fontsize',14,'FontName','cmr10')
ylabel('log of number of transistors','Fontsize',14,'FontName','cmr10')
set(get(gca,'YLabel'),'Rotation',90)
set(gcf,'color','w');
set(gca,'FontSize',12); 

end





