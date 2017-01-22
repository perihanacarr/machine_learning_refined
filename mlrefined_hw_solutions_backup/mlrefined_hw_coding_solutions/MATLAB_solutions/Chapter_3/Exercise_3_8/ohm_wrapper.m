function ohm_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
data = csvread('ohms_data.csv');
x = data(:,1);
y = data(:,2);

% fit model in transformed feature space
x_tilde = [ones(length(x),1) x];
y_transformed = y.^-1;
w_tilde = linsolve(x_tilde,y_transformed);
b = w_tilde(1);
w = w_tilde(2);

% plot model in the original space
figure(1)
hold on
model = [0:0.01:100];
out = 1./(b+ w*model);
plot(model,out,'m','LineWidth',2)
plot(x,y,'o','MarkerEdgeColor','k','MarkerFaceColor','k')
box on
axis tight

% make labels correct
xlabel('length of wire','Fontsize',14,'FontName','cmr10')
ylabel('current','Fontsize',14,'FontName','cmr10')
set(get(gca,'YLabel'),'Rotation',90)
set(gca,'FontSize',12); 

% define labels for subplot
xlabel('length of wire','Fontsize',14,'FontName','cmr10')
ylabel('inverse of current','Fontsize',14,'FontName','cmr10')
set(get(gca,'YLabel'),'Rotation',90)
set(gcf,'color','w');
set(gca,'FontSize',12); 

end





