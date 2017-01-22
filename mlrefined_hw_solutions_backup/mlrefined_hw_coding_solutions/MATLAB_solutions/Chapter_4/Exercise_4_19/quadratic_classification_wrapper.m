function quadratic_classification_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
data = csvread('quadratic_classification.csv');
X = data(:,1:end-1);
y = data(:,end);


% Map data to poly space
poly_deg = 2;
A = [X(:,1).^2, X(:,1), X(:,2), ones(size(X,1),1)];

% run logistic on mapped data (poly separator)
x0 = randn(poly_deg + 2,1);    % initial point
[x,in_g,out_g] = convex_logistic_fast_grad(A,y,.1,x0);

% Generate poly separator
x = -x/x(end - 1);
s = 0:0.001:1;
t = x(1)*s.^2 + x(2)*s + x(end);

% plot poly seperator
subplot(1,2,1)
title('original space')
plot(s,t,'-k','LineWidth',1.5)
axis([0 1 0 1])
axis square
set(gcf,'color','w');

% plot data in the original space 
hold on
ind = find(y == 1);
scatter(X(ind,1),X(ind,2),'o','MarkerEdgeColor','r','MarkerFaceColor','r')
hold on
ind = find(y == -1);
scatter(X(ind,1),X(ind,2),'o','MarkerEdgeColor','b','MarkerFaceColor','b')

% graph info labels
xlabel('x_1','Fontsize',12)
ylabel('x_2  ','Fontsize',12)
set(get(gca,'YLabel'),'Rotation',0)
grid off
axis([0 1 0 1])


% Generate feature space separator
x = -x/x(end - 1);
t = x(1)*s + x(2)*s + x(end);

% plot feature space separator
subplot(1,2,2)
title('transformed feature space')
plot(s,t,'-k','LineWidth',1.5)
axis([0 1 0 1])
axis square
set(gcf,'color','w');
hold on


% plot feature space data 
ind = find(y == 1);
scatter(A(ind,1),A(ind,3),'r','fill','LineWidth',0.5)
hold on
ind = find(y == -1);
scatter(A(ind,1),A(ind,3),'b','fill','LineWidth',0.5)

% graph info labels
xlabel('x_1^2','Fontsize',12)
ylabel('x_2  ','Fontsize',12)
set(get(gca,'YLabel'),'Rotation',0)
grid off
axis([0 1 0 1])


function [x,in,out] = convex_logistic_fast_grad(A,b,t,x0)

% Initializations 
x = x0;
z = x0;
in = [];
out = [];
stopper = 1;
max_its = 10000;      % stopping threshold
k = 1;
A = diag(b)*A;
while k <= max_its && stopper > 10^-9
    x0 = x;
    
    x = z + t*(A'*(sigmoid(-A*z)));
    z = x + k/(k+3)*(x - x0);

    in = [in x];
    out = [out ;-sum(log(sigmoid(A*x)))];
    % update stopping conditions
    if k > 1
       stopper = abs(out(k) - out(k-1))/abs(out(k-1));
    end
    k = k + 1;
end
in = in';

function y = sigmoid(z)
    y = 1./(1+exp(-z));
end

end
 
end
    
    








