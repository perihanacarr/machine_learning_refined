function knn_wrapper()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.


k = 5;

% load data
data = csvread('knn_data.csv');
A = data(:,1:2);
b = data(:,3);

% Create classified regions using k-NN
B_news = [];
pts = [];
for i = 1:0.1:10
    for j = 1:0.1:10
        a_new = [i,j];
        pts = [pts; a_new];
        b_new = my_knn(A,b,a_new,k);
        B_news = [B_news, b_new];
    end
end

% Plot labels of new points
ind = find(B_news == 0);
P1 = plot(pts(ind,1),pts(ind,2),'o')
set(P1,'Color',[0.5 0.5 1],'MarkerFaceColor',[0.5 0.5 1],'Markersize',10);
alpha(0.001)

hold on
ind = find(B_news == 1);
P1 = plot(pts(ind,1),pts(ind,2),'o');
set(P1,'Color',[1 0.5 0.5],'MarkerFaceColor',[1 0.5 0.5],'Markersize',10)
alpha(0.001)

hold on
% plot data
ind = find(b == 0);
ind = ind(end);
scatter(A(1:ind,1),A(1:ind,2),'bo','fill')
hold on
scatter(A(ind + 1:end,1),A(ind + 1:end,2),'ro','fill')
axis square

function b_new = my_knn(A,b,a_new,k)

% k-NN algorithm
D = repmat(a_new,size(A,1),1);
diffs = sum((A - D).*(A - D),2);
[vals,ind] = sort(diffs);
b_new = round(mean(b(ind(1:k))));

end

end




