function recommender_demo()

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
X = csvread('recommender_demo_data_true_matrix.csv');
X_corrupt = csvread('recommender_demo_data_dissolved_matrix.csv');

K = rank(X);

% run ALS for matrix completion
[C, W] = matrix_complete(X_corrupt, K); 

% plot results
plot_results(X, X_corrupt, C, W)


function [C, W] = matrix_complete(X, K)
    
    max_its = 50;
    lam = 1;  % regularization coeff
    [r c val] = find(X > 0);
    omega = [r c];

    % Initialize variables
    C = randn(size(X,1),K);
    W = randn(K,size(X,2));

    r = omega(:,1);
    c = omega(:,2);
    
    % Main
    for i = 1:max_its

        % Update W
        for k = 1:size(W,2)
            ind = r(find(c==k));
            P = zeros(size(W,1));
            q = zeros(size(W,1),1);
            for m = 1:length(ind)
                P = P + C(ind(m),:)'*C(ind(m),:); 
                q = q + X(ind(m),k)*(C(ind(m),:)');
            end

            W(:,k) = pinv(P + lam*eye(size(W,1)))*q; 
        end

        % Update C
        for k = 1:size(C,1)
            ind = c(find(r==k));
            P = zeros(size(C,2));
            q = zeros(1,size(C,2));
            for m = 1:length(ind)
                P = P + W(:,ind(m))*(W(:,ind(m))'); 
                q = q + X(k,ind(m))*(W(:,ind(m))');
            end

            C(k,:) = q*pinv(P + lam*eye(size(C,2))); 
        end


    end

end

function plot_results(X, X_corrupt, C, W)

    gaps_x = [1:size(X,2)];
    gaps_y = [1:size(X,1)];
    
    % plot original matrix
    subplot(1,3,1)
    imshow(X,[])
    colormap hot
    colorbar
    set(gca,'XTick',gaps_x)
    set(gca,'YTick',gaps_y)
    set(gca,'CLim',[0, max(max(X))])
    title('original')
    set(gcf,'color','w');

    % plot corrupted matrix
    subplot(1,3,2)
    imshow(X_corrupt,[])
    colormap hot
    colorbar
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    set(gca,'CLim',[0, max(max(X))])
    title('corrupted')
    set(gcf,'color','w');

    % plot reconstructed matrix
    hold on
    subplot(1,3,3)
    imshow(C*W,[])
    colormap('hot');
    colorbar
    set(gca,'XTick',gaps_x)
    set(gca,'YTick',gaps_y)
    set(gca,'CLim',[0, max(max(X))])
    RMSE_mat = sqrt(norm(C*W - X,'fro')/prod(size(X)));
    f = ['RMSE-ALS = ',num2str(RMSE_mat),'  rank = ', num2str(rank(C*W))];
    title(f)
    set(gcf,'color','w');

end


end

