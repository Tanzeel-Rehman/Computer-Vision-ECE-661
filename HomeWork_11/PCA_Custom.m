function normW = PCA_Custom(norm_vec_all)
%%
%This is the PCA based decomposition. Makesure don't name it as PCA as it
%is default name used by MAtlab's builtin function.

% Decompose the square matrix of size 630x630 into eig vectors and eg vals
[U,D]= eig(norm_vec_all'*norm_vec_all);
%Sort the diagonalized eigenvalues
[~,idx] = sort(-1 .* diag(D));
U = U(:,idx);
%Compute weights and normalize them using norm of column vectors
w=norm_vec_all*U;
normW = w./vecnorm(w);
%
end