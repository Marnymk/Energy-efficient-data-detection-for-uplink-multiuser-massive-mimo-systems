function MSE = functionBERmatrix(R_diag,q_powerallocation,B)
%
%INPUT:
%R_diag            = Nt Nr x Nt Nr diagonal covariance matrix
%q_powerallocation = Nt x 1 vector with training power allocation
%B                 = Length of the training sequence.
%
%OUTPUT:
%MSE               = Mean Squared Error for estimation of the channel matrix

P_tilde = kron(diag(sqrt(q_powerallocation)),eye(B));

MSE = trace(R_diag - R_diag*(P_tilde'/(P_tilde*R_diag*P_tilde'+eye(length(R_diag))))*P_tilde*R_diag);
