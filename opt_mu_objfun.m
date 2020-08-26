function omega=desecent_mu_objfun_JP(J, data_diff, alpha);
%% Using the Modified Objective function as explained in the paper and 
%% calculating the optimal alpha using fminsearh function

deltamu = alpha*((J'*data_diff));

omega = norm((data_diff - J*deltamu) ,2)^2;


