clc;
clear all;
close all;

rng('shuffle'); 


%Number of realizations in the Monte-Carlo simulations
nbrOfMonteCarloRealizations = 1;

T = 1; 

T = 5; 

Nt = 8; 
Nr = 8;

P = 0:0.1:0.3; 
P = 0:1:20; %In decibel scale

totalTrainingPower = 10.^(P/10); %In linear scale

option = optimset('Display','off','TolFun',1e-7,'TolCon',1e-7,'Algorithm','interior-point');
 SNR = 0:2:18;


average_BER_MMSE_estimator_RM = zeros(length(totalTrainingPower),T,2); %MMSE estimator with RM training
average_BER_MMSE_estimator_AMP = zeros(length(totalTrainingPower),T,2); %MMSE estimator with AMP training (from Heuristic 1)
average_BER_CDM_estimator = zeros(length(totalTrainingPower),T,2); %CDM estimator with RM training (uniform training)
average_BER_LMMSE_estimator = zeros(length(totalTrainingPower),T,2); %One-sided linear estimator 
average_BER_twosided_estimator = zeros(length(totalTrainingPower),T,2); %Two-sided linear estimator 

for statisticsIndex = 1:T
    
    %Generate coupling matrix V of the Weichselberger model 
    V = abs(randn(Nr,Nt)+1i*randn(Nr,Nt)).^2;
    V = Nt*Nr*V/sum(V(:)); 
    
    %Compute the covariance matrix for the given coupling matrix
    R = diag(V(:));
    
    R_T = diag(sum(V,1));
    R_R = diag(sum(V,2)); 
     
    trainingpower_MMSE_RM = zeros(Nt,length(totalTrainingPower)); 
    
    for k = 1:length(totalTrainingPower) 
        trainingpower_initial = totalTrainingPower(k)*ones(Nt,1)/Nt;         
        trainingpower_MMSE_RM(:,k) = fmincon(@(q) functionBERmatrix(R,q,Nr),trainingpower_initial,ones(1,Nt),totalTrainingPower(k),[],[],zeros(Nt,1),totalTrainingPower(k)*ones(Nt,1),[],option);
    end
    
    

    [eigenvalues_sorted,permutationorder] = sort(diag(R_T),'descend'); 
    [~,inversePermutation] = sort(permutationorder); 
    
    q_MMSE_AMP = zeros(Nt,length(totalTrainingPower));
    for k = 1:length(totalTrainingPower) 
        alpha_candidates = (totalTrainingPower(k)+cumsum(1./eigenvalues_sorted(1:Nt,1)))./(1:Nt)'; 
        RMIndex = find(alpha_candidates-1./eigenvalues_sorted(1:Nt,1)>0 & alpha_candidates-[1./eigenvalues_sorted(2:end,1); Inf]<0); 
        q_MMSE_AMP(:,k) = max([alpha_candidates(RMIndex)-1./eigenvalues_sorted(1:Nt,1) zeros(Nt,1)],[],2); 
    end
    
    q_MMSE_AMP = q_MMSE_AMP(inversePermutation,:); 
    
    
    q_uniform = (ones(Nt,1)/Nt)*totalTrainingPower;
    
    
    vecH_realizations = sqrtm(R)*( randn(Nt*Nr,nbrOfMonteCarloRealizations)+1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); 
    vecN_realizations = ( randn(Nt*Nr,nbrOfMonteCarloRealizations)+1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); 
    
    
    for k = 1:length(totalTrainingPower)
        
        P_tilde = kron(diag(sqrt(trainingpower_MMSE_RM(:,k))),eye(Nr)); 
        
        average_BER_MMSE_estimator_RM(k,statisticsIndex,1) = trace(R - (R*P_tilde'/(P_tilde*R*P_tilde' + eye(length(R))))*P_tilde*R); 
        
        H_hat = (R*P_tilde'/(P_tilde*R*P_tilde'+eye(length(R)))) * (P_tilde*vecH_realizations+vecN_realizations); 
        average_BER_MMSE_estimator_RM(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) );
        
        
        P_tilde = kron(diag(sqrt(q_MMSE_AMP(:,k))),eye(Nr));  
        
        average_BER_MMSE_estimator_AMP(k,statisticsIndex,1) = trace(R - (R*P_tilde'/(P_tilde*R*P_tilde' + eye(length(R))))*P_tilde*R); 
        
        H_hat = (R*P_tilde'/(P_tilde*R*P_tilde'+eye(length(R)))) * (P_tilde*vecH_realizations + vecN_realizations); 
        average_BER_MMSE_estimator_AMP(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); 
        
        
        P_training = diag(sqrt(q_uniform(:,k))); 
        P_tilde = kron(transpose(P_training),eye(Nr)); 
        P_tilde_pseudoInverse = kron((P_training'/(P_training*P_training'))',eye(Nr)); 
        
        average_BER_CDM_estimator(k,statisticsIndex,1) = Nt^2*Nr/totalTrainingPower(k); 
        
        H_hat = P_tilde_pseudoInverse'*(P_tilde*vecH_realizations + vecN_realizations); 
        average_BER_CDM_estimator(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) );
        
      
        
        P_training = diag(sqrt(q_MMSE_AMP(:,k))); 
        P_tilde = kron(P_training,eye(Nr)); 
        average_BER_LMMSE_estimator(k,statisticsIndex,1) = trace(inv(inv(R_T)+P_training*P_training'/Nr));
        
        Ao = (P_training'*R_T*P_training + Nr*eye(Nt))\P_training'*R_T;
        H_hat = kron(transpose(Ao),eye(Nr))*(P_tilde*vecH_realizations + vecN_realizations); 
        average_BER_LMMSE_estimator(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) );  
        
        
     
        
        P_training = diag(sqrt(q_uniform(:,k))); 
        P_tilde = kron(P_training,eye(Nr)); 
        R_calE = sum(1./q_uniform(:,k))*eye(Nr); 
        
        average_BER_twosided_estimator(k,statisticsIndex,1) = trace(R_R-(R_R/(R_R+R_calE))*R_R); 
        
        C1 = inv(P_training); 
        C2bar = R_R/(R_R+R_calE);
        H_hat = kron(transpose(C1),C2bar)*(P_tilde*vecH_realizations + vecN_realizations);
        average_BER_twosided_estimator(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); 
        BER=ber(SNR);
    end
    
end


%Select a subset of training power for which we will plot markers
subset = linspace(1,length(P),5);

normalizationFactor = Nt*Nr; %Set BER normalization factor to trace(R), so that the figures show normalized BERs from 0 to 1.


%Plot the numerical results using the theoretical BER formulas
figure(1); 
hold on; box on;

plot(P,mean(average_BER_CDM_estimator(:,:,1),2)/normalizationFactor,'b:','LineWidth',2);
plot(P,mean(average_BER_CDM_estimator(:,:,1),2)/normalizationFactor,'b:','LineWidth',2);

plot(P,mean(average_BER_twosided_estimator(:,:,1),2)/normalizationFactor,'k-.','LineWidth',1);
plot(P,mean(average_BER_twosided_estimator(:,:,1),2)/normalizationFactor,'k-.','LineWidth',1);

plot(P,mean(average_BER_LMMSE_estimator(:,:,1),2)/normalizationFactor,'r-','LineWidth',1);
plot(P,mean(average_BER_LMMSE_estimator(:,:,1),2)/normalizationFactor,'r-','LineWidth',1);


plot(P,mean(average_BER_MMSE_estimator_AMP(:,:,1),2)/normalizationFactor,'b-.','LineWidth',1);
plot(P,mean(average_BER_MMSE_estimator_RM(:,:,1),2)/normalizationFactor,'k-','LineWidth',1);

legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
grid on
set(gca,'YScale','Log'); %Set log-scale on vertical axis
xlabel('Total Training Power (dB)');
ylabel('MSE');

axis([0 P(end) 0.05 1]);

title('Total Training Power vs MSE');

figure;
semilogy (SNR,BER(:,7),'r-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,8),'b-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,1),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,2),'b-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,3),'r-+','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
 semilogy (SNR,BER(:,4),'b-+','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,5),'r-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,6),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
hold off
grid on
xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
title('QPSK 128 BS antennas 32 users')

figure;

semilogy (SNR,BER(:,10),'k-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,11),'r-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,12),'k-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,13),'r-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,14),'k-o','LineWidth',1,'MarkerFaceColor','r','Markersize',7);
hold on
semilogy (SNR,BER(:,15),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,16),'k-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,17),'r-+','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold off

ylim([0.000001 1])
grid on
xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
title('QPSK 128 BS antennas 48 users')


figure;
semilogy (SNR,BER(:,21),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,22),'r-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,18),'r-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,19),'b-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,20),'r-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,23),'b-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,24),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,25),'b-o','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold off
grid on
ylim([0.0001 10])
xlim([0 12])

xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
title('16-QAM 128 BS antennas 32 users')
 
figure;
semilogy (SNR,BER(:,26),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,27),'b-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,31),'r-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,28),'b-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,30),'r-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,29),'b-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,32),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,33),'b-o','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold off
grid on
ylim([0.0001 10])
xlim([0 12])

xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','CDM T=1','CDM T=5','LMMSE T=1','LMMSE T=5');
title('16-QAM 128 BS antennas 48 users')
  


figure;
semilogy (SNR,BER(:,34),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,35),'b-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,36),'r-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,37),'b-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,38),'r-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,39),'b-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,40),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,41),'b-o','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold off
grid on
ylim([0.0001 10])
 xlim([0 14])

xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
title('64-QAM 128 BS antennas 32 users')
  

figure;
semilogy (SNR,BER(:,42),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,43),'b-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,44),'r-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,45),'b-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,46),'r-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,47),'b-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,48),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,49),'b-o','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold off
grid on
ylim([0.001 10])
% xlim([0 18])

xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
title('64-QAM 128 BS antennas 48 users')
  

figure;
semilogy (SNR,BER(:,50),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,51),'b-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,52),'r-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER(:,53),'b-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,54),'r-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,55),'b-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,56),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,57),'b-o','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold off
grid on
ylim([0.001 10])
% xlim([0 18])

xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
title('QPSK 128 BS antennas 48 users p=0.2')


figure;
subplot(1,2,1)
semilogy (SNR,BER(:,111),'k-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,111),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,311),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
ylim([0.000001 1]);
xlim([0 10]);

hold off
grid on
xlabel('SNR');
ylabel('BER');
legend('LMMSE','CDM','SD');
title('p=0')
subplot(1,2,2)
semilogy (SNR,BER(:,411),'k-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,411),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,611),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
ylim([0.000001 1]);
xlim([0 10]);

hold off
grid on
xlabel('SNR');
ylabel('BER');
legend('LMMSE','CDM','SD');
title('p=0.3')
suptitle('BER performances of QPSK modulation scheme for a 128 × 16 massive MIMO')



figure;
subplot(1,2,1)
semilogy (SNR,BER(:,72),'k-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,72),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,92),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
ylim([0.000001 1]);
xlim([0 10]);

hold off
grid on
xlabel('SNR');
ylabel('BER');
legend('LMMSE','CDM','SD');
title('p=0')
subplot(1,2,2)
semilogy (SNR,BER(:,102),'k-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,102),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,122),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
ylim([0.000001 1]);
xlim([0 10]);

hold off
grid on
xlabel('SNR');
ylabel('BER');
legend('LMMSE','CDM','SD');
title('p=0.3')
suptitle('BER performances of 16-QAM modulation scheme for a 128 × 16 massive MIMO')


figure;
subplot(1,2,1)
semilogy (SNR,BER(:,132),'k-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,132),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,152),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
ylim([0.000001 1]);
xlim([0.00000 10]);

hold off
grid on
xlabel('SNR');
ylabel('BER');
legend('LMMSE','CDM','SD');
title('p=0')
subplot(1,2,2)
semilogy (SNR,BER(:,162),'k-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,162),'r-*','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER(:,182),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
ylim([0.000001 1]);
xlim([0.00000 10]);
hold off
grid on
xlabel('SNR');
ylabel('BER');
legend('LMMSE','CDM','SD');
title('p=0.3')
suptitle('BER performances of 64-QAM modulation scheme for a 128 × 16 massive MIMO')
BER1=berr(SNR);
SNR = -4:2:14;

figure;
semilogy (SNR,BER1(:,21),'b-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER1(:,22),'r-*','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER1(:,18),'r-','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold on
semilogy (SNR,BER1(:,19),'b-','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER1(:,20),'r-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER1(:,23),'b-+','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER1(:,24),'r-o','LineWidth',1,'MarkerFaceColor','b','Markersize',7);
hold on
semilogy (SNR,BER1(:,25),'b-o','LineWidth',1,'MarkerFaceColor','k','Markersize',7);
hold off
grid on
ylim([0.0001 10])
 xlim([-4 12])

xlabel('SNR(dB)');
ylabel('BER');
legend('RM T=1','RM T=5','AMP T=1','AMP T=5','LMMSE T=1','LMMSE T=5','CDM T=1','CDM T=5');
title('16-QAM 128 BS antennas 32 users')
