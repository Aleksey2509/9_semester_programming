clear all; clc;

snr_db_arr = 10 : 20;


figure;
for i = 1 : length(snr_db_arr)
    [FPR, TPR] = get_stats_for_ROC(snr_db_arr(i));
    
    plot(FPR, TPR);
    xlabel('FPR');
    ylabel('TPR');
    title('ROC curve');

    grid on;
    hold on;

end
    legend('SNR = 10', 'SNR = 11','SNR = 12','SNR = 13','SNR = 14', ...
        'SNR = 15','SNR = 16','SNR = 17','SNR = 18','SNR = 19', 'SNR = 20');

function [FPR, TPR] = get_stats_for_ROC(snr_dB)
    lambdas = 0 : 0.01 : 1;
    lambdas_len = length(lambdas);
    TPR = zeros(1, lambdas_len);
    FPR = zeros(1, lambdas_len);
    noise_pwr = 10 ^ (-snr_dB / 10);
    for i = 1 : lambdas_len
        lamb = lambdas(i);
    
        % TPR(i) = marcumq(1 / sqrt(noise_pwr), lamb);
        % FPR(i) = exp(-lamb * lamb / (2 * noise_pwr)) * sqrt(noise_pwr);
        
        TPR(i) = erfc((lamb - 1) / sqrt(2 * noise_pwr)) / 2;
        FPR(i) = erfc(lamb / sqrt(2 * noise_pwr)) / 2;
    end
end