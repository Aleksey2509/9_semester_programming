%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc
c=3e8; %speed of light
fc=77e9; %carrier freq
lambda = c/fc;
deltaF=300e6; %sweep freq
T=40e-6; %one period
alph=deltaF/T; %sweep rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R=[24 65 65]; %initial distance of the target
v=[-15 0 0]; %speed of the target
SNRdb = [100 100 100]; % signal-to-noise ratio
Azimuth = [-25 10 -25]; % azimuth angle, degrees
Elevation = [0 0 0];% elevation angle, degrees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D=128; % #of doppler cells OR #of sent periods
N=512; %for length of time

Ant_pos = (0:0.5:5.5)*lambda; % antennas coordinates
Ant_pos = [Ant_pos; Ant_pos*0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SNR = 10.^(SNRdb./20);
Ntargets = length(R);
Nant = size(Ant_pos, 2);
t=linspace(0,D*T,D*N); %total time
nT=length(t)/D; %length of one period
a=zeros(1,length(t)); %transmitted signal
b=zeros(1,length(t)); %received signal
r_t=zeros(1,length(t));
ta=zeros(1,length(t));
M4 = zeros(2*D, N, Nant);
for nant = 1:Nant
    b = (D+N)/7.0*randn(1,length(t));
    for ntar = 1:Ntargets
        r1=R(ntar);
        v1=v(ntar);
        SNR1 = SNR(ntar);
        az1 = Azimuth(ntar);
        el1 = Elevation(ntar);
        psi = 2*pi/lambda*(Ant_pos(1, nant)*sind(az1)*cosd(el1)+Ant_pos(2, nant)*sind(az1)*sind(el1));
        n=0;
        for i=1:length(t)
            r_t(i)=r1+v1*t(i); % range of the target in terms of its velocity and initial range
            ta(i)=2*r_t(i)/c; % delay for received signal
            if i>n*nT && i<=(n+1)*nT % doing this for D of periods (nt length of pulse)
                a(i)=sin(2*pi*(fc*t(i)+.5*alph*t(i)^2-alph*t(i)*n*T)); %transmitted signal
                b(i)=b(i)+SNR1*sin(2*pi*(fc*(t(i)-ta(i))+.5*alph*(t(i)-ta(i))^2-alph*(t(i)-ta(i))*n*T) + psi); %received signal
            else
                n=n+1;
                a(i)=sin(2*pi*(fc*t(i)+.5*alph*t(i)^2-alph*t(i)*n*T)); %transmitted signal
                b(i)=b(i)+SNR1*sin(2*pi*(fc*(t(i)-ta(i))+.5*alph*(t(i)-ta(i))^2-alph*(t(i)-ta(i))*n*T)+psi); %received signal
            end
        end
    end
    mixed1=(a.*b); %video signal OR IF signal (output of mixer)
    m1=reshape(mixed1,length(mixed1)/D,D); %generating matrix ---> each row showing range info for one period AND each column showing number of periods
    [My,Ny]=size(m1');
    win=hamming(Ny);
%    win = taylorwin(Ny,3,-20);
    m2=conj(m1).*(win*ones(1,My)); %taking conjugate and applying window for sidelobe reduction (in time domain)
    M2=(fft(m2,2*N))/N*2; %First FFT for range information
    M3=fftshift(fft(M2',2*D)/D*2); %Second FFT for doppler information
    M4(:,:,nant) = M3(:,N+(1:N));%positive range values
end

% non-cogerent integration
Mnci = 0;
for nant = 1:Nant
    Mnci = Mnci + abs(M4(:,:,nant)).^2;
end
mean(mean(Mnci))

[My,Ny]=size(Mnci);
doppler=linspace(-D,D,My)*c/D/T/fc/4;
range=linspace(0,Ny-1,Ny)*c/2/deltaF/2;
figure;
contour(range,doppler,Mnci);grid on
xlabel('Range, m')
ylabel('Doppler, m/s')
% return
figure;mesh(range,doppler,10*log10(Mnci))
xlabel('Range, m')
ylabel('Doppler, m/s')
figure;
plot(range,10*log10(Mnci(D+1,:)));
grid on;
xlabel('Range,m')

%% 


qw = CFAR_CA(Mnci); %% create by yourself

figure(1);
hold on;
scatter(range(qw(:,2)),doppler(qw(:,1)),'ro');
hold off;
legend('R-V data', 'CFAR detections')
title('CFAR CA')
% return

%%

qw = CFAR_OS(Mnci); %% create by yourself

figure;
contour(range,doppler,Mnci);grid on
xlabel('Range, m')
ylabel('Doppler, m/s')
hold on;
scatter(range(qw(:,2)),doppler(qw(:,1)),'ro');
hold off;
legend('R-V data', 'CFAR detections')
title('CFAR OS')
% return

%%
% Ndet = size(qw,1);
% theta = zeros(Ndet,1);
% rho = zeros(Ndet,1);
% lateral = zeros(Ndet,1);
% longitudinal = zeros(Ndet,1);
% for det = 1:Ndet
%     x = squeeze(M4(qw(det,1),qw(det,2),:));
%     theta(det) = DoA_ML(x); % create by yourself
%     rho(det) = range(qw(det,2));
%     longitudinal(det) = rho(det)*cosd(theta(det));
%     lateral(det) = rho(det)*sind(theta(det));
% end
% figure;
% scatter(lateral,longitudinal,'r.');
% ylim([0 80])
% axis('equal')
% grid on;
% xlabel('Lateral, m')
% ylabel('Longitudinal, m')
% axis('equal')