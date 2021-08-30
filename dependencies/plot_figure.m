function plot_figure(ppg,Fs,p,delta)
% Figure to plot the predicted point by point estimation of RR intervals
% ppg: single PPG signal
% Fs: Sampling rate of PPG
% p: Model order
% delta: time increment in updating parameters (in seconds)

    [~, peak_times]=findpeaks(-ppg,'MinpeakDistance', Fs / 2, 'MinPeakProminence', 100,'MinPeakHeight',100);
    peak_times = peak_times./Fs;
    [Thetap,Kappa,opt_old] = regr_likel(peak_times,p);
    [Mu,opt] = pplikel_ASR(peak_times,[opt_old.Theta0;Thetap],delta,p);
    t = opt.t0 + (0:length(Mu)-1) * opt.delta;
    figure; hold on

    plot(t, 1000*Mu,'linewidth',1.5)
    plot(peak_times(2:end), 1000*diff(peak_times), '*')
    legend( 'HDIG estimated interval','True RR interval')
    xlabel('time [s]')
    ylabel('Interval [ms]')
    set(gca,'FontName','Times New Roman','fontsize',12)
    set(gca, 'XColor', [0 0 0], 'YColor', [0 0 0])
end