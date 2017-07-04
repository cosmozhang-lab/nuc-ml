function [  ] = try_poisson(  )

l = 0.2;
t0 = 5;
k = 2;
kf = 12;
sigmoid = @(x) 1./(1+exp(-x));
f = @(t) (l.*(t-t0)).^k .* exp(-l.*(t-t0)) .* sigmoid((1000*l).*(t-t0)) ./ kf;

figure;
% for k = 3:0.1:4
% for kf = 12:24
%     f = @(t) (l.*(t-t0)).^k .* exp(-l.*(t-t0)) ./ kf;
%     tt = 0:0.01:10;
%     plot(tt,f(tt)); hold on;
% end

tt = 1:1:40;
plot(tt,f(tt));
% plot(tt, sigmoid((1000*l).*(tt-t0)) );

end

