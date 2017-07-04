function [  ] = try_vargaussian(  )

s0 = 10;
sig = 5;
bs = 0.6;
idx = 0.9;
s1 = 0;
sp1 = 0.1;
amp = 1;

f = @(s) amp * exp( - (bs.^(s-s1) - s0).^2 ./ (2*sig^2) );

figure;
ss = 1:40;
% plot(ss,f(ss));
plot(ss,bs.^(sp1.*ss-s1));

end

