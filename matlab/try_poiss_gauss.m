function [  ] = try_poiss_gauss(  )

p_l = 0.3; % poiss_ratio
p_s0 = 0; % poiss_shift, this should always be 0
p_k = 4; % poiss_index
p_kf = 12; % 1/poiss_amp, this is redundant with gauss_amp
p_b = 0.2; % poiss_bias
g_t0 = 800; % gauss_shift
g_s = 10; % gauss_sig
g_a = 1; % gauss_amp

f_p = @(s) (p_l.*(s-p_s0)).^p_k .* exp(-p_l.*(s-p_s0)) ./ p_kf + p_b;
f_g = @(t) g_a .* exp(-(t-g_t0).^2 ./ (2*g_s^2));

f = @(t,s) f_g(t) .* f_p(s);

tt = 1:2000;
ss = 1:40;

data = zeros( max(size(ss)), max(size(tt)) );
for si = 1:max(size(ss))
    s = ss(si);
    data(si,:) = f(tt,s);
end

figure;
imagesc(data);

end

