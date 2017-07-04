function [  ] = view_kernels(  )

w = load('w.mat');
kernel = w.kernel;

ksize = size(kernel);
ksize = ksize(1:2);

figure;
itercnt = 0;
for i = 1:5
    for j = 1:5
        itercnt = itercnt + 1;
        subplot(5,5,itercnt);
        imagesc(reshape(kernel(:,:,1,itercnt),ksize));
        axis off;
    end
end

end

