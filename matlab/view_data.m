function [] = view_data()

dataset = load('../../datasets/TNT.mat');
data = dataset.data;
labels = dataset.labels;

datasize = size(data);
imsize = datasize(2:3);
imchannel = datasize(4);
total = datasize(1);

% close all;

figure;
for ch = 1:imchannel
    subplot(imchannel,1,ch);
    imagesc(reshape(data(3,:,:,ch),imsize));
    title(sprintf('Channel %d',ch));
end

figure;
spn = 4;
subplot(spn,1,1);
plot(reshape(data(1,:,959,2),[1,40]));
subplot(spn,1,2);
plot(diff(reshape(data(1,:,959,2),[1,40])));
subplot(spn,1,3);
plot(reshape(data(1,:,645,1),[1,40]));
subplot(spn,1,4);
plot(reshape(data(1,:,645,2),[1,40]));

end

