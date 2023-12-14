clear all
close all
clc
%% 
load data/calcium/FS.mat 
%% 
figure(6)
subplot(2,1,1)
N = size(TOTMAT,1);
time = 0.1*(1:size(TOTMAT,2));
imagesc(time,1:N,TOTMAT,[0,0.1]), hold on 
colormap('hot')

lab = {'US Pre','FS','US+1','US+2','US+3','HC Pre','HC Post','HC Pre +1','HC Post +1','HC Pre +2','HC Post +2','HC Pre + 3','HC Post +3'};
dtimes = [1,ctimes];
for j = 1:length(ctimes)
plot(0.1*[ctimes(j),ctimes(j)],[0,N],'m')
text(0.1*(dtimes(j) + 0.5*(dtimes(j+1)-dtimes(j))),N+4,lab{j})
end
set(gca,'ydir','normal')
subplot(2,1,2)
plot(time,mean(TOTMAT)), hold on 
for j = 1:length(ctimes)
plot(0.1*[ctimes(j),ctimes(j)],[0,1.2*max(mean(TOTMAT))],'m')
end
xlim([0,time(end)])
%% 
[u,s,v] = svds(TOTMAT',3);
figure(3)
for j = [1,3,6,7]
plot(u(dtimes(j):dtimes(j+1),1),u(dtimes(j):dtimes(j+1),2),'color',rand(3,1)), hold on
end
legend(lab([1,3,6,7]))
title('Impact of US')
%% 
figure(4)

for j = 6:13
plot(u(dtimes(j):dtimes(j+1),1),u(dtimes(j):dtimes(j+1),2),'color',rand(3,1)), hold on
end
legend(lab(6:13))
title('Impact of US')


