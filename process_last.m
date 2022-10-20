%% The training and testing sets are divided and normalized
clc
clear
data_all=[];data_t=[];
list1=dir(['F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\yuan\2\data\','*.mat']);
len=length(list1);
for n=1:len
    str = strcat ('F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\yuan\2\data\', list1(n).name);
    % 	validation_data = dlmread(str);
    n
    data=importdata(str);
    data_a=data(:,2:end);data_b=data(:,1);
    data_all=[data_all;data_a];data_t=[data_t;data_b];
end

label_all=[];
list2=dir(['F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\yuan\2\label\','*.mat']);
len=length(list2);
for n=1:len
    str = strcat ('F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\yuan\2\label\', list2(n).name);
    % 	validation_data = dlmread(str);
    n
    data=importdata(str);
    data_a=data(:,:);
    label_all=[label_all;data_a];
end
data_all=cell2mat(data_all);

save label_all_y_1 label_all
save data_all_y_1 data_all
save data_t_c_zu data_t

%%  Read the data again
clc
clear
data_=[];label_=[];data_t=[];
list1=dir(['F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\data\','*.mat']);
len=length(list1);
for n=1:len
    str = strcat ('F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\data\', list1(n).name);
    data=importdata(str);
    data_=[data_;data];
end

list2=dir(['F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\label\','*.mat']);
len=length(list2);
for n=1:len
    str = strcat ('F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\label\', list2(n).name);
    label=importdata(str);
    label_=[label_;label];
end

list3=dir(['F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\t\','*.mat']);
len=length(list3);
for n=1:len
    str = strcat ('F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\t\', list3(n).name);
    t=importdata(str);
    data_t=[data_t;t];
end
data_all=data_;label_all=label_;

% Log normalization
data_new=data_all;
data_new=log10(data_all);
a=max(data_new,[],'all');
b=min(data_new,[],'all');
data_new=(data_new-b)/(a-b);


label_all=log10(label_all);
label_all(label_all<-1)=-2;
c=max(label_all,[],'all');
d=min(label_all,[],'all');
label_new=(label_all-d)/(c-d);

save label_new label_new
save data_new data_new

% Divide the test training set
% Produces a non-repeating number from 1 to n
mm_last=randperm(40752);

save mm_last mm_last
label_dd=label_new';
label_dd=reshape(label_dd,[1600,length(mm_last)]);
label_dd=label_dd';
test_dz=data_new(mm_last(1:8150),:);
train_dz=data_new(mm_last(8151:end),:);
test_lz=label_dd(mm_last(1:8150),:);
train_lz=label_dd(mm_last(8151:end),:);

test_lz=test_lz';test_lz=reshape(test_lz,[40,40*length(test_dz)]);test_lz=test_lz';
train_lz=train_lz';train_lz=reshape(train_lz,[40,40*length(train_dz)]);train_lz=train_lz';

save  test_dz_w test_dz
save  train_dz_w train_dz
save  test_lz test_lz
save  train_lz train_lz

i=39455;
figure
subplot 211
plot(1:208, data_all(i,:))
%     ylim([0 1]);
subplot 212
imshow((label_new((i-1)*40+1:i*40,:)),[]);
ll=label_((i-1)*40+1:i*40,:);
str0=['F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\figure\'];
if  ~exist(str0,'file')
    mkdir(str0)
end
str1=strcat('data_',num2str(i),'—',data_t{i});
str2='.emf';
saveas(gcf, [str0,str1,str2]); 
%%   porcess_prediction_test
clc
clear
load F:\02POINT\02研究工作\博士\02导电机理（ERT）\正演模拟数据\data\test_lz
load F:\02POINT\02研究工作\博士\02导电机理（ERT）\正演模拟数据\data\test_num
test=6*(test_lz)-2;  test=10.^(test);  test(test<=0.01)=0;
 result=6*(PP_result)-2; result=10.^(result);  result(result<0.01)=0;
%result=6*(y_samples_mean)-2; result=10.^(result);  result(result<0.01)=0;
last=abs(test_lz-y_samples_mean);
m=mean(last,'all');
% Statistical CC
R=[];
for i=1:8150
    r=corr2(test_lz((i-1)*40+1:i*40,:),y_samples_mean((i-1)*40+1:i*40,:));
    R=[R,r];
end
a=abs(min(R)); b=max(R); c=mean(R);
k = find(R>0.96);
% Statistical accuracy
f=last; num_1=0;
for i=1:326000
    for j=1:40
        if test(i,j)<5&&f(i,j)<=1
            f(i,j)=0;
        elseif test(i,j)>5&&f(i,j)<=5
            f(i,j)=0;
        end
    end
end
for i=1:326000
    for j=1:40
        if f(i,j)==0
            num_1=num_1+1;
        end
    end
end
arc_1=num_1/(326000*40);

result=6*(PP_result)-2; result=10.^(result);  result(result<=0.05)=0;
%result=6*(y_samples_mean)-2; result=10.^(result);  result(result<0.05)=0;
test_1=test;PP=result;
for i=1:326000
    for j=1:40
        if test_1(i,j)<=50&&test_1(i,j)>0.05
            test_1(i,j)=0.5;
        end
        if PP(i,j)<=50&&PP(i,j)>0.1
            PP(i,j)=0.5;
        end
    end
end
test_1(test_1>=50)=1;PP(PP>=10)=1;
PP_CNN=PP;

num_2=0;
for i=1:326000
    for j=1:40
        if test_1(i,j)==PP(i,j)
            num_2=num_2+1;
        end
    end
end
arc_2=num_2/(326000*40);

% N0=find(test_1==0);N1=find(test_1==0.5);N2=find(test_1==1);
% num_0=0;num_1=0;num_2=0;
%  for i=1:326000
%     for j=1:40
%         if test_1(i,j)==PP(i,j)&&test_1(i,j)==0
%             num_0=num_0+1;
%         end
%         if test_1(i,j)==PP(i,j)&&test_1(i,j)==0.5
%             num_1=num_1+1;
%         end
%         if test_1(i,j)==PP(i,j)&&test_1(i,j)==1
%             num_2=num_2+1;
%         end
%     end
% end
% nn=test_1-PP;

% The mean of the variances of the different categories
k1 = find(test<0.1);
k2 = find(test>0.1 & test<10);
k3 = find(test>10);
std = reshape(y_samples_std,[13040000,1]);
std_11=std(k1);std_22=std(k2);std_33=std(k3);
m1=mean(std_11);m2=mean(std_22);m3=mean(std_33);
% The mean of the errors for different categories
result=6*(y_samples_mean)-2; result=10.^(result);  result(result<0.01)=0;
last=abs(test-result);
last = reshape(last,[13040000,1]);
er_11=last(k1);er_22=last(k2);er_33=last(k3);
e1=mean(er_11);e2=mean(er_22);e3=mean(er_33);
ee1=6*(e1)-2;ee1=10.^(ee1);ee2=6*(e2)-2;ee2=10.^(ee2);ee3=6*(e3)-2;ee3=10.^(ee3);

%  The variance of six sample types was counted
ST=[];
for i=1:8150
    st=mean(y_samples_std((i-1)*40+1:i*40,:),'all');
    ST=[ST,st];
end
kk1 = find(test_num<=7920);kk2 = find(test_num<=15408 & test_num>=7921);
kk3 = find(test_num<=22896 & test_num>=15409);kk4 = find(test_num<=26928 & test_num>=22897);
kk5 = find(test_num<=32113 & test_num>=26929);kk6 = find(test_num<=40753 & test_num>=32114);
ST_11=mean(ST(kk1));ST_22=mean(ST(kk2));ST_33=mean(ST(kk3));
ST_44=mean(ST(kk4));ST_55=mean(ST(kk5));ST_66=mean(ST(kk6));

% Statistical variance of different types of different samples
test_1=[];std_1=[];test_2=[];std_2=[];test_3=[];std_3=[];
test_4=[];std_4=[];test_5=[];std_5=[];test_6=[];std_6=[];
kk1 = find(test_num<=7920);kk2 = find(test_num<=15408 & test_num>=7921);
kk3 = find(test_num<=22896 & test_num>=15409);kk4 = find(test_num<=26928 & test_num>=22897);
kk5 = find(test_num<=32113 & test_num>=26929);kk6 = find(test_num<=40753 & test_num>=32114);
for i=kk1
    test_=test((i-1)*40+1:i*40,:);std_=y_samples_std((i-1)*40+1:i*40,:);
    test_1=[test_1;test_];std_1=[std_1;std_];
end
k11 = find(test_1<0.1);k12 = find(test_1>0.1 & test_1<10);k13 = find(test_1>10);
std = reshape(std_1,[length(kk1)*40*40,1]);std_11=mean(std(k11));std_12=mean(std(k12));std_13=mean(std(k13));
for i=kk2
    test_=test((i-1)*40+1:i*40,:);std_=y_samples_std((i-1)*40+1:i*40,:);
    test_2=[test_2;test_];std_2=[std_2;std_];
end
k21 = find(test_2<0.1);k22 = find(test_2>0.1 & test_2<10);k23 = find(test_2>10);
std = reshape(std_2,[length(kk2)*40*40,1]);std_21=mean(std(k21));std_22=mean(std(k22));std_23=mean(std(k23));
for i=kk3
    test_=test((i-1)*40+1:i*40,:);std_=y_samples_std((i-1)*40+1:i*40,:);
    test_3=[test_3;test_];std_3=[std_3;std_];
end
k31 = find(test_3<0.1);k32 = find(test_3>0.1 & test_3<10);k33 = find(test_3>10);
std = reshape(std_3,[length(kk3)*40*40,1]);std_31=mean(std(k31));std_32=mean(std(k32));std_33=mean(std(k33));
for i=kk4
    test_=test((i-1)*40+1:i*40,:);std_=y_samples_std((i-1)*40+1:i*40,:);
    test_4=[test_4;test_];std_4=[std_4;std_];
end
k41 = find(test_4<0.1);k42 = find(test_4>0.1 & test_4<10);k43 = find(test_4>10);
std = reshape(std_4,[length(kk4)*40*40,1]);std_41=mean(std(k41));std_42=mean(std(k42));std_43=mean(std(k43));
for i=kk5
    test_=test((i-1)*40+1:i*40,:);std_=y_samples_std((i-1)*40+1:i*40,:);
    test_5=[test_5;test_];std_5=[std_5;std_];
end
k51 = find(test_5<0.1);k52 = find(test_5>0.1 & test_5<10);k53 = find(test_5>10);
std = reshape(std_5,[length(kk5)*40*40,1]);std_51=mean(std(k51));std_52=mean(std(k52));std_53=mean(std(k53));
for i=kk6
    test_=test((i-1)*40+1:i*40,:);std_=y_samples_std((i-1)*40+1:i*40,:);
    test_6=[test_6;test_];std_6=[std_6;std_];
end
k61 = find(test_6<0.1);k62 = find(test_6>0.1 & test_6<10);k63 = find(test_6>10);
std = reshape(std_6,[length(kk6)*40*40,1]);std_61=mean(std(k61));std_62=mean(std(k62));std_63=mean(std(k63));

% x=linspace(0,1,501)';
% y1=normpdf(x,0.2857,0.0545);y2=normpdf(x,0.3011,0.0716);y3=normpdf(x,0.3243,0.0691);
% plot(x,y1,'-',x,y2,'-.',x,y3,'--')
% 
% x=linspace(-0.15,0.15,501)';
% y1=normpdf(x,0.0251,0.0168);y2=normpdf(x,0.0233,0.0211);y3=normpdf(x,0.0261,0.0225);
% plot(x,y1,'-',x,y2,'-.',x,y3,'--')
% 
% x=linspace(-0.001,0.001,501)';
% y1=normpdf(x,0.00045,0.00014);y2=normpdf(x,0.00031,0.00015);y3=normpdf(x,0.00031,0.00021);
% plot(x,y1,'-',x,y2,'-.',x,y3,'--')
%% figure
% grayscale
mean_3=PP;
y_samples_mean_1=y_samples_mean;
i=21
figure
subplot 211
imshow((test_lz((i-1)*40+1:i*40,:)),[]);
subplot 212
imshow((y_samples_std((i-1)*40+1:i*40,:)),[]);
nn=y_samples_std((i-1)*40+1:i*40,:);


cc=y_samples_mean-mean_1;
PP_CNN=PP_result;
mean_1=y_samples_mean;
std_1=y_samples_std;
std_1(std_1<0.01)=0;
for i=1:length(test_lz)/40
    i=1996
    figure
    subplot 131
    surf(test_lz((i-1)*40+1:i*40,:));shading interp;view(0,90);
    set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
    set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
    caxis ([0 1]);
%     imshow((test_1((i-1)*40+1:i*40,:)),[]);
    subplot 132
    surf(mean_1((i-1)*40+1:i*40,:));shading interp;view(0,90);
    set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
    set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
    caxis ([0 1]);
%     imshow((mean_1((i-1)*40+1:i*40,:)),[]);
    subplot 133
    surf(std_1((i-1)*40+1:i*40,:));shading interp;view(0,90);
    set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
    set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
    caxis ([0 0.15]);
%     imshow((std_1((i-1)*40+1:i*40,:)),[]);
%     subplot 241
%     imshow((test_1((i-1)*40+1:i*40,:)),[]);
%     subplot 242
%     imshow((PP_1((i-1)*40+1:i*40,:)),[]);
%     subplot 243
%     imshow((PP_2((i-1)*40+1:i*40,:)),[]);
%     subplot 244
%     imshow((PP_3((i-1)*40+1:i*40,:)),[]);
%     subplot 245
%     imshow((PP_CNN((i-1)*40+1:i*40,:)),[]);
%     subplot 246
%     imshow((mean_1((i-1)*40+1:i*40,:)),[]);
%     subplot 247
%     imshow((mean_2((i-1)*40+1:i*40,:)),[]);
%     subplot 248
%     imshow((mean_3((i-1)*40+1:i*40,:)),[]);
    str0=['F:\02POINT\02研究工作\博士\02导电机理（ERT）\正演模拟数据\data\uncertainty_new\'];
    if  ~exist(str0,'file')
        mkdir(str0)
    end
    str1=strcat('test_',num2str(i));
    str2='.jpg';
    saveas(gcf, [str0,str1,str2]); 
    close
end

mm=test_lz((i-1)*40+1:i*40,:);
nn=PP_result((i-1)*40+1:i*40,:);
ll=abs(mm-nn);

for i=5475:length(test_lz)/40
    figure
    subplot 211
    imshow((test_lz((i-1)*40+1:i*40,:)),[]);
    subplot 212
    imshow((PP_result((i-1)*40+1:i*40,:)),[]);
    str0=['F:\02POINT\02研究工作\博士\02导电机理（ERT）\正演模拟数据\data\3\'];
    if  ~exist(str0,'file')
        mkdir(str0)
    end
    str1=strcat('test_',num2str(i));
    str2='.jpg';
    saveas(gcf, [str0,str1,str2]); %保存当前窗口的图像
    close
end

i=1;
% RGB
Z1 = test_1((i-1)*40+1:i*40,:);Z2 = PP((i-1)*40+1:i*40,:);
PP_3=PP_result;
mean_3=y_samples_mean;

for i=1:length(test_lz)/40
i=215;
figure
subplot 241
surf(test_1((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);
subplot 242
surf(PP_1((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);
subplot 243
surf(PP_2((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);
subplot 244
surf(PP_3((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);
subplot 245
surf(PP_CNN((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);
subplot 246
surf(mean_1((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);
subplot 247
surf(mean_2((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);
subplot 248
surf(mean_3((i-1)*40+1:i*40,:));shading interp;view(0,90);
set(gca,'xtick',[],'xticklabel',[],'xcolor','w');
set(gca,'ytick',[],'yticklabel',[],'ycolor','w');
caxis ([0 1]);

str0=['F:\02POINT\02研究工作\博士\02导电机理（ERT）\正演模拟数据\data\result_4\'];
    if  ~exist(str0,'file')
        mkdir(str0)
    end
    str1=strcat('test_',num2str(i));
    str2='.jpg';
    saveas(gcf, [str0,str1,str2]); %保存当前窗口的图像
    close
end
subplot
surf(Z1);shading interp
subplot 122
surf(Z2);shading interp

i=1;
xx=test_dz(i,:);
plot(xx);
x1=reshape(xx,[13,16]);
plot(x1);
%%
clc
clear
load C:\Users\lan\Desktop\test\train_dz.mat
load C:\Users\lan\Desktop\test\train_lz.mat
load C:\Users\lan\Desktop\test\test_dz.mat
load C:\Users\lan\Desktop\test\test_lz.mat
load C:\Users\lan\Desktop\test\train_t.mat
load C:\Users\lan\Desktop\test\test_t.mat

for i=1:length(test_lz)/40
    figure
    subplot 211
    plot(1:208, test_dz(i,:))
    %     ylim([0 1]);
    subplot 212
    imshow((test_lz((i-1)*140+1:i*40,:)),[]);
    str0=['F:\02POINT\02研究工作\博士\02导电机理（ERT）\last\test\'];
    if  ~exist(str0,'file')
        mkdir(str0)
    end
    str1=strcat('test_',num2str(i),'—',test_t{i});
    str2='.jpg';
    saveas(gcf, [str0,str1,str2]); 
    close
end

for i=1:length(train_lz)/40
    figure
    subplot 211
    plot(train_dz(i,:))
    ylim([0 1]);
    subplot 212
    imshow((train_lz((i-1)*100+1:i*100,:)),[]);
    str0=['C:\Users\lan\Desktop\test\figure\train\'];
    if  ~exist(str0,'file')
        mkdir(str0)
    end
    str1=strcat('train_',num2str(i),'—',train_t{i});
    str2='.jpg';
    saveas(gcf, [str0,str1,str2]); 
    close
end
aa=test_dz';
bb=aa(:,1:16);