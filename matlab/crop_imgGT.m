clear all;
close all;
%path='E:\fjj\SeaShips_SMD\JPEGImages\MVI_1478_VIS_00405.jpg';
path='E:\paper\半监督船舶检测en\复杂场景图1\003426.jpg';%MVI_1478_VIS_00405.jpg 002365.jpg MVI_1486_VIS_00013.jpg MVI_1474_VIS_00429.jpg 003426.jpg 
img=imread(path);
[H,W,C]=size(img);
figure(1);
imshow(img)
h=imrect(gca,[10,10,W/10,H/10]);
setFixedAspectRatioMode(h,true);
while true
pos=wait(h)

% w = waitforbuttonpress;
%pos = getPosition(h)
col=round(pos(1)) : round(pos(1)+pos(3));  %根据pos计算行下标
row=round(pos(2)) : round(pos(2) + pos(4));   %根据pos计算列下标
subwin=img(row,col,:);
figure(2);

imshow(subwin);
if strcmpi(get(gcf,'CurrentCharacter'),'y')
   break; 
end
end
% rectangle('Position',[1,1,300,200])
