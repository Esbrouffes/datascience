A=imread('population-density-map.bmp');
height=size(A,1); width=size(A,2);
cell_size=150
points1=1:cell_size:width;
points2=1:cell_size:height;
figure, imshow(A)
for i=1:size(points1,2)
    hold on,
    plot([points1(1,i),points1(1,i)],[0,height],'g')
end
for i=1:size(points2,2)
    hold on,
    plot([0,width],[points2(1,i),points2(1,i)],'g')
end