A=imread('population-density-map.bmp');
B=imread('elevation1x1_new-mer-bleue.bmp');
%imshow(A)
x1=...
[1039, 2365; %Spain Cabo
1735,353; %Dinamarca tip
2580,2247; %Italy taco
3727,1718; %MarNegro Sebastopol
584,1134; %MarNegro Sebastopol
236,2555; %MarNegro Sebastopol
893,1834; %MarNegro Sebastopol
1269,1895;
687,267;
4451,1927;
4064,2708;
1138,1048;
2478,238]; %MarNegro Sebastopol


%imshow(B)
x2=...
[966,3540;
2114, 1616;
2532,3645;
3729,3021;
877,2188;
145,3466;
948,2992;
1308,3123;
1255,1300;
4403,3070;
3994,3977;
1394,2254;
2790,1518];

figure, imshow(A), hold on , plot(x1(:,1),x1(:,2),'ro')
figure, imshow(B), hold on , plot(x2(:,1),x2(:,2),'ro')
match_plot(A,B,x1,x2)

% Computing Homography
tform = cp2tform(x2,x1,'polynomial')


% Computing Left Warped Image
info = imfinfo('population-density-map.bmp');

wim1 = imtransform(B,tform,...
    'XData',[1 info.Width],'YData',[1 info.Height]);

figure,imshowpair(A,wim1,'blend'),hold on , plot(x1(:,1),x1(:,2),'go');

imwrite(wim1,'elevation1x1_new-mer-bleue_registred.bmp')
