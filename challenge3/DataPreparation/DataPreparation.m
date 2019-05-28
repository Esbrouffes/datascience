im_popu=imread('../ImageRegistration/population-density-map.bmp');
im_elevation=imread('../ImageRegistration/elevation1x1_new-mer-bleue_registred.bmp');
load('MyColormap_64.mat');
im_popu=imcrop(im_popu,[181.5 0.5 4649 2842]);
im_elevation=imcrop(im_elevation,[181.5 0.5 4649 2842]);

[IND] = rgb2ind(im_elevation,mymap);
%imwrite(IND,'elevation_image_64levels.bmp');

% figure
% imagesc(IND)
% colormap(mymap)
% colormapeditor
% ax = gca;
% mymap = colormap(ax);
% save('MyColormap_64','mymap')
% 
% figure, imshow(IND,[]);
% 
% %Creating array of cells ponderated
% kr = 15; kc = 15; % Row and Column sizes of sub-block
% [m n] = size(IND); mk = floor(m/kr); nk = floor(n/kc); % Size of input and result
% IND_cells = reshape(sum(reshape(reshape(sum(reshape(IND,kr,[])),mk,[]).',kc,[])),nk,mk).'/(kr*kc);


depth_new=IND(1:2835,1:4635);
kr = 15;
kc = 15;
[m, n] = size(depth_new);
mk = m/kr;
nk = n/kc;
IND_cells  = reshape(sum(reshape( ...
reshape(sum(reshape(depth_new,kr,[])), mk,[]).', kc, [])), nk, mk).' / (kr*kc);

figure,imshow(IND_cells);
imwrite(IND_cells,'depth_cells.bmp');
elevation_cells=double(IND_cells)*4810/63;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Analisis Impopulation
im_popu_gray=im_popu(:,:,1);
imshow(im_popu_gray)

im_popu_new=im_popu_gray(1:2835,1:4635);
kr = 15;
kc = 15;
[m, n] = size(im_popu_new);
mk = m/kr;
nk = n/kc;
popu_cells  = reshape(sum(reshape( ...
reshape(sum(reshape(im_popu_new,kr,[])), mk,[]).', kc, [])), nk, mk).' / (kr*kc);
popu_cells=popu_cells*3000/256;
%imagesc(popu_cells)
csvwrite('popu_cells.csv',popu_cells);