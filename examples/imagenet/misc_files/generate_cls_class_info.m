load meta_clsloc.mat;
fid = fopen('cls_class_info.txt', 'wt');
fprintf(fid, 'ILSVRC2014_ID\tWNID\twords\tgloss\n');
for i=1:1000
    fprintf(fid, '%d\t%s\t%s\t%s\n', synsets(i).ILSVRC2014_ID, synsets(i).WNID, synsets(i).words, synsets(i).gloss);
end
fclose(fid);
