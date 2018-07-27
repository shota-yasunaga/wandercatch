% Needs to have a structure chan_loc
% it has to have pos and labels

loc_file = '/Users/macbookpro/Dropbox/College/TsuchiyaLab/wandercatch/EEG/chan_loc.xyz';

num_channels = length(chan_loc.labels);

f = fopen(loc_file,'w');

for i = 1:num_channels
    format = '%d %f %f 0 %s\n';
    fprintf(f,format,i,chan_loc.pos(i,:),char(chan_loc.labels(i)));
end


fclose(f);