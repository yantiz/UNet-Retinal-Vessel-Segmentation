matList = dir('mat_files/*.mat');

for i = 1:length(matList)
    matfile = matList(i);
    
    data = load(fullfile(matfile.folder, matfile.name));
    data = data.data;
    rect = data.rect;
    
    name_cell = split(matfile.name, '.');
    pred_name = strcat(name_cell{1}, '_pred.bmp');
    pred = imread(fullfile('results', pred_name(5:end)));
    
    lower_height = rect(1)-rect(5);
    upper_height = rect(2)+rect(5);
    lower_width = rect(3)-rect(5);
    upper_width = rect(4)+rect(5);
    
    if lower_height < 1
        lower_height = 1;
    end
    if lower_width < 1
        lower_width = 1;
    end
    if upper_height > rect(6)
        upper_height = rect(6);
    end
    if upper_width > rect(7)
        upper_width = rect(7);
    end
    
    %pred = pred(rect(1)-rect(5):rect(2)+rect(5),rect(3)-rect(5):rect(4)+rect(5),:);
    pred = pred(lower_height:upper_height, lower_width:upper_width, :);
    
    pred_artery = pred(:,:,1) / 255;
    pred_background = pred(:,:,2) / 255;
    pred_vein = pred(:,:,3) / 255;
    
    for j = 1:length(data.tree.vessel)
        vs = data.tree.vessel(j);
        
        count_a = 0;
        count_b = 0;
        count_v = 0;
        for k = 1:length(vs.pixels)
            p = vs.pixels(k,:);
            result_a = pred_artery(p(1,2), p(1,1));
            result_b = pred_background(p(1,2), p(1,1));
            result_v = pred_vein(p(1,2), p(1,1));
            
            if result_a == 1
                count_a = count_a + 1;
            elseif result_v == 1
                count_v = count_v + 1;
            else
                count_b = count_b + 1;
            end
        end
        
        count_max = max([count_a, count_b, count_v]);
        
        if count_max == count_a
            vs.type = 1;
        elseif count_max == count_v
            vs.type = -1;
        else
            vs.type = 0;
        end
        
        data.tree.vessel(j) = vs;
    end
   
    new_name = strcat(name_cell{1}, '_pred.', name_cell{2});
    save(fullfile('results', new_name), 'data');
end