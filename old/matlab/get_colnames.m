function [name_meta,name_smart] = get_colnames(data)
    column_names = data.Properties.VariableNames;
    name_meta = column_names(find(~contains(column_names,'smart')));
    name_smart = column_names(find(contains(column_names,'smart')));
end