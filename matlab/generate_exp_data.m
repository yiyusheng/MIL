function [train,test] = generate_exp_data(data,selected_features,sn_folds,fold_id)
    [name_meta,name_smart] = get_colnames(data);
    name_train_test = [name_meta,selected_features.name_smart'];
    
    sn_train = sn_folds.serial_number(sn_folds.(fold_id)==1);
    sn_test = sn_folds.serial_number(sn_folds.(fold_id)==0);
    
    train = data(ismember(data.serial_number,sn_train),name_train_test);
    test = data(ismember(data.serial_number,sn_test),name_train_test);

    st = dbstack;
    fprintf('[%s]%s end...\n', datetime,st.name)    
end