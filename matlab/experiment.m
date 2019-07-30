% Experiment:
%     1. generate training set and testing set
%     2. build MIL model on training set
%     3. predict the result for validating set
%     4. evaluate the predicted result

%## MAIN
fprintf('[%s]%s start...\n', datetime,'main')
    
path_model = fullfile(getenv("HOME"),'/Data/backblaze/model_file/');
path_preprocess = fullfile(getenv("HOME"),'/Data/backblaze/model_preprocess/');
model_names = dir(path_model);

for mn = model_names(4)
    data = load_file(fullfile(path_model,'1test'));
    selected_features = load_file(fullfile(path_preprocess,strcat('selected_features_',mn.name)));
    sn_folds = load_file(fullfile(path_preprocess,strcat('sn_folds_',mn.name)));
    
    for i = 0:4
        fold_id = strcat('fold',num2str(i));
        [train,test] = generate_exp_data(data,selected_features,sn_folds,fold_id);
        
        model_mil = build_model_mil(train);
        predicted_result_mil = predict_model_mil(model_mil,test);
        perf_mil = evaluate_model_mil(predicted_result_mil);
    end
    
end

fprintf('[%s]%s end...\n', datetime,'main')    
