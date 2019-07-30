function data = load_file(path)
    data = readtable(path,'Delimiter',',');
    fprintf('read %s success...\n',path);
end