for i = 1:size(Matrices, 1)
    for j = 1:size(Matrices, 2)
        [W_all{i,j}, lambda_all(i,j)] = priority_vector(Matrices{i,j});
    end
end

for i = 1:size(lambda_all, 1)
    for j = 1:size(lambda_all, 2)
        [CI_all(i,j), CR_all(i,j), isConsistent_all(i,j)] = ...
            check_consistency_3(lambda_all(i,j));
    end
end