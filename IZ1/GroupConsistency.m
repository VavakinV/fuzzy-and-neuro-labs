clearvars; clc;
run('InputData.m');
N_groups = 3;
N_terms = 3;
N_matrices = numel(Matrices);
% Вычисление весов и lambda для каждой матрицы
W_all = zeros(N_matrices, N_terms);
lambda_all = zeros(N_matrices, 1);
isConsistent_mat = false(N_matrices,1);

for k = 1:N_matrices
    A = Matrices{k};
    [W, lambda_max] = priority_vector(A);
    [CI, CR, isConsistent] = check_consistency_3(lambda_max);

    W_all(k, :) = W(:)';          
    lambda_all(k) = lambda_max;
    isConsistent_mat(k) = isConsistent;
    
    fprintf('Matrix %2d: lambda_max=%.4f, CI=%.4f, CR=%.4f, consistent=%d\n', ...
        k, lambda_max, CI, CR, isConsistent);
end

% Сопоставление "матрица -> группа"
Active_Group_Index = zeros(N_groups*N_terms, 1);
idx = 1;
for g = 1:N_groups
    for t = 1:N_terms
        Active_Group_Index(idx) = g;
        idx = idx + 1;
    end
end

% Инициализация
Group_Status = ones(N_groups,1);
Group_Final_W = cell(N_groups,1);
Final_Group_Status = Group_Status;
active_groups_count = sum(Group_Status);

fprintf('\nПроверка согласованности групп:\n');

for i = 1:N_groups
    if Final_Group_Status(i) == 0
        fprintf('Группа %d уже исключена\n', i);
        continue;
    end
    
    group_indices = (Active_Group_Index == i);
    Group_W = W_all(group_indices, :);
    m = size(Group_W, 1);
    n = size(Group_W, 2);
    
    if m == 0
        Final_Group_Status(i) = 0;
        fprintf('Группа %d: нет активных экспертов — исключена\n', i);
        continue;
    end
    
    fprintf('\nГруппа %d: %d активных экспертов\n', i, m);
    
    % Вычисление CV
    mean_W = mean(Group_W, 1);
    std_W = std(Group_W, 0, 1);
    CV = (std_W ./ mean_W) * 100;
    
    fprintf('Коэффициент вариации (T1..T3):\n');
    disp(CV);
    
    CV_fail = (sum(CV > 33) >= 2);
    
    Ranks = tiedrank(Group_W')';
    
    % Коэффициент конкордации
    R_sum = sum(Ranks, 1);                     
    S = sum( (R_sum - mean(R_sum)).^2 );
    W_kendall = (12 * S) / (m^2 * (n^3 - n));
    
    fprintf('Коэффициент конкордации (W): %.3f\n', W_kendall);
    
    if W_kendall < 0.50 || CV_fail
        Final_Group_Status(i) = 0;
        active_groups_count = active_groups_count - 1;
        fprintf('Группа %d исключена (W < 0.50 или CV_fail)\n', i);
    else
        Group_Final_W{i} = mean_W;
        fprintf('Группа %d согласована\n', i);
    end
    
    fprintf('Активных групп сейчас: %d\n', active_groups_count);
end

fprintf('\nИтог (Final_Group_Status):\n');
disp(Final_Group_Status');

