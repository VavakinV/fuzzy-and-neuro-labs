Student_Labels = {'Вася', 'Саша', 'Женя', 'Настя', 'Маша'};
Term_Labels = {'Low', 'Mid', 'High'};
N_students = length(Student_Labels);
N_terms = length(Term_Labels);

Group_W_Matrix = zeros(N_students, N_terms, 3);

Group_W_Matrix(:, 1, 1) = [0.15; 0.30; 0.20; 0.25; 0.10]; %T1 (группа 1)
Group_W_Matrix(:, 2, 1) = [0.15; 0.35; 0.20; 0.20; 0.10]; %T2 (группа 1)
Group_W_Matrix(:, 3, 1) = [0.15; 0.15; 0.20; 0.25; 0.25]; %T3 (группа 1)

Group_W_Matrix(:, 1, 2) = [0.10; 0.35; 0.25; 0.20; 0.10]; %T1 (группа 2)
Group_W_Matrix(:, 2, 2) = [0.20; 0.25; 0.20; 0.15; 0.20]; %T2 (группа 2)
Group_W_Matrix(:, 3, 2) = [0.10; 0.20; 0.20; 0.25; 0.25]; %T3 (группа 2)

Group_W_Matrix(:, 1, 3) = [0.15; 0.55; 0.15; 0.10; 0.05]; %T1 (группа 3)
Group_W_Matrix(:, 2, 3) = [0.20; 0.25; 0.20; 0.20; 0.15]; %T2 (группа 3)
Group_W_Matrix(:, 3, 3) = [0.25; 0.20; 0.15; 0.10; 0.30]; %T3 (группа 3)

active_groups_count = 3;
Group_status = [1, 1, 1];
disp('Агрегация функций принадлежности');
Final_Mus = zeros(N_students, N_terms);
for i = 1:N_terms
    iTerm_W = [];
    for j = 1:3
        if Group_Status(j) == 1
            iTerm_W = [iTerm_W, Group_W_Matrix(:, i, j)];
        end
    end
    
    Aggregated_W = mean(iTerm_W, 2);
    Final_Mus(:, i) = Aggregated_W;
end

T = table(Student_Labels', Final_Mus(:, 1), Final_Mus(:, 2), Final_Mus(:, 3),...
    'VariableNames', {'Student', Term_Labels{1}, Term_Labels{2}, Term_Labels{3}});
disp(T);

High_W = Final_Mus(:, 3);
[max_mu, best_student_i] = max(High_W);
best_student_label = Student_Labels{best_student_i};

fprintf('Наилучший студент для выступления по терму "Высокая готовность":\n');
fprintf('Студент: %s\n', best_student_label);
fprintf('Степень принадлежности(mu): %.3f\n', max_mu);


figure;
bar(Final_Mus, 0.8, 'grouped');
set(gca, 'XTickLabel', Student_Labels);
legend(Term_Labels, 'Location', 'NorthWest');
ylabel('Степень принадлежности (mu)');
xlabel('Студенты');
