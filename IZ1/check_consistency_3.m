function [CI, CR, isConsistent] = check_consistency_3(lambda_max)
    RI = 0.58;
    CI = (lambda_max - 3) / (3 - 1);
    CR = CI / RI;
    isConsistent = CR <= 0.10;
end
