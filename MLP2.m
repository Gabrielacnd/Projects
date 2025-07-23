close all;
num_classes = input('Introduceți numărul de clase: ');
all_data = [];
all_labels = [];
label_names = {};

for class_label = 1:num_classes
    file_path = input(sprintf('Introduceți calea fișierului CSV pentru clasa %d: ', class_label), 's');
    T = readtable(file_path);

    [~, name_only, ~] = fileparts(file_path);
    label_names{class_label} = upper(name_only);

    cycles = unique(T.Cycle);

    for c = 1:length(cycles)
        cycle_data = T(T.Cycle == cycles(c), :);

        % Verificare: daca avem 8 senzori
        if height(cycle_data) ~= 8
            continue;
        end

        % 1. Medii
        avg_temp = mean(cycle_data.AvgTemp);
        avg_humidity = mean(cycle_data.AvgHumidity);
        avg_pressure = mean(cycle_data.AvgPressure);

        % 2. Rezistențele de la toți senzorii
        gas_res = [];
        for s = 1:8
            for i = 0:9
                val = cycle_data.(sprintf('GasRes_%d', i))(s);
                gas_res = [gas_res, val];
            end
        end

        % Vector complet 1x83
        full_feature = [avg_temp, avg_humidity, avg_pressure, gas_res];

        all_data = [all_data; full_feature];
        all_labels = [all_labels; class_label];
    end
end

% === Antrenare MLP ===
X = all_data;
Y = all_labels;

cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));

mu = mean(X_train);
sigma = std(X_train);
sigma(sigma == 0) = 1;

X_train_norm = (X_train - mu) ./ sigma;
X_test_norm = (X_test - mu) ./ sigma;

net = patternnet([16 8]);
net.trainParam.epochs = 80;
net.trainParam.lr = 0.001;
net.trainParam.showWindow = true;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

[net, tr] = train(net, X_train_norm', dummyvar(Y_train)');

%Evaluare
Y_pred = net(X_test_norm');
[~, predicted_labels] = max(Y_pred, [], 1);
acc = sum(predicted_labels' == Y_test) / numel(Y_test);
fprintf('Acuratețe MLP: %.2f%%\n', acc * 100);

figure;
confusionchart(Y_test, predicted_labels);
title('Confusion Matrix');

% === Export parametri într-un singur fișier .h pentru ESP32 ===
w1 = net.IW{1};         
w2 = net.LW{2,1};      
w3 = net.LW{3,2};      
b1 = net.b{1};
b2 = net.b{2};
b3 = net.b{3};

fid = fopen('mlp_model_esp32.h', 'w');
fprintf(fid, '#ifndef MLP_MODEL_ESP32_H\n#define MLP_MODEL_ESP32_H\n\n');
fprintf(fid, '// MLP model exportat pentru ESP32\n');
fprintf(fid, '// Acuratețe: %.2f%%\n\n', acc * 100);

% Dimensiuni
fprintf(fid, '#define N_IN %d\n', size(w1, 2));
fprintf(fid, '#define H1 %d\n', size(w1, 1));
fprintf(fid, '#define H2 %d\n', size(w2, 1));
fprintf(fid, '#define N_OUT %d\n\n', size(w3, 1));

% Parametri
write_array(fid, 'w1', w1);
write_array(fid, 'b1', b1);
write_array(fid, 'w2', w2);
write_array(fid, 'b2', b2);
write_array(fid, 'w3', w3);
write_array(fid, 'b3', b3);
write_array(fid, 'mu', mu);
write_array(fid, 'sigma', sigma);

% Funcție de inferență
fprintf(fid, '\n#include <math.h>\n');
fprintf(fid, 'inline void mlp_predict_vector(float xn[N_IN], float probs[N_OUT], int &pred) {\n');
fprintf(fid, '  float a1[H1];\n');
fprintf(fid, '  for (int i = 0; i < H1; ++i) {\n');
fprintf(fid, '    float z = b1[i];\n');
fprintf(fid, '    for (int j = 0; j < N_IN; ++j) z += w1[i][j] * xn[j];\n');
fprintf(fid, '    a1[i] = tanhf(z);\n');
fprintf(fid, '  }\n\n');

fprintf(fid, '  float a2[H2];\n');
fprintf(fid, '  for (int i = 0; i < H2; ++i) {\n');
fprintf(fid, '    float z = b2[i];\n');
fprintf(fid, '    for (int j = 0; j < H1; ++j) z += w2[i][j] * a1[j];\n');
fprintf(fid, '    a2[i] = tanhf(z);\n');
fprintf(fid, '  }\n\n');

fprintf(fid, '  float z3[N_OUT];\n');
fprintf(fid, '  for (int i = 0; i < N_OUT; ++i) {\n');
fprintf(fid, '    float z = b3[i];\n');
fprintf(fid, '    for (int j = 0; j < H2; ++j) z += w3[i][j] * a2[j];\n');
fprintf(fid, '    z3[i] = z;\n');
fprintf(fid, '  }\n\n');

fprintf(fid, '  float z_max = z3[0];\n');
fprintf(fid, '  for (int i = 1; i < N_OUT; ++i) if (z3[i] > z_max) z_max = z3[i];\n');
fprintf(fid, '  float sum = 0.f;\n');
fprintf(fid, '  for (int i = 0; i < N_OUT; ++i) {\n');
fprintf(fid, '    probs[i] = expf(z3[i] - z_max);\n');
fprintf(fid, '    sum += probs[i];\n');
fprintf(fid, '  }\n');
fprintf(fid, '  for (int i = 0; i < N_OUT; ++i) probs[i] /= sum;\n');
fprintf(fid, '  pred = 0;\n');
fprintf(fid, '  for (int i = 1; i < N_OUT; ++i) if (probs[i] > probs[pred]) pred = i;\n');
fprintf(fid, '}\n\n');

fprintf(fid, '#endif // MLP_MODEL_ESP32_H\n');
fclose(fid);
% fprintf("Parametrii salvați în 'mlp_model_esp32.h'\n");

% === Funcție pentru scriere matrici/vectors în fișierul .h ===
function write_array(fid, name, M)
    if isvector(M)
        M = M(:);
        fprintf(fid, 'float %s[%d] = {', name, numel(M));
        fprintf(fid, '%ff, ', M(1:end-1));
        fprintf(fid, '%ff};\n\n', M(end));
    else
        [rows, cols] = size(M);
        fprintf(fid, 'float %s[%d][%d] = {\n', name, rows, cols);
        for i = 1:rows
            fprintf(fid, '  {');
            fprintf(fid, '%ff, ', M(i,1:end-1));
            fprintf(fid, '%ff}', M(i,end));
            if i < rows
                fprintf(fid, ',\n');
            else
                fprintf(fid, '\n');
            end
        end
        fprintf(fid, '};\n\n');
    end
end
