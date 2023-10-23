
%%
% Function:
% Yao 2020/8/07
%%

clc;
clear all;
close all;
st=cputime;
% load('Img_all.mat');
% load('COtarget.mat');
% load('indices.mat');
% load('test_lable');
% load('train_lable')

%% load Data



% five fold cross validation
fold_num = 5;
a = zeros(1,fold_num);
sensitivity=zeros(1,fold_num);
specificity=zeros(1,fold_num);
precision=zeros(1,fold_num);
recall=zeros(1,fold_num);
Fscore=zeros(1,fold_num);
% the number of classes
numClasses = 2;
cnnSelect = 'resnet18';
hstAcc = zeros(fold_num,1);
COVIDEXTRACT_Fea_v5(cnnSelect);
load COfeatureresnet18;
load COtarget;

COtarget = COtarget';



%% load the data, and set the group


indices = crossvalind('Kfold',COtarget,fold_num);
cp=classperf(COtarget);

%% K-fold
for k = 1:fold_num
    
    test_lable=(indices ==k);
    find(test_lable==1);
    train_lable=~test_lable;
    % ---------------------------------------------------------------------

    %  transfer learning model
    XTrain = COFeature(:,:,:,train_lable);
    YTrain = categorical(COtarget(train_lable,:));
%     XValidation = validationFeature;
%     YValidation = YTrain(1:10);
    XTest = COFeature(:,:,:,test_lable);
    YTest = categorical(COtarget(test_lable,:));
%     clear trainFeature
%     clear validationFeature
%     clear testFeature
    
    optimVars = [
        optimizableVariable('SectionDepth',[2 3],'Type','integer')
        optimizableVariable('InitialLearnRate',[1e-6 1e-1],'Transform','log')
        optimizableVariable('Momentum',[0.8 0.98])
        optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
    
%     % Perform Bayesian Optimization
    ObjFcn = makeObjFcn(XTrain,YTrain,XTest,YTest);
%     
    BayesObject = bayesopt(ObjFcn,optimVars, ...
        'MaxTime',0, ...
        'IsObjectiveDeterministic',false, ...
        'UseParallel',false);
    
    % Evaluate Final Network
    bestIdx = BayesObject.IndexOfMinimumTrace(end);
    fileName = BayesObject.UserDataTrace{bestIdx};
    savedStruct = load(fileName);
    valError = savedStruct.valError;
    
    % Predict the labels of the test set and calculate the test error. Treat the classification of each image in the test set as independent events with a certain probability of success, which means that the number of incorrectly classified images follows a binomial distribution. Use this to calculate the standard error (testErrorSE) and an approximate 95% confidence interval (testError95CI) of the generalization error rate. This method is often called the Wald method. bayesopt determines the best network using the validation set without exposing the network to the test set. It is then possible that the test error is higher than the validation error.
    [YPredicted,probs] = classify(savedStruct.trainedNet,XTest);

    testError = 1 - mean(YPredicted == YTest);
    accuracy_matrix = sum(YPredicted == YTest)/numel(YPredicted)
    testerrorvalue(k,1)=testError;
    accuracy_matrixvalue(k,1)=accuracy_matrix;
    %  classification;

    hstAcc(k,1)=sum(YPredicted == YTest)/numel(YPredicted)
    NTest = numel(YTest);
    testErrorSE = sqrt(testError*(1-testError)/NTest);
    testError95CI = [testError - 1.96*testErrorSE, testError + 1.96*testErrorSE];
    testErrorSEvalue(k,1)=testErrorSE;

    a(k) = valError;
    
%    ROCresult=plot_roc(YPredicted,YTest);  
%    disp(ROCresult);
    %% Plot result
    % Plot the confusion matrix for the test data. Display the precision and recall for each class by using column and row summaries.
    figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
    cm = confusionchart(YTest,YPredicted);
    cm.Title = 'Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';

    %     %Display some test images together with their predicted classes and the probabilities of those classes.
    %         figure
    %         idx = randperm(numel(YTest),9);
    %         for i = 1:numel(idx)
    %             subplot(3,3,i)
    %             imshow(XTest(:,:,:,idx(i)));
    %             prob = num2str(100*max(probs(idx(i),:)),3);
    %             predClass = char(YPredicted(idx(i)));
    %             label = [predClass,', ',prob,'%'];
    %             title(label)
    %         end
    
    % Calculate sensitivity; specificity; recall; precision
    tp = 0;
    tn = 0;
    fn = 0;
    fp = 0;

   
    for y = 1:length(YPredicted)
        if (YPredicted(y,1)=='0') && (YTest(y,1)=='0')
            tp=tp+1;
        elseif (YPredicted(y,1)=='1') && (YTest(y,1)=='1')
            tn=tn+1;
        elseif (YPredicted(y,1)=='0') && (YTest(y,1)=='1')
            fp=fp+1;
        elseif (YPredicted(y,1)=='1') && (YTest(y,1)=='0')
            fn=fn+1;

        end
    end
    sens = tp/(tp+fn);
    spec = tn/(fp+tn);
    pre = tp/(tp+fp);
    rec = sens;
    F1s = 2*(pre*rec)/(pre+rec);


    
    sensitivity(k)=sens;
    specificity(k)=spec;
    precision(k)=pre;
    recall(k)=rec;
    Fscore(k)=F1s;


    filename1=strcat('YTest',num2str(k));
    save(filename1,'YTest');
    filename2=strcat('YPredicted',num2str(k));
    save(filename2,'YPredicted');
    filename3=strcat('precision');
    save(filename3,'precision');
    filename4=strcat('hstAcc');
    save(filename4,'hstAcc');
    
    filename6=strcat('recall');
    save(filename6,'recall');

    
    filename9=strcat('Fscore');
    save(filename9,'Fscore');


end
%cp.ErrorRate
%   ind = randperm(size(Train_data.Features, 1));
%  train_X = Train_data.Features(:,ind);
% Train_data.Labels =Train_data.Labels(ind);
% figure ();
% TrainFeatures=squeeze(XTrain);
% TrainFeatures=PERMUTE(TrainFeatures,order);
% Set parameters
%    no_dims = 2;
%    initial_dims = 50;
%    perplexity = 30;
% Run t?SNE
% mappedX = tsne(TrainFeatures);
% % Plot results
% gscatter(mappedX(:,1), mappedX(:,2), YTrain);



mean(hstAcc)
max(hstAcc)
min(hstAcc)

fa=figure(1);
saveas(fa,'B1.png');
fb=figure(2);
saveas(fb,'C1.png');
fc=figure(3);
saveas(fc,'B2.png');
fd=figure(4);
saveas(fd,'C2.png');
fe=figure(5);
saveas(fe,'B3.png');
ff=figure(6);
saveas(ff,'C3.png');
fg=figure(7);
saveas(fg,'B4.png');
fh=figure(8);
saveas(fh,'C4.png');
fi=figure(9);
saveas(fi,'B5.png');
fj=figure(10);
saveas(fj,'C5.png');

et=cputime-st;

%% Objective Function for Optimization
function ObjFcn = makeObjFcn(XTrain,YTrain,XTest,YTest)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)

        %Define the convolutional neural network architecture.
        input_imageSize = size(XTrain);
        numClasses = numel(unique(YTrain));
        numF = round(16/sqrt(optVars.SectionDepth));
        
        %% construct the structure of transferred and retrained layers
        layers = [
            imageInputLayer(input_imageSize(1:3),'Name','data','Normalization','zerocenter')
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            convBlock(3,2*numF,optVars.SectionDepth)
% %             maxPooling2dLayer(3,'Stride',2,'Padding','same') 
% %             convBlock(3,4*numF,optVars.SectionDepth)
%             averagePooling2dLayer(2)
%           convolution2dLayer([1 1],15,"Name","conv_1","BiasLearnRateFactor",1,"WeightLearnRateFactor",1)
%             convolution2dLayer([1 1],3,"Name","conv_2","BiasLearnRateFactor",1,"WeightLearnRateFactor",1)
            fullyConnectedLayer(15,'Name','fc100','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
            fullyConnectedLayer(numClasses,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
            softmaxLayer('Name','fc2_softmax')
            classificationLayer('Name','ClassificationLayer_fc2')];
%         analyzeNetwork(layers);
%         Temlayers=        [
%             fullyConnectedLayer(15,'Name','fc100','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%             fullyConnectedLayer(numClasses,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%             softmaxLayer('Name','fc2_softmax')
%             classificationLayer('Name','ClassificationLayer_fc2')];
%         newlgraph = addLayers(lgraph,Temlayers)
        %% Set hyperparameters
        MiniBatchSize = 30;
        validationFrequency = floor(numel(YTrain)/MiniBatchSize);
        options = trainingOptions(   'adam', ...
            'InitialLearnRate',3e-4, ...
            'SquaredGradientDecayFactor',0.99, ...
            'MaxEpochs',50, ...
            'LearnRateDropFactor',0.2, ... 
            'LearnRateSchedule','piecewise', ...
            'MiniBatchSize',10, ...
            'Shuffle','every-epoch', ...
            'Verbose',false, ...
            'Plots','training-progress', ...
            'ValidationData',{XTrain(:,:,:,1:10),YTrain(1:10)}, ...
            'ValidationFrequency',validationFrequency, ...
            'LearnRateDropPeriod',20);
% %% Set hyperparameters
%         MiniBatchSize = 30;
%         validationFrequency = floor(numel(YTrain)/MiniBatchSize);
%         options = trainingOptions('sgdm', ...
%             'InitialLearnRate',optVars.InitialLearnRate, ...
%             'MaxEpochs',30, ...//12
%             'LearnRateSchedule','piecewise', ...
%             'MiniBatchSize',MiniBatchSize, ...
%             'Shuffle','every-epoch', ...
%             'Verbose',false, ...
%             'Plots','training-progress', ...
%             'ValidationData',{XTrain(:,:,:,1:10),YTrain(1:10)}, ...
%             'ValidationFrequency',validationFrequency, ...
%             'InitialLearnRate',0.001,...
%             'SequencePaddingValue', 0, ...
%             'LearnRateDropPeriod',30);
        %dropperiod50 batchsize turn up
        % Train the network and plot the training progress during training. Close all training plots after training finishes.
        trainedNet = trainNetwork(XTrain, YTrain, layers, options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'));
      %            'LearnRateDropFactor',0.000001, ...  
        % Evaluate the trained network on the val1idation set, calculate the predicted image labels, and calculate the error rate on the validation data.
        YPredicted = classify(trainedNet,XTest);
        valError = 1 - mean(YPredicted == YTest);
%         sum(YPredicted == YTest)/numel(YPredicted);
%         
        %         YPredicted.Test = classify(trainedNet,XTest);
        %         valError.Test = 1 - mean(YPredicted.Test == YTest);
        %         sum(YPredicted.Test == YTest)/numel(YPredicted.Test);
        %         YPredicted.Train = classify(trainedNet,XTrain);
        %         valError.Train = 1 - mean(YPredicted.Train == YTrain);
        %         sum(YPredicted.Train == YTrain)/numel(YPredicted.Train);
        
        % Create a file name containing the validation error, and save the network, validation error, and training options to disk. The objective function returns fileName as an output argument, and bayesopt returns all the file names in BayesObject.UserDataTrace. The additional required output argument cons specifies constraints among the variables. There are no variable constraints.
        fileName = num2str(valError) + ".mat";
        save(fileName,'trainedNet','valError','options');
        cons = [];
        
    end
end


%% The convBlock function creates a block of numConvLayers convolutional layers, each with a specified filterSize and numFilters filters, and each followed by a batch normalization layer and a ReLU layer.
function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Padding','same')
    batchNormalizationLayer
    reluLayer];
layers = repmat(layers,numConvLayers,1);
end

