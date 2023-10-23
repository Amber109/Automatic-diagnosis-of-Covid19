clc;
clear all;
close all;
load YPredicted5;

Image = imread(strcat('E:\Newcovid\COVID19New7\COVID19(33).jpg'));  
% if numberOfColorChannels~=3
%         Image = cat(3,Image,Image,Image);
% end
Image_resize = imresize(Image,[224,224]);

numClasses = numel(unique(YTrain));
numF = round(16/sqrt(2));
        layers = [
            imageInputLayer(input_imageSize(1:3),'Name','data','Normalization','zerocenter')
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            convBlock(3,2*numF,2)
% %             maxPooling2dLayer(3,'Stride',2,'Padding','same') 
% %             convBlock(3,4*numF,optVars.SectionDepth)
%             averagePooling2dLayer(2)
%           convolution2dLayer([1 1],15,"Name","conv_1","BiasLearnRateFactor",1,"WeightLearnRateFactor",1)
%             convolution2dLayer([1 1],3,"Name","conv_2","BiasLearnRateFactor",1,"WeightLearnRateFactor",1)
            fullyConnectedLayer(15,'Name','fc100','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
            fullyConnectedLayer(numClasses,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
            softmaxLayer('Name','fc2_softmax')
            classificationLayer('Name','ClassificationLayer_fc2')];
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
trainedNet = trainNetwork(XTrain, YTrain, layers, options);
[classfn,score]=classify(trainedNet,Image);
show(Image);
title(sprintf("%s(%.2f)",classfn,score(classfn)));
map=gradCAM(trainedNet,Image,classfn);
imshow(Image);
hold on;
imagesc(map,'AlphaData',0.5);
color jet
hold off;
title(Grad-CAM);
function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Padding','same')
    batchNormalizationLayer
    reluLayer];
layers = repmat(layers,numConvLayers,1);
end
