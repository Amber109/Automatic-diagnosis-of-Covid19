function COVIDEXTRACT_Fea_v5(netType)
% data_dir = 'E:\\Newcovid\';
% image_size = [224,224,3];
% image_size = size(data);
% files_dir = 'E:\\Newcovid\\COVID19New3\';
image = dir('Users/images/*.*');
files = dir(fullfile('Users/amber/Dropbox/My Mac (MacBook-Pro.local)/Documents/FuzzyPID/Covid_dataset(4)/SARS-CoV-2(2482)/Covid/','*.png'));
lengthFiles = length(files);
COtarget=zeros(1,lengthFiles);
Img_all = zeros(224,224,3,lengthFiles);
for i = 1:lengthFiles
%         Img_resize = [];      
        Img = imread(strcat('Users/amber/Dropbox/My Mac (MacBook-Pro.local)/Documents/FuzzyPID/Covid_dataset(4)/SARS-CoV-2(2482)/Covid/',files(i).name));     
        
        [rows, columns, numberOfColorChannels] = size(Img);
        
        if numberOfColorChannels~=3
            fprintf("Image name is %s, channel is %d\n",files(i).name,numberOfColorChannels)
            Img = cat(3,Img,Img,Img);
        end
        Img_resize = imresize(Img,[224,224]);
        Img_all(:,:,:,i) = Img_resize;
        if contains(files(i).name,'Covid')
            COtarget(:,i)=0;
        elseif contains(files(i).name,'NORMAL')
            COtarget(:,i)=1;
        end
end
M = size(Img_all);


%     indices = crossvalind('Kfold', COtarget, fold_num);
%     cp = classperf(COtarget);
%%  Select nerual networks CNNs             
switch netType
    case 'resnet18'
        net = resnet18;
        layers = net.Layers;
        lgraph = layerGraph(layers);
%         LBdata = augimdsTrain;
        %  the extraction of features from bottom to "pool5"
        layersTransfor = 'pool5';%Resnet18 101 vgg19 vgg16 alexnet
        
        %layersTransfor = 'pool5-drop_7x7_s1'% Googlenet
        %layersTransfor = 'avg_pool';%resnet18 inceptionv3
%         LBFeatures_resnet18 = activations(net, LBdata, layersTransfor);
        
    case 'resnet50'
        net = resnet50;
        layers = net.Layers;
        lgraph = layerGraph(layers);
%         LBdata = augimdsTrain;
        layersTransfor = 'add_16';%Resnet18 101 vgg19 vgg16 alexnet
        
%         LBFeatures_resnet50 = activations(net, LBdata, layersTransfor);
        
    case 'resnet101'
        net = resnet101;
        layers = net.Layers;
        lgraph = layerGraph(layers);
%         LBdata = augimdsTrain;
        layersTransfor = 'pool5';%Resnet18 101 vgg19 vgg16 alexnet
        
%         LBFeatures_resnet101 = activations(net, LBdata, layersTransfor);
    case 'vgg16'
      
        net = vgg16;
        layers = net.Layers;
        lgraph = layerGraph(layers);
%         LBdata = augimdsTrain;
        layersTransfor = 'pool5';%Resnet18 101 vgg19 vgg16 alexnet
        
%         LBFeatures_vgg16 = activations(net, LBdata, layersTransfor);
    case 'vgg19'
        net = vgg19;
        layers = net.Layers;
        lgraph = layerGraph(layers);
%         LBdata = augimdsTrain;
        layersTransfor = 'pool5';%Resnet18 101 vgg19 vgg16 alexnet
        
%         LBFeatures_vgg19 = activations(net, LBdata, layersTransfor);
    case 'densenet201'
        net = densenet201;
%         analyzeNetwork(net); 
         layers = net.Layers;
       lgraph = layerGraph(layers);
%         LBdata = augimdsTrain;
        layersTransfor = 'avg_pool';%Resnet18 101 vgg19 vgg16 alexnet
       
%         LBFeatures_densenet201 = activations(net, LBdata, layersTransfor);
    case 'googlenet'
        net = googlenet;
        layers = net.Layers;
        lgraph = layerGraph(layers);
        COdata=Img_all;
%         LBdata = augimdsTrain;
        layersTransfor = 'pool5-drop_7x7_s1';
        COFeature = activations(net, COdata, layersTransfor);
        
%         LBFeatures_googlenet = activations(net, LBdata, layersTransfor);

    case 'densenet201y'
        net = threednet;
        layers = net.Layers;
        lgraph = layerGraph(layers);
%         LBdata = augimdsTrain;
        layersTransfor = 'pool5-drop_7x7_s1';
 
%         LBFeatures_googlenet = activations(net, LBdata, layersTransfor);
        COFeature = activations(net, COdata, layersTransfor);  
end

%% Augmentation and Activation
% % ---------------------augmentation operation
% imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandXTranslation',[-30,30],'RandYTranslation',[-30,30],'RandRotation',[-20,20]);
% augimdsTrain = augmentedImageDatastore(image_size(1:3), data(:,:,:,train),COtarget(train), 'DataAugmentation', imageAugmenter);
% augimdsValidation = augmentedImageDatastore(image_size(1:3), data(:,:,:,train(1:10:end)),COtarget(train(1:10:end)), 'DataAugmentation', imageAugmenter);
%   
% trainFeature = activations(net, augimdsTrain, layersTransfor);    
% validationFeature = activations(net, augimdsValidation, layersTransfor);  


%% Save Data
% save_train_name = strcat('trainFeature', netType);
% save_validation_name = strcat('validationFeature', netType);
save_data_name = strcat('COFeature', netType);
save_lable_name = strcat('COtarget');
saveaddr='/Users/amber/Dropbox/My Mac (MacBook-Pro.local)/Documents/FuzzyPID/补充实验/';
save([saveaddr, save_data_name], 'COFeature');
save([saveaddr, save_lable_name], 'COtarget');



