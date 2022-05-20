import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from typing import Union

class CharacterLevelCNN(nn.Module):
    def __init__(self, args): 
        super(CharacterLevelCNN, self).__init__()
        with open(args.config_path) as f:
            self.config = json.load(f) 
        self.usembedding = args.usembedding    
        self.embedlength = args.embedlength
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv_layers = [] 
        if self.usembedding:
            in_channels = args.number_of_characters + len(args.extra_characters)
            embedding = nn.Embedding(in_channels,self.embedlength)
            embedding = embedding.to(self.device)
            conv_layers.append(embedding)        
        for i, conv_layer_parameter in enumerate(self.config['model_parameters'][args.size]['conv']):
            if i == 0:
                in_channels = args.number_of_characters + len(args.extra_characters)
                out_channels = conv_layer_parameter[0]
            else:
                in_channels, out_channels = conv_layer_parameter[0], conv_layer_parameter[0]

            if conv_layer_parameter[2] != -1:
                conv_layer = nn.Sequential(nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=conv_layer_parameter[1], padding=0).to(self.device),
                                           nn.ReLU(),
                                           nn.MaxPool1d(conv_layer_parameter[2]).to(self.device))                    
            else:
                conv_layer = nn.Sequential(nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=conv_layer_parameter[1], padding=0).to(self.device),
                                           nn.ReLU())
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers) 
        input_shape = (args.batch_size, args.max_length,
                       args.number_of_characters + len(args.extra_characters))
        _, dimension = self._get_conv_output(input_shape)
        print('dimension :', dimension)
        fc_layer_parameter = self.config['model_parameters'][args.size]['fc'][0]
        fc_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dimension, fc_layer_parameter), 
                          nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(fc_layer_parameter,fc_layer_parameter), 
                          nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(fc_layer_parameter, args.number_of_classes))  
        ])
        self.fc_layers = fc_layers
        if args.size == 'small':
            self._create_weights(mean=0.0, std=0.05)
        elif args.size == 'large':
            self._create_weights(mean=0.0, std=0.02) 

    def _create_weights(self, mean: float = 0.0, std: float = 0.05):
        for module in self.modules():
            #print('weights initialization:-->')
            print(module)
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape: tuple) -> Union[torch.Tensor, torch.Size]:
        """ 
        Test function for processing the random toy-Tensor which perform its
        successive passes via the predefined CNN without any 
        supplementary backpropagation.
        This funciton is only dedicated for checking the resulting shape of
        feeded tensor while pulling it over prebuild CNN layers
        """        
        input = torch.rand(shape)
        input = input.to(self.device)
        if self.usembedding:
            for batch in range(input.shape[0]):
                for i in range(input.shape[1]):
                    if self.device.type == 'cpu':
                        res = torch.from_numpy(self.conv_layers[0](torch.LongTensor([np.argmax(input[batch][i])])).detach().numpy().squeeze(0))
                        input[batch][i] = F.pad(res, pad=(0, len(input[batch][i])-len(res)), mode='constant', value=0)
                    else:            
                        indx = torch.LongTensor([torch.argmax(input[batch][i])])
                        indx = indx.to(self.device)
                        res = torch.from_numpy(self.conv_layers[0](indx).cpu().detach().numpy().squeeze(0))
                        input[batch][i] = F.pad(res, pad=(0, len(input[batch][i])-len(res)), mode='constant', value=0).to(self.device)        
            output = input.transpose(1, 2)
            # forward pass through conv layers
            for i in range(1,len(self.conv_layers)):
                output = self.conv_layers[i](output)
                print('conv({0}) shape:{1}'.format(i,np.shape(output)))       
        else:
            output = input.transpose(1, 2)
            # forward pass through conv layers
            for i in range(len(self.conv_layers)):
                output = self.conv_layers[i](output)
                print('conv({0}) shape:{1}'.format(i,np.shape(output)))       
        output = output.view(output.size(0), -1)
        print('output reshape: {}'.format(np.shape(output)))       
        n_size = output.size(1)           
        
        return output, n_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ 
        Main function for processing the real input Tensor which perform its
        successive passes via the predefined CNN/Dence layers without any 
        supplementary backpropagation 
        """
        if self.usembedding:
            for batch in range(input.shape[0]):
                for i in range(input.shape[1]):                   
                    if self.device.type == 'cpu':
                        res = torch.from_numpy(self.conv_layers[0](torch.LongTensor([np.argmax(input[batch][i])])).detach().numpy().squeeze(0))
                        input[batch][i] = F.pad(res, pad=(0, len(input[batch][i])-len(res)), mode='constant', value=0)
                    else: 
                        indx = torch.LongTensor([torch.argmax(input[batch][i])])
                        indx = indx.to(self.device)
                        res = torch.from_numpy(self.conv_layers[0](indx).cpu().detach().numpy().squeeze(0))
                        input[batch][i] = F.pad(res, pad=(0, len(input[batch][i])-len(res)), mode='constant', value=0).to(self.device)
            output = input.transpose(1, 2)
            # forward pass through conv layers
            for i in range(1,len(self.conv_layers)):
                output = self.conv_layers[i](output)
                #print('conv({0}) shape:{1}'.format(i,np.shape(output))) 
        else:
            output = input.transpose(1, 2)
            # forward pass through conv layers
            for i in range(len(self.conv_layers)):
                output = self.conv_layers[i](output)            
        output = output.view(output.size(0), -1)        
        # forward pass through fc layers
        for i in range(len(self.fc_layers)):
            output = self.fc_layers[i](output)
            
        return output