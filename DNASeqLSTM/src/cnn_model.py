import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from typing import Union
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class BlockLSTM(nn.Module):
    def __init__(self, features=1, num_layers=2, lstm_hs=128, dropout=0.4, attention=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.Sequential(OrderedDict(
                                              [("LSTM", nn.LSTM(input_size=features, 
                                                                hidden_size=lstm_hs, 
                                                                num_layers=num_layers,
                                                                batch_first= True, 
                                                                bias=True, 
                                                                bidirectional= True).to(self.device))]
                                             )
                                 )
        self.dropout = nn.Sequential(OrderedDict([("DropoutAfterLSTM", nn.Dropout(p=dropout).to(self.device))]))
    def forward(self, x):
        # input is of the form (batch_size, time_steps, num_variables), e.g. (128, 60, 5)
#         x = torch.transpose(x, 1, 2)
        # lstm layer is of the form (batch_size, num_variables, time_steps)
        x, _ = self.lstm(x)
        # dropout layer input shape:
        y = self.dropout(x)
        # output shape is of the form ()
        return y

class BlockFCL (nn.Module):
    def __init__(self, total_dim, number_of_classes, config, fcsize):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_dim = total_dim
        self.config = config, 
        self.fcsize = fcsize
        self.number_of_classes  = number_of_classes 
        
        fc_layers=[]
        for i, fc_layer_parameter in enumerate(self.config[0]['model_parameters'][self.fcsize]['fc']):
            if i == 0:
                fc_layer = nn.Sequential(OrderedDict([("FC_{}".format(i),
                                                        nn.Sequential(nn.Linear(self.total_dim, fc_layer_parameter),
                                                                      nn.Dropout(0.5), 
                                                                      nn.ReLU()).to(self.device)
                                                      )
                                                     ])
                                        )
            else:
                fc_layer = nn.Sequential(OrderedDict([("FC_{}".format(i),
                                                        nn.Sequential(nn.Linear(prev_fc_layer_parameter, fc_layer_parameter),
                                                                      nn.Dropout(0.5), 
                                                                      nn.ReLU()).to(self.device)
                                                      )
                                                     ])
                                        )                
             
            fc_layers.append(fc_layer)
            prev_fc_layer_parameter = fc_layer_parameter
            
        fin = \
        nn.Sequential(OrderedDict([("FC_out",
                                    nn.Sequential(nn.Linear(fc_layer_parameter, 
                                                            self.number_of_classes)).to(self.device))
                                  ])
                     ) 
        fc_layers.append(fin)    
        self.fc_layers = nn.ModuleList(fc_layers)

#         fc_layer_parameter = self.config[0]['model_parameters'][self.fcsize]['fc'][0]
#         fc_layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(self.total_dim, fc_layer_parameter),
#                           nn.Dropout(0.5), nn.ReLU()),
#             nn.Sequential(nn.Linear(fc_layer_parameter,fc_layer_parameter),
#                           nn.Dropout(0.5), nn.ReLU()),
#             nn.Sequential(nn.Linear(fc_layer_parameter, self.number_of_classes))
#         ])         
#         self.fc_layers = nn.ModuleList(fc_layers)
        
    def forward(self, x):
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)        
        return x 
    

class CharacterLevelLSTMCNN(nn.Module):
    def __init__(self, args):
        super(CharacterLevelLSTMCNN, self).__init__()
        
        with open(args.config_path) as f:
            self.config = json.load(f)

        if args.extra_characters != []:
            self.vocabulary = list(args.alphabet) + list(args.extra_characters)
            self.number_of_characters = args.number_of_characters + \
                len(args.extra_characters)
        else:
            self.vocabulary = list(args.alphabet)
            self.number_of_characters = args.number_of_characters            
            
        self.useSeparatedConv1D = args.useSeparatedConv1D 
        self.useLSTM = args.useLSTM
        self.LSTMAvgPoolKernelSize = args.setLSTMAvgPoolKernelSize
        self.applyConv1DForLSTM = args.applyConv1DForLSTM
        self.changeConv1DDirLSTM = args.changeConv1DDirLSTM
        self.useBOCNGrams = args.useBOCNGrams
        self.useSentencePieceTokenizer = args.useSentencePieceTokenizer
        self.useNGramBPETokenizer      = args.useNGramBPETokenizer        
        self.usembedding = args.usembedding
        self.embedlength = args.embedlength
        self.embedAfterBatches = args.embedAfterBatches
        self.configConvSize = args.size
        self.number_of_classes = args.number_of_classes                            
        self.useBatchNormalization = args.useBatchNormalization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv_layers = []
        
        if self.usembedding:
            if self.useBOCNGrams:
                in_channels = args.trigram_map_len
            elif self.useSentencePieceTokenizer:
                in_channels = args.vocabSize 
            elif self.useNGramBPETokenizer:
                in_channels = args.vocabSize                
            else:
                in_channels = args.number_of_characters + len(args.extra_characters)
            
            embedding = nn.Sequential(OrderedDict([("Embedding",
                                                    nn.Embedding(in_channels,self.embedlength))]))
            embedding = embedding.to(self.device) 
                              
            # Generate Embed vectors during Forward pass    
            if self.embedAfterBatches:    
                conv_layers.append(embedding)
            # Save Embedding Vectors for Cross-Class usage
            self._embedding = embedding
        for i, conv_layer_parameter in enumerate(self.config['model_parameters'][args.size]['conv']):
            if i == 0:
                if self.useBOCNGrams:
                    if self.usembedding and not self.embedAfterBatches:
                        in_channels =  self.embedlength
                    else:
                        in_channels =  args.trigram_map_len
                    out_channels = conv_layer_parameter[0]
                elif self.useSentencePieceTokenizer or self.useNGramBPETokenizer:
                    if self.usembedding and not self.embedAfterBatches:
                        in_channels =  self.embedlength
                    else:
                        in_channels =  args.vocabSize
                    out_channels = conv_layer_parameter[0]     
                else:
                    if self.usembedding and not self.embedAfterBatches:
                        in_channels =  self.embedlength
                    else:
                        in_channels = args.number_of_characters + len(args.extra_characters)
                        out_channels = conv_layer_parameter[0]
            else: 
                in_channels, out_channels = conv_layer_parameter[0], conv_layer_parameter[0]

            if conv_layer_parameter[2] != -1:
                if self.useBatchNormalization:
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("Conv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("Conv1D_{}_ReLU".format(i),nn.ReLU()),
                                               ("Conv1D_{}_MaxPool1d".format(i),
                                                nn.MaxPool1d(conv_layer_parameter[2]).to(self.device)
                                               ),
                                               ("Conv1D_{}_BatchNorm1d".format(i),
                                                nn.BatchNorm1d(out_channels, momentum=0.7).to(self.device)
                                               )
                                                           ]))                     
                else:
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("Conv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("Conv1D_{}_ReLU".format(i),nn.ReLU()),
                                               ("Conv1D_{}_MaxPool1d".format(i),
                                                nn.MaxPool1d(conv_layer_parameter[2]).to(self.device)
                                               )
                                                           ]))                    
            else:
                if self.useBatchNormalization:
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("Conv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("Conv1D_{}_ReLU".format(i),nn.ReLU()),
                                               ("Conv1D_{}_BatchNorm1d".format(i),
                                                nn.BatchNorm1d(out_channels, momentum=0.7).to(self.device)
                                               )
                                                           ]))                     
                else:    
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("Conv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("Conv1D_{}_ReLU".format(i),nn.ReLU())
                                                           ])) 
            conv_layers.append(conv_layer)
            
        ### Add a layer with Dropout after the final Conv1D layer:
        conv_layers.append(nn.Sequential(nn.Dropout(0.2).to(self.device)))
        self.conv_layers = nn.ModuleList(conv_layers)
        
        if self.useBOCNGrams:
            if self.usembedding and not self.embedAfterBatches:
                input_shape = (args.batch_size, args.max_length,
                               args.embedlength)             
            else:
                input_shape = (args.batch_size, args.max_length,
                               args.trigram_map_len)
        elif self.useSentencePieceTokenizer or self.useNGramBPETokenizer:
            if self.usembedding and not self.embedAfterBatches:
                input_shape = (args.batch_size, args.max_length,
                               args.embedlength)             
            else:            
                input_shape = (args.batch_size, args.max_length,
                               args.vocabSize)            
        else:
            if self.usembedding and not self.embedAfterBatches:
                input_shape = (args.batch_size, args.max_length,
                               args.embedlength)             
            else:            
                input_shape = (args.batch_size, args.max_length,
                               args.number_of_characters + len(args.extra_characters)
                              )
        input = torch.rand(input_shape)
        input = input.to(self.device)
        print('START: input shape: {}'.format(np.shape(input)))
        conv_out, conv_dim = self._get_conv_output(input, verbosity=1)
        print('conv all dimension :', conv_dim)
        if self.useLSTM:
            lstm_out, lstm_dim = self._get_lstm_output(input, verbosity=1)
            print('lstm_conv dimension :', lstm_dim)
            self.total_dim = conv_dim+lstm_dim
    #         self.fc_layers = fc_layers
            out = torch.cat((conv_out,lstm_out),1)
            print('total shape before FC: {}'.format(np.shape(out)))
        else:
            out = conv_out
            print('total shape before FC: {}'.format(np.shape(out)))
            self.total_dim = conv_dim

        _, fc_dim = self._get_fc_output(out, verbosity=1)
        print('final dimension :', fc_dim)
        
        del input, out, fc_dim 
        
        if args.size == 'small':
            self._create_weights(mean=0.0, std=0.05)
        elif args.size == 'large':
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean: float = 0.0, std: float = 0.05):
        for module in self.modules():
            #print('weights initialization:-->')
#             print(module)
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, input: torch.Tensor, verbosity = 0) -> Union[torch.Tensor, torch.Size]:
        """
        Test function for processing the random toy-Tensor which perform its
        successive passes via the predefined CNN without any
        supplementary backpropagation.
        This function is only dedicated for checking the resulting shape of
        feeded tensor while pulling it over through prebuilded CNN layers
        """

        ################################################################
        # FIRST signal pass (Convolution over sequenses only)
        ################################################################
        if self.usembedding and self.embedAfterBatches:
            for batch in range(input.shape[0]):
                for i in range(input.shape[1]):
                    if self.device.type == 'cpu':
                        res = torch.from_numpy(self.conv_layers[0](torch.LongTensor([np.argmax(input[batch][i])])).detach().numpy().squeeze(0))
                        input[batch][i] = F.pad(res, pad=(0, len(input[batch][i])-len(res)), mode='constant', value=0)
                    else:
                        indx = torch.LongTensor([torch.argmax(input[batch][i])]).to(self.device)
                        res = torch.from_numpy(self.conv_layers[0](indx).cpu().detach().numpy().squeeze(0))
                        input[batch][i] = F.pad(res, pad=(0, len(input[batch][i])-len(res)), mode='constant', value=0).to(self.device)
            output = input.transpose(1, 2)
            # forward pass through conv layers
            for i in range(1,len(self.conv_layers)):
                output = self.conv_layers[i](output)
                if verbosity:
                    print('conv({0}) shape:{1}'.format(i,np.shape(output)))
        else:
            output = input.transpose(1, 2)
            if verbosity:
                print('conv input shape: {}'.format(np.shape(output)))
            # forward pass through conv layers
            for i in range(len(self.conv_layers)):
                output = self.conv_layers[i](output)
                if verbosity:
                    print('conv({0}) shape:{1}'.format(i,np.shape(output)))
        output = output.view(output.size(0), -1)
        if verbosity:
            print('conv output reshape: {}'.format(np.shape(output)))
        
        ################################################################
        # SECOND signal pass (Alternating convolution: 
        # first pass - over sequences, second  one over filters)
        ################################################################
        if self.useSeparatedConv1D:
            # First Conv1D over sequenses channels 
            output2 = input.transpose(1, 2)
            if verbosity:
                print('SeparatedConv1D: input shape for 1 iter:{0}'.format(np.shape(output2)))
            output2 = self.conv_layers[0](output2)
            # Continue to apply a Convolution1D operation over filters rather than over sequences
            # Shuffle dims again
            output2 = output2.transpose(1, 2)
            if verbosity:
                print('SeparatedConv1D: input shape for next iters:{0}'.format(np.shape(output2)))
            for i, conv_layer_parameter in enumerate(self.config['model_parameters'][self.configConvSize]['conv_separated']): 
                if i == 0:
                    in_channels = output2.size(1)
                    out_channels = conv_layer_parameter[0]
                else:
                    in_channels, out_channels = conv_layer_parameter[0], conv_layer_parameter[0]
                if conv_layer_parameter[2] != -1:
                    if self.useBatchNormalization:
                        conv_layer = nn.Sequential(OrderedDict([
                                                   ("SeparatedConv1D_{}".format(i),
                                                    nn.Conv1d(in_channels,
                                                              out_channels,
                                                              kernel_size=conv_layer_parameter[1], 
                                                              stride=1, padding=0).to(self.device)
                                                   ),
                                                   ("SeparatedConv1D_ReLU_{}".format(i),nn.ReLU()),
                                                   ("SeparatedConv1D_AvgPool1d_{}".format(i),
                                                    nn.AvgPool1d(conv_layer_parameter[2]).to(self.device)
                                                   ),
                                                   ("SeparatedConv1D_BatchNorm1d_{}".format(i),
                                                    nn.BatchNorm1d(out_channels, momentum=0.7).to(self.device)
                                                   )
                                                               ]))                                           
                    else:                        
                        conv_layer = nn.Sequential(OrderedDict([
                                                   ("SeparatedConv1D_{}".format(i),
                                                    nn.Conv1d(in_channels,
                                                              out_channels,
                                                              kernel_size=conv_layer_parameter[1], 
                                                              stride=1, padding=0).to(self.device)
                                                   ),
                                                   ("SeparatedConv1D_ReLU_{}".format(i),nn.ReLU()),
                                                   ("SeparatedConv1D_AvgPool1d_{}".format(i),
                                                    nn.AvgPool1d(conv_layer_parameter[2]).to(self.device)
                                                   )
                                                               ]))
                else:
                    if self.useBatchNormalization:
                        conv_layer = nn.Sequential(OrderedDict([
                                                   ("SeparatedConv1D_{}".format(i),
                                                    nn.Conv1d(in_channels,
                                                              out_channels,
                                                              kernel_size=conv_layer_parameter[1], 
                                                              stride=1, padding=0).to(self.device)
                                                   ),
                                                   ("SeparatedConv1D_ReLU_{}".format(i),nn.ReLU()),
                                                   ("SeparatedConv1D_BatchNorm1d_{}".format(i),
                                                    nn.BatchNorm1d(out_channels, momentum=0.7).to(self.device)
                                                   )
                                                               ]))                  
                    else:
                        conv_layer = nn.Sequential(OrderedDict([
                                                   ("SeparatedConv1D_{}".format(i),
                                                    nn.Conv1d(in_channels,
                                                              out_channels,
                                                              kernel_size=conv_layer_parameter[1], 
                                                              stride=1, padding=0).to(self.device)
                                                   ),
                                                   ("SeparatedConv1D_ReLU_{}".format(i),nn.ReLU())
                                                               ])) 
                output2 = conv_layer(output2)
                if verbosity:
                    print('SeparatedConv1D({0}) shape:{1}'.format(i,np.shape(output2)))
                
            output2 = output2.view(output.size(0), -1)
            if verbosity:
                print('SeparatedConv1D output reshape: {}'.format(np.shape(output2)))
      
        if self.useSeparatedConv1D:              
            out = torch.cat((output,output2),1)
        else:
            out = output
        n_size = out.size(1)

        return out, n_size

    def _get_conv_output_lstm(self, input: torch.Tensor, verbosity = 0) -> Union[torch.Tensor, torch.Size]:
        if self.changeConv1DDirLSTM:
            # Shuffle dims to apply a Conv1D operation over sequence channel
            output = input.transpose(1, 2)
            if verbosity:
                print('shuffle dims lstm shape:{0}'.format(np.shape(output)))
        else:
            # Normal mode: convolve over hidden space channel
            output = input
        for i, conv_layer_parameter in enumerate(self.config['model_parameters'][self.configConvSize]['conv_lstm']): 
            if i == 0:
                in_channels = output.size(1)
                out_channels = conv_layer_parameter[0]
            else:
                in_channels, out_channels = conv_layer_parameter[0], conv_layer_parameter[0]
            if conv_layer_parameter[2] != -1:
                if self.useBatchNormalization:
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("LSTMConv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("LSTMConv1D_ReLU_{}".format(i),nn.ReLU()),
                                               ("LSTMConv1D_AvgPool1d_{}".format(i),
                                                nn.AvgPool1d(conv_layer_parameter[2]).to(self.device)
                                               ),
                                               ("LSTMConv1D_BatchNorm1d_{}".format(i),
                                                nn.BatchNorm1d(out_channels, momentum=0.7).to(self.device)
                                               )
                                                           ]))                                       
                else:                        
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("LSTMConv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("LSTMConv1D_ReLU_{}".format(i),nn.ReLU()),
                                               ("LSTMConv1D_AvgPool1d_{}".format(i),
                                                nn.AvgPool1d(conv_layer_parameter[2]).to(self.device)
                                               )
                                                           ]))
            else:
                if self.useBatchNormalization:
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("LSTMConv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("LSTMConv1D_ReLU_{}".format(i),nn.ReLU()),
                                               ("LSTMConv1D_BatchNorm1d_{}".format(i),
                                                nn.BatchNorm1d(out_channels, momentum=0.7).to(self.device)
                                               )
                                                           ]))                 
                else:
                    conv_layer = nn.Sequential(OrderedDict([
                                               ("LSTMConv1D_{}".format(i),
                                                nn.Conv1d(in_channels,
                                                          out_channels,
                                                          kernel_size=conv_layer_parameter[1], 
                                                          stride=1, padding=0).to(self.device)
                                               ),
                                               ("LSTMConv1D_ReLU_{}".format(i),nn.ReLU())
                                                           ]))
            output = conv_layer(output)
            if verbosity:
                print('conv_lstm({0}) shape:{1}'.format(i,np.shape(output)))

        output = output.view(output.size(0), -1)
        if verbosity:
            print('conv_lstm output reshape: {}'.format(np.shape(output)))
      
        out = output
        n_size = out.size(1)

        return out, n_size    
    
    
    def _get_lstm_output(self, input: torch.Tensor, verbosity = 0) -> Union[torch.Tensor, torch.Size]:
        '''
            Used only for internal call within CharacterLevelLSTMCNN class with artificial input tensor
        '''
        # LSTM pass:
        LSTM = BlockLSTM(features=input.size(2), num_layers=2)
        _lstm_out = LSTM.lstm(input)
        if verbosity:
            print('lstm shape: {}'.format(np.shape(_lstm_out[0]))) 
        if self.applyConv1DForLSTM:
            lstm_out, _ = self._get_conv_output_lstm(_lstm_out[0], verbosity=verbosity)
        else:
            # Do a moving average over hidden space channels 
            output = nn.AvgPool1d(self.LSTMAvgPoolKernelSize).to(self.device)(_lstm_out[0])
            print('AvgPool1D lstm shape: {}'.format(np.shape(output)))
            # Obtain a flattern vector as an input to the FC layer
            lstm_out = output.view(output.size(0), -1)
            
        n_size = lstm_out.size(1)

        return lstm_out, n_size    

    def _get_lstm_output_fwd(self, input: torch.Tensor, verbosity = 0) -> Union[torch.Tensor, torch.Size]:
        '''
            This method will be directly invocated from main forward method
        '''
        # LSTM pass:
        LSTM = BlockLSTM(features=input.size(2), num_layers=2)
        _lstm_out = LSTM(input)
        if verbosity:
            print('lstm shape: {}'.format(np.shape(_lstm_out)))     
        if self.applyConv1DForLSTM:    
            lstm_out, _ = self._get_conv_output_lstm(_lstm_out, verbosity=verbosity)        
        else:
            # Do a moving average over hidden space channels 
            output = nn.AvgPool1d(self.LSTMAvgPoolKernelSize).to(self.device)(_lstm_out)
            # Obtain a flattern vector as an input to the FC layer
            lstm_out = output.view(output.size(0), -1)        
        
        n_size = lstm_out.size(1)

        return lstm_out, n_size     

    def _get_fc_output(self, input: torch.Tensor, verbosity = 0) -> Union[torch.Tensor, torch.Size]:
        '''
             Used only for internal call within CharacterLevelLSTMCNN class with artificial input tensor
        '''    
        FC = BlockFCL(total_dim=self.total_dim, 
                      number_of_classes=self.number_of_classes, 
                      config=self.config, 
                      fcsize=self.configConvSize)
 
        for i in range(len(FC.fc_layers)):
            input = FC.fc_layers[i](input) 
            if verbosity:
                print('fc shape: {}'.format(np.shape(input)))   
        n_size = input.size(1)
        return input, n_size  
    
    def _get_fc_output_fwd(self, input: torch.Tensor, verbosity = 0) -> Union[torch.Tensor, torch.Size]:
        '''
            This method will be directly invocated from main forward method
        '''    
        FC = BlockFCL(total_dim=self.total_dim, 
                      number_of_classes=self.number_of_classes, 
                      config=self.config, 
                      fcsize=self.configConvSize)
        fin = FC(input) 
        
        return fin

        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Main function for processing the real input Tensor on input 
        that performs a series of successive transformations using Modules 
        defined in the constructor (CNN/Dence layers) without any supplementary 
        backpropagation
        """
        ########################################################
        # Pass out input tensor via set of convolutional layers:
        ########################################################                            
        conv_out, _ = self._get_conv_output(input)
        if self.useLSTM:
            # Create Additional LSTM signal
            lstm_out, _ = self._get_lstm_output_fwd(input)
            out = torch.cat((conv_out,lstm_out),1)
        else:
            out = conv_out
        ######################################
        # Do a forward pass through FC layers:
        ######################################                            
#         for i in range(len(self.fc_layers)):
#             out = self.fc_layers[i](out)
        out = self._get_fc_output_fwd(out)                   


        return out
