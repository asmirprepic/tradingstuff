import pandas as pd
import numpy as np
import torch 
import torch.nn as optim
from sklearn.model_selection import train_test_split

class TransformerClassifier(nn.module): 
  def __init__(self,input_dim, model_dim, num_heads, num_layers, num_classes, max_seq_len):
    """
    A transformer based classifier. 

    Args:
      input_dim (int): Number of features per time step
      model_dim (int): Dimension projected space
      num_heads (int): Number of heads
      num_layers (int): Number of transfomer encoder layers. 
      num_classes (int): Number of output classes. 
      max_seq_len (int): Length of the input sequence

    """

    super(TransformerClassifier,self).__init__()
    self.input_projection = nn.Linear(input_dim, model_dim)
    self.pos_embedding = nn.Parameter(torch.zeros(1,max_seq_len,model_dim))
    encoder_layer = nn.TransformerEncoderLayer(d_model = model_dim, nhead = num_heads)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
    self.fc = nn.Linear(model_dim, num_classes)
def forward(self,x):
  x = self.input_projection(x) + self.pos_embedding[:,:x.size(1),:]
  x = self.transfomer_encoder(x.transpose(0,1))
  x = x.mean(dim = 0)
  logits = self.fc(x)
  return logits


