import torch.nn as nn
from transformers import AutoModel
import torch

class InfluencerProfiler(nn.Module):

    def __init__(self, model ,n_classes):
        super(InfluencerProfiler, self).__init__()
        self.pretrained = AutoModel.from_pretrained(model)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.pretrained.config.hidden_size, 350)
        self.fc2 = nn.Linear(350,n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
  
    def forward(self, input_ids, attention_mask):
        output = self.pretrained(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        # Get the first element of output which is the hidden state
        # Get the embeddings of CLS token
        cls_embeddings = output[0][:,0,:]

        # Layer 1
        output = self.drop(cls_embeddings)
        output = self.fc1(output)
        output = self.relu(output)
        
        # Layer 2
        output = self.drop(output)
        output = self.fc2(output)
        output = self.relu(output)
        return self.softmax(output)
        

    def requires_grad_embeddings(self, val):
        for param in self.pretrained.parameters():
            param.requires_grad = val   


# Save the model to a file
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Load the model from a file
def load_model(model_class, n_classes,n_features, file_path):
    model = model_class(n_classes, n_features)
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {file_path}")
    return model