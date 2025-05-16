import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Dict, Optional

class MRL_Linear_Layer(nn.Module):
    def __init__(self, nesting_list, num_classes=1000, efficient=False, **kwargs):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        self.efficient = efficient
        
        if self.efficient:
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))
        else:    
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))    

    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()

    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
                else:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
            else:
                nesting_logits += (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

        return nesting_logits

class FixedFeatureLayer(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out

class BlurPoolConv2d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

def apply_blurpool(mod: nn.Module):
    for (name, child) in mod.named_children():
        if isinstance(child, nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
            setattr(mod, name, BlurPoolConv2d(child))
        else: 
            apply_blurpool(child)


class MRLLoader:
    """
    Wrapper class for MRL models that allows easy switching between embedding dimensions
    and simplified prediction.
    """
    
    def __init__(self, weights_path: str, model_arch: str = 'resnet50', 
                 use_pretrained: bool = True, efficient: bool = True, 
                 nesting_start: int = 3, use_blurpool: bool = True,
                 num_classes: int = 1000, device: Optional[str] = None):
        """
        Initialize the MRL model wrapper.
        
        Args:
            weights_path: Path to the model weights file
            model_arch: Architecture of the base model (default: 'resnet50')
            use_pretrained: Whether to use pretrained weights for the base model
            efficient: Whether to use MRL-E (efficient mode)
            nesting_start: Starting exponent for nesting dimensions (default: 3, means 2^3=8)
            use_blurpool: Whether to apply BlurPool to the model
            num_classes: Number of classes in the model (default: 1000 for ImageNet)
            device: Device to load the model on (default: cuda if available, otherwise cpu)
        """
        self.weights_path = weights_path
        self.model_arch = model_arch
        self.use_pretrained = use_pretrained
        self.efficient = efficient
        self.nesting_start = nesting_start
        self.use_blurpool = use_blurpool
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Generate nesting list
        self.nesting_list = [2**i for i in range(self.nesting_start, 12)]  # 8, 16, 32, ..., 2048
        
        # Track current embedding dimension index
        self.current_emb_dim_index = 0
        
        # Load the model
        self._load_model()
        
        # Set transformation pipeline for input preprocessing
        self.transform = self._get_transform()
    
    def _load_model(self):
        """Load the MRL model and weights."""
        try:
            # Create the base model
            print(f"Initializing {self.model_arch} model...")
            base_model = getattr(models, self.model_arch)(pretrained=self.use_pretrained)
            
            # Replace the final layer with MRL layer
            base_model.fc = MRL_Linear_Layer(self.nesting_list, num_classes=self.num_classes, efficient=self.efficient)
            
            # Apply BlurPool if requested
            if self.use_blurpool:
                print("Applying BlurPool...")
                apply_blurpool(base_model)
            
            # Load the pretrained weights
            try:
                print(f"Loading weights from {self.weights_path}...")
                checkpoint = torch.load(self.weights_path, map_location='cpu')
                
                # Handle case where weights were saved in DataParallel format
                if list(checkpoint.keys())[0].startswith('module.'):
                    # Remove 'module.' prefix
                    clean_ckpt = {}
                    for k, v in checkpoint.items():
                        clean_ckpt[k[7:] if k.startswith('module.') else k] = v
                    checkpoint = clean_ckpt
                
                base_model.load_state_dict(checkpoint)
                print("Model weights loaded successfully!")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Continuing with pretrained weights only.")
            
            # Move model to specified device
            self.model = base_model.to(self.device)
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model.eval()
            
            print(f"Model loaded on {self.device}")
            print(f"Available embedding dimensions: {self.nesting_list}")
            
            # Set default embedding dimension to the first one
            self.set_output_emb_dim(self.nesting_list[0])
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _get_transform(self):
        """Create the transformation pipeline for input preprocessing."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def set_output_emb_dim(self, dim: int):
        """
        Set the output embedding dimension to use during prediction.
        
        Args:
            dim: Embedding dimension to use (must be one of the available dimensions)
        
        Raises:
            ValueError: If the specified dimension is not available in the model
        """
        if dim not in self.nesting_list:
            available_dims = ", ".join(str(d) for d in self.nesting_list)
            raise ValueError(f"Embedding dimension {dim} is not available. Available dimensions: {available_dims}")
        
        self.current_emb_dim_index = self.nesting_list.index(dim)
        self.current_emb_dim = dim
        print(f"Output embedding dimension set to {dim}")
    
    def _process_input(self, input_data: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray]) -> torch.Tensor:
        """
        Process various input types into a standardized tensor format.
        
        Args:
            input_data: Input data in various formats
            
        Returns:
            Processed tensor ready for model inference
        """
        # Handle different input types
        if isinstance(input_data, Image.Image):
            # Single PIL image
            img_tensor = self.transform(input_data).unsqueeze(0)
            img_tensor = img_tensor.to(self.device, memory_format=torch.channels_last)
        
        elif isinstance(input_data, list) and all(isinstance(img, Image.Image) for img in input_data):
            # List of PIL images
            tensors = [self.transform(img) for img in input_data]
            img_tensor = torch.stack(tensors)
            img_tensor = img_tensor.to(self.device, memory_format=torch.channels_last)
        
        elif isinstance(input_data, np.ndarray):
            # Numpy array
            if input_data.ndim == 3:
                # Single image, add batch dimension
                img = Image.fromarray(input_data.astype(np.uint8))
                img_tensor = self.transform(img).unsqueeze(0)
            elif input_data.ndim == 4:
                # Batch of images
                tensors = [self.transform(Image.fromarray(img.astype(np.uint8))) 
                          for img in input_data]
                img_tensor = torch.stack(tensors)
            else:
                raise ValueError(f"Invalid numpy array shape: {input_data.shape}")
            
            img_tensor = img_tensor.to(self.device, memory_format=torch.channels_last)
        
        elif isinstance(input_data, torch.Tensor):
            # Already a tensor
            if input_data.dim() == 3:
                # Single image, add batch dimension
                img_tensor = input_data.unsqueeze(0)
            else:
                img_tensor = input_data
            
            img_tensor = img_tensor.to(self.device, memory_format=torch.channels_last)
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
            
        return img_tensor
    
    def _extract_embeddings(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings from the backbone of the model.
        
        Args:
            img_tensor: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            Feature embeddings tensor
        """
        # Run the backbone of the model to get embeddings
        x = self.model.conv1(img_tensor)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def predict(self, input_data: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray], 
                return_probabilities: bool = False, return_top_k: int = None):
        """
        Make a prediction using the model with the currently set embedding dimension.
        
        Args:
            input_data: Input data, can be:
                - A single PIL Image
                - A list/batch of PIL Images
                - A preprocessed tensor of shape [batch_size, 3, height, width]
                - A numpy array of shape [batch_size, height, width, 3] or [height, width, 3]
            return_probabilities: Whether to return probability distribution (softmax)
                                  instead of raw logits. Default: False
            return_top_k: If not None, return only the top k predictions (class indices and values)
                         Default: None (return all)
        
        Returns:
            Prediction results based on the settings:
            - If return_top_k is None and return_probabilities is False: raw logits
            - If return_top_k is None and return_probabilities is True: probability distribution
            - If return_top_k is not None: tuple of (top_k_values, top_k_indices)
        """
        self.model.eval()
        
        # Process input data
        with torch.no_grad():
            # Process input to tensor format
            img_tensor = self._process_input(input_data)
            
            # Forward pass
            outputs = self.model(img_tensor)
            
            # Get output for the current embedding dimension
            logits = outputs[self.current_emb_dim_index]
            
            # Process output based on parameters
            if return_top_k is not None:
                if return_probabilities:
                    probs = F.softmax(logits, dim=1)
                    values, indices = torch.topk(probs, k=return_top_k, dim=1)
                else:
                    values, indices = torch.topk(logits, k=return_top_k, dim=1)
                
                return values, indices
            
            elif return_probabilities:
                return F.softmax(logits, dim=1)
            
            else:
                return logits
    
    def get_available_dimensions(self) -> List[int]:
        """Get the list of available embedding dimensions."""
        return self.nesting_list
    
    def get_current_dimension(self) -> int:
        """Get the currently set embedding dimension."""
        return self.current_emb_dim
    
    def get_embedding(self, input_data: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray]) -> torch.Tensor:
        """
        Extract the feature embeddings directly before the classification layer.
        This returns the raw feature vector at the selected dimension.
        
        Args:
            input_data: Input data (same format options as predict method)
            
        Returns:
            Embedding tensor of shape [batch_size, current_emb_dim]
        """
        self.model.eval()
        
        # Process input data to tensor similar to predict()
        with torch.no_grad():
            # Process input to tensor format
            img_tensor = self._process_input(input_data)
            
            # Get embeddings
            embeddings = self._extract_embeddings(img_tensor)
            
            # Return features up to the current embedding dimension
            return embeddings[:, :self.current_emb_dim]
    
    def predict_with_embeddings(self, input_data: Union[torch.Tensor, Image.Image, List[Image.Image], np.ndarray],
                               return_probabilities: bool = False, return_top_k: int = None):
        """
        Make a prediction and also return the corresponding embeddings.
        This is more efficient than calling predict() and get_embedding() separately.
        
        Args:
            input_data: Input data (same formats as predict method)
            return_probabilities: Whether to return probability distribution (softmax)
            return_top_k: If not None, return only the top k predictions
            
        Returns:
            A tuple of (predictions, embeddings) where:
            - predictions: Same output format as the predict() method based on the parameters
            - embeddings: Feature embeddings of shape [batch_size, current_emb_dim]
        """
        self.model.eval()
        
        # Process input data
        with torch.no_grad():
            # Process input to tensor format
            img_tensor = self._process_input(input_data)
            
            # Get embeddings first (more efficient than running full model twice)
            full_embeddings = self._extract_embeddings(img_tensor)
            
            # Slice the embeddings to the current dimension
            embeddings = full_embeddings[:, :self.current_emb_dim]
            
            # Now get predictions using the current classifier and embeddings
            if self.efficient:
                # For MRL-E, we need to use the appropriate slice of the weights
                classifier = getattr(self.model.fc, f"nesting_classifier_0")
                if classifier.bias is None:
                    logits = torch.matmul(embeddings, 
                                         (classifier.weight[:, :self.current_emb_dim]).t())
                else:
                    logits = torch.matmul(embeddings, 
                                         (classifier.weight[:, :self.current_emb_dim]).t()) + classifier.bias
            else:
                # For regular MRL, use the specific classifier for this dimension
                classifier = getattr(self.model.fc, f"nesting_classifier_{self.current_emb_dim_index}")
                logits = classifier(embeddings)
            
            # Process logits based on parameters
            if return_top_k is not None:
                if return_probabilities:
                    probs = F.softmax(logits, dim=1)
                    values, indices = torch.topk(probs, k=return_top_k, dim=1)
                else:
                    values, indices = torch.topk(logits, k=return_top_k, dim=1)
                
                return (values, indices), embeddings
            
            elif return_probabilities:
                return F.softmax(logits, dim=1), embeddings
            
            else:
                return logits, embeddings


# Example usage
if __name__ == "__main__":
    # Example: Load model
    model_path = "/path/to/your/model.pth"
    model = MRLLoader(weights_path=model_path)
    
    # Example: Set embedding dimension
    model.set_output_emb_dim(16)  # Will throw error if not available
    
    # Example: Get model predictions with random data
    random_input = torch.randn(4, 3, 224, 224)
    predictions = model.predict(random_input)
    print(f"Prediction shape: {predictions.shape}")
    
    # Example: Get predictions AND embeddings in one call
    predictions, embeddings = model.predict_with_embeddings(random_input)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Embedding shape: {embeddings.shape}")  # [4, 16]
    
    # Example: Get top-5 predictions with probabilities and embeddings
    (top_probs, top_indices), embeddings = model.predict_with_embeddings(
        random_input, return_probabilities=True, return_top_k=5
    )
    print(f"Top 5 class indices: {top_indices}")
    print(f"Top 5 probabilities: {top_probs}")
    print(f"Embedding shape: {embeddings.shape}")