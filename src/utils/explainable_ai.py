import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients,
    LayerGradCam,
    Occlusion,
    NoiseTunnel,
    visualization
)

class ExplainableAI:
    """Class for implementing explainable AI techniques."""
    
    def __init__(self, model, device=None):
        """
        Initialize the ExplainableAI class.
        
        Args:
            model: Trained PyTorch model
            device: Device to use for computation
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def _preprocess_image(self, image_path):
        """
        Preprocess an image for the model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        from torchvision import transforms
        
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply preprocessing
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)
        return img_tensor
    
    def _postprocess_attribution(self, attribution, original_image=None):
        """
        Postprocess attribution for visualization.
        
        Args:
            attribution: Attribution from explainability method
            original_image: Original image for overlay
            
        Returns:
            Processed attribution for visualization
        """
        attribution = attribution.cpu().detach().numpy()
        
        # Squeeze dimensions if needed
        if len(attribution.shape) == 4:
            attribution = attribution.squeeze(0)
        
        # Convert to absolute values and normalize
        attribution = np.abs(attribution)
        attribution = np.transpose(attribution, (1, 2, 0))
        
        # Convert to heatmap
        heatmap = np.sum(attribution, axis=2)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Overlay on original image if provided
        if original_image is not None:
            original_image = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))
            overlayed = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
            return overlayed
        
        return heatmap
    
    def integrated_gradients(self, image_path, target_class=None):
        """
        Apply Integrated Gradients method.
        
        Args:
            image_path: Path to the image file
            target_class: Target class for attribution (default: predicted class)
            
        Returns:
            Attribution and visualization
        """
        # Preprocess image
        img_tensor = self._preprocess_image(image_path)
        
        # Get prediction if target class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(img_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Initialize Integrated Gradients
        ig = IntegratedGradients(self.model)
        
        # Compute attributions
        attributions = ig.attribute(img_tensor, target=target_class, n_steps=50)
        
        # Load original image for visualization
        original_img = cv2.imread(image_path)
        
        # Process attribution for visualization
        visualization = self._postprocess_attribution(attributions, original_img)
        
        return attributions, visualization, target_class
    
    def grad_cam(self, image_path, target_layer, target_class=None):
        """
        Apply Grad-CAM method.
        
        Args:
            image_path: Path to the image file
            target_layer: Target layer for Grad-CAM
            target_class: Target class for attribution (default: predicted class)
            
        Returns:
            Attribution and visualization
        """
        # Preprocess image
        img_tensor = self._preprocess_image(image_path)
        
        # Get prediction if target class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(img_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Initialize Grad-CAM
        grad_cam = LayerGradCam(self.model, target_layer)
        
        # Compute attributions
        attributions = grad_cam.attribute(img_tensor, target=target_class)
        
        # Load original image for visualization
        original_img = cv2.imread(image_path)
        
        # Process attribution for visualization
        visualization = self._postprocess_attribution(attributions, original_img)
        
        return attributions, visualization, target_class
    
    def occlusion(self, image_path, target_class=None, sliding_window_shapes=(3, 3, 3), strides=3):
        """
        Apply Occlusion method.
        
        Args:
            image_path: Path to the image file
            target_class: Target class for attribution (default: predicted class)
            sliding_window_shapes: Shape of the sliding window
            strides: Stride for the sliding window
            
        Returns:
            Attribution and visualization
        """
        # Preprocess image
        img_tensor = self._preprocess_image(image_path)
        
        # Get prediction if target class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(img_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Initialize Occlusion
        occlusion = Occlusion(self.model)
        
        # Compute attributions
        attributions = occlusion.attribute(
            img_tensor,
            target=target_class,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides
        )
        
        # Load original image for visualization
        original_img = cv2.imread(image_path)
        
        # Process attribution for visualization
        visualization = self._postprocess_attribution(attributions, original_img)
        
        return attributions, visualization, target_class
    
    def visualize_multiple_methods(self, image_path, target_layer=None, target_class=None, save_path=None):
        """
        Apply and visualize multiple explainability methods.
        
        Args:
            image_path: Path to the image file
            target_layer: Target layer for Grad-CAM
            target_class: Target class for attribution (default: predicted class)
            save_path: Path to save the visualization
            
        Returns:
            Visualization figure
        """
        # Preprocess image
        img_tensor = self._preprocess_image(image_path)
        
        # Get prediction if target class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(img_tensor)
                target_class = torch.argmax(output, dim=1).item()
                probs = torch.softmax(output, dim=1)
                confidence = probs[0, target_class].item()
        
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (224, 224))
        
        # Create figure
        plt.figure(figsize=(20, 10))
        
        # Plot original image
        plt.subplot(2, 3, 1)
        plt.imshow(original_img)
        plt.title(f"Original Image\nPredicted: Class {target_class} ({confidence:.2f})")
        plt.axis('off')
        
        # Integrated Gradients
        ig = IntegratedGradients(self.model)
        ig_attr = ig.attribute(img_tensor, target=target_class, n_steps=50)
        ig_vis = self._postprocess_attribution(ig_attr)
        ig_vis = cv2.cvtColor(ig_vis, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 3, 2)
        plt.imshow(ig_vis)
        plt.title("Integrated Gradients")
        plt.axis('off')
        
        # Grad-CAM
        if target_layer is not None:
            grad_cam = LayerGradCam(self.model, target_layer)
            gc_attr = grad_cam.attribute(img_tensor, target=target_class)
            gc_vis = self._postprocess_attribution(gc_attr)
            gc_vis = cv2.cvtColor(gc_vis, cv2.COLOR_BGR2RGB)
            
            plt.subplot(2, 3, 3)
            plt.imshow(gc_vis)
            plt.title("Grad-CAM")
            plt.axis('off')
        
        # Occlusion
        occlusion = Occlusion(self.model)
        occ_attr = occlusion.attribute(
            img_tensor,
            target=target_class,
            sliding_window_shapes=(3, 3, 3),
            strides=3
        )
        occ_vis = self._postprocess_attribution(occ_attr)
        occ_vis = cv2.cvtColor(occ_vis, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 3, 4)
        plt.imshow(occ_vis)
        plt.title("Occlusion")
        plt.axis('off')
        
        # DeepLift
        deep_lift = DeepLift(self.model)
        dl_attr = deep_lift.attribute(img_tensor, target=target_class)
        dl_vis = self._postprocess_attribution(dl_attr)
        dl_vis = cv2.cvtColor(dl_vis, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 3, 5)
        plt.imshow(dl_vis)
        plt.title("DeepLift")
        plt.axis('off')
        
        # GradientShap
        gradient_shap = GradientShap(self.model)
        # Generate baseline
        baseline = torch.zeros_like(img_tensor)
        gs_attr = gradient_shap.attribute(img_tensor, baselines=baseline, target=target_class)
        gs_vis = self._postprocess_attribution(gs_attr)
        gs_vis = cv2.cvtColor(gs_vis, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 3, 6)
        plt.imshow(gs_vis)
        plt.title("GradientShap")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()