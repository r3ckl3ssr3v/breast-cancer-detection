import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import numpy as np
from fpdf import FPDF
import argparse

class ReportGenerator:
    """Class for generating a comprehensive report of project results."""
    
    def __init__(self, results_dir, output_dir=None):
        """
        Initialize the report generator.
        
        Args:
            results_dir: Directory containing results
            output_dir: Directory to save the report
        """
        self.results_dir = results_dir
        self.output_dir = output_dir if output_dir else results_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find all model results
        self.model_names = self._get_model_names()
        
        # Load comparison data if available
        self.comparison_df = self._load_comparison_data()
        
        # Load cross-dataset comparison if available
        self.cross_dataset_df = self._load_cross_dataset_data()
    
    def _get_model_names(self):
        """Get names of all models with results."""
        model_names = []
        
        # Look for model metrics files
        for file in glob.glob(os.path.join(self.results_dir, "*_metrics.csv")):
            model_name = os.path.basename(file).replace("_metrics.csv", "")
            model_names.append(model_name)
        
        return model_names
    
    def _load_comparison_data(self):
        """Load model comparison data if available."""
        comparison_path = os.path.join(self.results_dir, "model_comparison.csv")
        if os.path.exists(comparison_path):
            return pd.read_csv(comparison_path)
        return None
    
    def _load_cross_dataset_data(self):
        """Load cross-dataset comparison data if available."""
        cross_dataset_path = os.path.join(self.results_dir, "cross_dataset_comparison.csv")
        if os.path.exists(cross_dataset_path):
            return pd.read_csv(cross_dataset_path)
        return None
    
    def generate_pdf_report(self):
        """Generate a comprehensive PDF report."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Breast Cancer Detection using Hybrid CNN and Transformer Models", ln=True, align="C")
        pdf.ln(5)
        
        # Date and time
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        
        # Introduction
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "1. Introduction", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, "This report presents the results of a comprehensive study on early detection of breast cancer using hybrid CNN and Transformer models on mammograms. The project implements and compares multiple deep learning architectures to evaluate their performance in classifying mammogram images into normal, benign, and malignant categories.")
        pdf.ln(5)
        
        # Models overview
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "2. Models Overview", ln=True)
        pdf.set_font("Arial", "", 12)
        
        models_description = {
            "CNN": "Pure CNN model based on ResNet50 architecture",
            "Transformer": "Pure Transformer model based on Vision Transformer (ViT)",
            "HybridSequential": "Hybrid model with CNN features fed into a Transformer encoder",
            "HybridParallel": "Hybrid model with parallel CNN and Transformer branches"
        }
        
        for model_name in self.model_names:
            if model_name in models_description:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"2.{self.model_names.index(model_name)+1}. {model_name}", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, models_description.get(model_name, "No description available"))
        
        pdf.ln(5)
        
        # Results
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. Results", ln=True)
        
        # Add comparison table if available
        if self.comparison_df is not None:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "3.1. Performance Metrics Comparison", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, "The table below shows the performance metrics for all models:")
            
            # Add comparison table
            pdf.set_font("Arial", "B", 10)
            
            # Table header
            metrics = [col for col in self.comparison_df.columns if col != "Model"]
            col_width = 180 / (len(metrics) + 1)
            
            pdf.cell(col_width, 10, "Model", border=1)
            for metric in metrics:
                pdf.cell(col_width, 10, metric.capitalize().replace("_", " "), border=1)
            pdf.ln()
            
            # Table data
            pdf.set_font("Arial", "", 10)
            for _, row in self.comparison_df.iterrows():
                pdf.cell(col_width, 10, row["Model"], border=1)
                for metric in metrics:
                    pdf.cell(col_width, 10, f"{row[metric]:.4f}", border=1)
                pdf.ln()
            
            pdf.ln(5)
            
            # Add comparison chart
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "3.2. Performance Metrics Visualization", ln=True)
            
            # Create and save comparison chart
            chart_path = os.path.join(self.output_dir, "metrics_comparison_chart.png")
            if not os.path.exists(chart_path):
                plt.figure(figsize=(10, 6))
                
                # Melt the dataframe for easier plotting
                plot_df = self.comparison_df.melt(id_vars=["Model"], var_name="Metric", value_name="Value")
                
                # Plot
                sns.barplot(x="Model", y="Value", hue="Metric", data=plot_df)
                plt.title("Performance Metrics Comparison")
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
            
            # Add chart to PDF
            pdf.image(chart_path, x=10, y=None, w=180)
            pdf.ln(140)  # Add space after the image
        
        # Add individual model results
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "3.3. Individual Model Results", ln=True)
        
        for i, model_name in enumerate(self.model_names):
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 10, f"3.3.{i+1}. {model_name}", ln=True)
            
            # Add confusion matrix
            cm_path = os.path.join(self.results_dir, f"{model_name}_confusion_matrix.png")
            if os.path.exists(cm_path):
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 10, "Confusion Matrix:", ln=True)
                pdf.image(cm_path, x=10, y=None, w=90)
                pdf.ln(95)  # Add space after the image
            
            # Add ROC curve
            roc_path = os.path.join(self.results_dir, f"{model_name}_roc_curve.png")
            if os.path.exists(roc_path):
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 10, "ROC Curve:", ln=True)
                pdf.image(roc_path, x=10, y=None, w=90)
                pdf.ln(95)  # Add space after the image
            
            # Add explainable AI examples if available
            explain_dir = os.path.join(self.results_dir, f"{model_name}_explanations")
            if os.path.exists(explain_dir):
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 10, "Explainable AI Visualizations:", ln=True)
                
                # Find explanation images
                explanation_images = glob.glob(os.path.join(explain_dir, "explanation_sample_*.png"))
                
                for j, img_path in enumerate(explanation_images[:2]):  # Limit to 2 examples
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 10, f"Example {j+1}:", ln=True)
                    pdf.image(img_path, x=10, y=None, w=180)
                    pdf.ln(140)  # Add space after the image
            
            # Add page break between models
            if i < len(self.model_names) - 1:
                pdf.add_page()
        
        # Cross-dataset comparison if available
        if self.cross_dataset_df is not None:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "4. Cross-Dataset Comparison", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, "This section compares model performance across different datasets:")
            
            # Create and save cross-dataset chart
            chart_path = os.path.join(self.output_dir, "cross_dataset_chart.png")
            if not os.path.exists(chart_path):
                plt.figure(figsize=(12, 8))
                
                # Plot
                sns.barplot(x="Model", y="accuracy", hue="Dataset", data=self.cross_dataset_df)
                plt.title("Accuracy Across Datasets")
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
            
            # Add chart to PDF
            pdf.image(chart_path, x=10, y=None, w=180)
            pdf.ln(140)  # Add space after the image
        
        # Statistical analysis
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "5. Statistical Analysis", ln=True)
        
        # Add statistical comparison results if available
        anova_path = os.path.join(self.results_dir, "anova_results.csv")
        if os.path.exists(anova_path):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "5.1. ANOVA Results", ln=True)
            
            # Load ANOVA results
            anova_df = pd.read_csv(anova_path)
            
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 10, "The table below shows the ANOVA results comparing all models:")
            
            # Add ANOVA table
            pdf.set_font("Arial", "B", 10)
            
            # Table header
            metrics = anova_df.columns.tolist()
            col_width = 180 / (len(metrics) + 1)
            
            pdf.cell(col_width, 10, "Metric", border=1)
            for metric in metrics:
                pdf.cell(col_width, 10, metric.capitalize().replace("_", " "), border=1)
            pdf.ln()
            
            # Table data
            pdf.set_font("Arial", "", 10)
            for metric, row in anova_df.iterrows():
                pdf.cell(col_width, 10, metric, border=1)
                for col in metrics:
                    value = row[col]
                    if isinstance(value, (int, float)):
                        pdf.cell(col_width, 10, f"{value:.4f}", border=1)
                    else:
                        pdf.cell(col_width, 10, str(value), border=1)
                pdf.ln()
        
        # Conclusion
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "6. Conclusion", ln=True)
        pdf.set_font("Arial", "", 12)
        
        # Generate conclusion based on results
        best_model = None
        best_accuracy = 0
        
        if self.comparison_df is not None:
            for _, row in self.comparison_df.iterrows():
                if row["accuracy"] > best_accuracy:
                    best_accuracy = row["accuracy"]
                    best_model = row["Model"]
        
        conclusion_text = f"""
        This study implemented and compared four different deep learning architectures for breast cancer detection from mammograms:
        1. Pure CNN model
        2. Pure Transformer model
        3. Hybrid Sequential CNN-Transformer model
        4. Hybrid Parallel CNN-Transformer model
        
        Based on the experimental results, the {best_model if best_model else 'Hybrid'} model achieved the best overall performance with an accuracy of {best_accuracy:.4f}.
        
        The explainable AI visualizations provide insights into the model's decision-making process, highlighting the regions of interest in the mammogram images that contribute most to the classification decision.
        
        Future work could focus on:
        1. Incorporating additional imaging modalities (ultrasound, MRI)
        2. Exploring more sophisticated hybrid architectures
        3. Implementing attention mechanisms specifically designed for medical imaging
        4. Collecting and incorporating more diverse datasets
        """
        
        pdf.multi_cell(0, 10, conclusion_text.strip())
        
        # Save the PDF
        report_path = os.path.join(self.output_dir, "breast_cancer_detection_report.pdf")
        pdf.output(report_path)
        
        print(f"PDF report generated: {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive project report")
    parser.add_argument("--results_dir", type=str, default="c:\\SDP_Project_P3\\results",
                        help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the report (default: same as results_dir)")
    
    args = parser.parse_args()
    
    generator = ReportGenerator(args.results_dir, args.output_dir)
    generator.generate_pdf_report()

if __name__ == "__main__":
    main()