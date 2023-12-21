import tkinter as tk
import json

def show_evaluasi_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Evaluasi Klasifikasi", font=("Arial", 24))
    label.pack()

    # Read the JSON file
    with open("classification_results.json", "r") as file:
        results = json.load(file)

    # Display overall evaluation metrics
    overall_metrics_label = tk.Label(content_frame, text="Overall Evaluation Metrics:")
    overall_metrics_label.pack()

    accuracy_label = tk.Label(content_frame, text=f"Accuracy: {results['accuracy']}")
    accuracy_label.pack()

    # Display best model parameters
    best_params_label = tk.Label(content_frame, text=f"Best Model Parameters: {results['best_model_parameters']}")
    best_params_label.pack()

    # Display confusion matrix if available
    if 'confusion_matrix' in results:
        confusion_matrix_label = tk.Label(content_frame, text="Confusion Matrix:")
        confusion_matrix_label.pack()

        for i in range(len(results['confusion_matrix'])):
            row_label = tk.Label(content_frame, text=f"{results['confusion_matrix'][i]}")
            row_label.pack()

    # Display per-class evaluation metrics
    class_metrics_label = tk.Label(content_frame, text="Per-Class Evaluation Metrics:")
    class_metrics_label.pack()

    for class_label, metrics in results['classification_report'].items():
        if class_label.isdigit():  # Check if the key is a digit (to avoid 'accuracy', 'macro avg', 'weighted avg', etc.)
            metrics_str = f"Class {class_label}: Precision={metrics['precision']}, Recall={metrics['recall']}, F1-score={metrics['f1-score']}, Support={metrics['support']}"
            class_label = tk.Label(content_frame, text=metrics_str)
            class_label.pack()

    # Display predictions and true labels
    predictions_label = tk.Label(content_frame, text=f"Predictions: {results['predictions']}")
    predictions_label.pack()

    true_labels_label = tk.Label(content_frame, text=f"True Labels: {results['true_labels']}")
    true_labels_label.pack()

    # Display test accuracy and report
    test_accuracy_label = tk.Label(content_frame, text=f"Test Accuracy: {results['test_accuracy']}")
    test_accuracy_label.pack()

    test_report_label = tk.Label(content_frame, text="Test Classification Report:")
    test_report_label.pack()

    for class_label, metrics in results['test_classification_report'].items():
        if class_label.isdigit():
            metrics_str = f"Class {class_label}: Precision={metrics['precision']}, Recall={metrics['recall']}, F1-score={metrics['f1-score']}, Support={metrics['support']}"
            class_label = tk.Label(content_frame, text=metrics_str)
            class_label.pack()

# Now you can call this function with the appropriate content frame in your application.
