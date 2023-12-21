import tkinter as tk
import json

def show_evaluasi_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Evaluasi Klasifikasi", font=("Arial", 24))
    label.pack()

    # Read the JSON file
    with open("classification_results.json", "r") as file:
        results = json.load(file)

    # Display accuracy
    accuracy_label = tk.Label(content_frame, text=f"Accuracy: {results['accuracy']}")
    accuracy_label.pack()

    # Display classification report
    report_label = tk.Label(content_frame, text="Classification Report:")
    report_label.pack()

    for class_label, metrics in results['classification_report'].items():
        class_label = str(class_label)
        metrics_str = f"Class {class_label}: Precision={metrics['precision']}, Recall={metrics['recall']}, F1-score={metrics['f1-score']}"
        class_label = tk.Label(content_frame, text=metrics_str)
        class_label.pack()

    # Display best model parameters
    best_params_label = tk.Label(content_frame, text=f"Best Model Parameters: {results['best_model_parameters']}")
    best_params_label.pack()

    # Display test accuracy
    test_accuracy_label = tk.Label(content_frame, text=f"Test Accuracy: {results['test_accuracy']}")
    test_accuracy_label.pack()

    # Display test classification report
    test_report_label = tk.Label(content_frame, text="Test Classification Report:")
    test_report_label.pack()

    for class_label, metrics in results['test_classification_report'].items():
        class_label = str(class_label)
        metrics_str = f"Class {class_label}: Precision={metrics['precision']}, Recall={metrics['recall']}, F1-score={metrics['f1-score']}"
        class_label = tk.Label(content_frame, text=metrics_str)
        class_label.pack()

