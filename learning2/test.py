from sklearn.metrics import classification_report
import json

# Contoh data
y_test = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1]

# Mendapatkan classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Menyimpan classification report dalam bentuk JSON
with open('classification_report.json', 'w') as json_file:
    json.dump(report, json_file, indent=4)

print("Classification report telah disimpan dalam file 'classification_report.json'")
