from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Model data with actual results from your output
model_metrics = [
    {
        'name': 'Random Forest',
        'accuracy': 0.767699,
        'error': 0.232301,
        'training_time': 5.52,
        'confusion_matrix': np.array([
            [9459, 67, 184, 0, 1],
            [1350, 6250, 36, 0, 0],
            [772, 166, 1485, 0, 0],
            [2460, 0, 2, 110, 2],
            [192, 0, 1, 4, 3]
        ]),
        'classification_report': {
            'benign': {'precision': 0.66, 'recall': 0.97, 'f1-score': 0.79, 'support': 9711},
            'dos': {'precision': 0.96, 'recall': 0.82, 'f1-score': 0.89, 'support': 7636},
            'probe': {'precision': 0.87, 'recall': 0.61, 'f1-score': 0.72, 'support': 2423},
            'r2l': {'precision': 0.96, 'recall': 0.04, 'f1-score': 0.08, 'support': 2574},
            'u2r': {'precision': 0.50, 'recall': 0.01, 'f1-score': 0.03, 'support': 200},
            'accuracy': 0.767699,
            'macro avg': {'precision': 0.79, 'recall': 0.49, 'f1-score': 0.50, 'support': 22544},
            'weighted avg': {'precision': 0.82, 'recall': 0.77, 'f1-score': 0.73, 'support': 22544}
        }
    },
    {
        'name': 'K-Nearest Neighbors',
        'accuracy': 0.763174,
        'error': 0.236826,
        'training_time': 0.28,
        'confusion_matrix': np.array([
            [9443, 55, 210, 2, 1],
            [1610, 5937, 89, 0, 0],
            [595, 180, 1648, 0, 0],
            [2347, 2, 53, 171, 1],
            [105, 0, 85, 4, 6]
        ]),
        'classification_report': {
            'benign': {'precision': 0.67, 'recall': 0.97, 'f1-score': 0.79, 'support': 9711},
            'dos': {'precision': 0.96, 'recall': 0.78, 'f1-score': 0.86, 'support': 7636},
            'probe': {'precision': 0.79, 'recall': 0.68, 'f1-score': 0.73, 'support': 2423},
            'r2l': {'precision': 0.97, 'recall': 0.07, 'f1-score': 0.12, 'support': 2574},
            'u2r': {'precision': 0.75, 'recall': 0.03, 'f1-score': 0.06, 'support': 200},
            'accuracy': 0.763174,
            'macro avg': {'precision': 0.83, 'recall': 0.51, 'f1-score': 0.51, 'support': 22544},
            'weighted avg': {'precision': 0.82, 'recall': 0.76, 'f1-score': 0.73, 'support': 22544}
        }
    },
    {
        'name': 'Support Vector Machine',
        'accuracy': 0.757452,
        'error': 0.242548,
        'training_time': 107.96,
        'confusion_matrix': np.array([
            [9112, 423, 168, 7, 1],
            [1511, 6112, 13, 0, 0],
            [733, 112, 1578, 0, 0],
            [2299, 4, 2, 269, 0],
            [191, 0, 0, 4, 5]
        ]),
        'classification_report': {
            'benign': {'precision': 0.66, 'recall': 0.94, 'f1-score': 0.77, 'support': 9711},
            'dos': {'precision': 0.92, 'recall': 0.80, 'f1-score': 0.86, 'support': 7636},
            'probe': {'precision': 0.90, 'recall': 0.65, 'f1-score': 0.75, 'support': 2423},
            'r2l': {'precision': 0.96, 'recall': 0.10, 'f1-score': 0.19, 'support': 2574},
            'u2r': {'precision': 0.83, 'recall': 0.03, 'f1-score': 0.05, 'support': 200},
            'accuracy': 0.757452,
            'macro avg': {'precision': 0.85, 'recall': 0.50, 'f1-score': 0.52, 'support': 22544},
            'weighted avg': {'precision': 0.81, 'recall': 0.76, 'f1-score': 0.73, 'support': 22544}
        }
    },
    {
        'name': 'Logistic Regression',
        'accuracy': 0.751464,
        'error': 0.248536,
        'training_time': 3.29,
        'confusion_matrix': np.array([
            [8983, 92, 631, 2, 3],
            [1560, 6055, 21, 0, 0],
            [484, 92, 1842, 5, 0],
            [2522, 2, 1, 49, 0],
            [183, 3, 0, 2, 12]
        ]),
        'classification_report': {
            'benign': {'precision': 0.65, 'recall': 0.93, 'f1-score': 0.77, 'support': 9711},
            'dos': {'precision': 0.97, 'recall': 0.79, 'f1-score': 0.87, 'support': 7636},
            'probe': {'precision': 0.74, 'recall': 0.76, 'f1-score': 0.75, 'support': 2423},
            'r2l': {'precision': 0.84, 'recall': 0.02, 'f1-score': 0.04, 'support': 2574},
            'u2r': {'precision': 0.80, 'recall': 0.06, 'f1-score': 0.11, 'support': 200},
            'accuracy': 0.751464,
            'macro avg': {'precision': 0.80, 'recall': 0.51, 'f1-score': 0.51, 'support': 22544},
            'weighted avg': {'precision': 0.79, 'recall': 0.75, 'f1-score': 0.71, 'support': 22544}
        }
    },
    {
        'name': 'SGD Classifier',
        'accuracy': 0.746629,
        'error': 0.253371,
        'training_time': 7.83,
        'confusion_matrix': np.array([
            [9462, 87, 162, 0, 0],
            [1705, 5915, 16, 0, 0],
            [847, 126, 1450, 0, 0],
            [2561, 0, 8, 5, 0],
            [199, 0, 1, 0, 0]
        ]),
        'classification_report': {
            'benign': {'precision': 0.64, 'recall': 0.97, 'f1-score': 0.77, 'support': 9711},
            'dos': {'precision': 0.97, 'recall': 0.77, 'f1-score': 0.86, 'support': 7636},
            'probe': {'precision': 0.89, 'recall': 0.60, 'f1-score': 0.71, 'support': 2423},
            'r2l': {'precision': 1.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 2574},
            'u2r': {'precision': 0.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 200},
            'accuracy': 0.746629,
            'macro avg': {'precision': 0.70, 'recall': 0.47, 'f1-score': 0.47, 'support': 22544},
            'weighted avg': {'precision': 0.81, 'recall': 0.75, 'f1-score': 0.70, 'support': 22544}
        }
    },
    {
        'name': 'AdaBoost',
        'accuracy': 0.670999,
        'error': 0.329001,
        'training_time': 5.03,
        'confusion_matrix': np.array([
            [8825, 555, 331, 0, 0],
            [1444, 4375, 1817, 0, 0],
            [190, 306, 1927, 0, 0],
            [1937, 2, 635, 0, 0],
            [86, 0, 114, 0, 0]
        ]),
        'classification_report': {
            'benign': {'precision': 0.71, 'recall': 0.91, 'f1-score': 0.80, 'support': 9711},
            'dos': {'precision': 0.84, 'recall': 0.57, 'f1-score': 0.68, 'support': 7636},
            'probe': {'precision': 0.40, 'recall': 0.80, 'f1-score': 0.53, 'support': 2423},
            'r2l': {'precision': 0.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 2574},
            'u2r': {'precision': 0.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 200},
            'accuracy': 0.670999,
            'macro avg': {'precision': 0.39, 'recall': 0.46, 'f1-score': 0.40, 'support': 22544},
            'weighted avg': {'precision': 0.63, 'recall': 0.67, 'f1-score': 0.63, 'support': 22544}
        }
    },
    {
        'name': 'Neural Network (MLP)',
        'accuracy': 0.762686,
        'error': 0.237314,
        'training_time': 635.16,
        'confusion_matrix': np.array([
            [9005, 477, 225, 4, 0],
            [1120, 6340, 25, 151, 0],
            [557, 338, 1527, 1, 0],
            [2212, 28, 12, 322, 0],
            [143, 0, 48, 9, 0]
        ]),
        'classification_report': {
            'benign': {'precision': 0.69, 'recall': 0.93, 'f1-score': 0.79, 'support': 9711},
            'dos': {'precision': 0.88, 'recall': 0.83, 'f1-score': 0.86, 'support': 7636},
            'probe': {'precision': 0.83, 'recall': 0.63, 'f1-score': 0.72, 'support': 2423},
            'r2l': {'precision': 0.66, 'recall': 0.13, 'f1-score': 0.21, 'support': 2574},
            'u2r': {'precision': 0.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 200},
            'accuracy': 0.762686,
            'macro avg': {'precision': 0.61, 'recall': 0.50, 'f1-score': 0.51, 'support': 22544},
            'weighted avg': {'precision': 0.76, 'recall': 0.76, 'f1-score': 0.73, 'support': 22544}
        }
    },
    {
        'name': 'Voting Classifier',
        'accuracy': 0.768675,
        'error': 0.231325,
        'training_time': 8.41,
        'confusion_matrix': np.array([
            [9453, 56, 200, 1, 1],
            [1591, 5975, 70, 0, 0],
            [469, 166, 1788, 0, 0],
            [2460, 0, 0, 112, 2],
            [178, 0, 17, 4, 1]
        ]),
        'classification_report': {
            'benign': {'precision': 0.67, 'recall': 0.97, 'f1-score': 0.79, 'support': 9711},
            'dos': {'precision': 0.96, 'recall': 0.78, 'f1-score': 0.86, 'support': 7636},
            'probe': {'precision': 0.86, 'recall': 0.74, 'f1-score': 0.80, 'support': 2423},
            'r2l': {'precision': 0.96, 'recall': 0.04, 'f1-score': 0.08, 'support': 2574},
            'u2r': {'precision': 0.25, 'recall': 0.01, 'f1-score': 0.01, 'support': 200},
            'accuracy': 0.768675,
            'macro avg': {'precision': 0.74, 'recall': 0.51, 'f1-score': 0.51, 'support': 22544},
            'weighted avg': {'precision': 0.82, 'recall': 0.77, 'f1-score': 0.73, 'support': 22544}
        }
    },
    {
        'name': 'LightGBM',
        'accuracy': 0.771292,
        'error': 0.228708,
        'training_time': 209.27,
        'confusion_matrix': np.array([
            [9432, 70, 206, 2, 1],
            [1375, 6180, 81, 0, 0],
            [697, 163, 1563, 0, 0],
            [2371, 0, 1, 199, 3],
            [181, 0, 2, 3, 14]
        ]),
        'classification_report': {
            'benign': {'precision': 0.67, 'recall': 0.97, 'f1-score': 0.79, 'support': 9711},
            'dos': {'precision': 0.96, 'recall': 0.81, 'f1-score': 0.88, 'support': 7636},
            'probe': {'precision': 0.84, 'recall': 0.65, 'f1-score': 0.73, 'support': 2423},
            'r2l': {'precision': 0.98, 'recall': 0.08, 'f1-score': 0.14, 'support': 2574},
            'u2r': {'precision': 0.78, 'recall': 0.07, 'f1-score': 0.13, 'support': 200},
            'accuracy': 0.771292,
            'macro avg': {'precision': 0.85, 'recall': 0.51, 'f1-score': 0.54, 'support': 22544},
            'weighted avg': {'precision': 0.82, 'recall': 0.77, 'f1-score': 0.74, 'support': 22544}
        }
    },
    {
        'name': 'CatBoost',
        'accuracy': 0.805225,
        'error': 0.194775,
        'training_time': 62.38,
        'confusion_matrix': np.array([
            [9448, 72, 188, 2, 1],
            [1220, 6388, 28, 0, 0],
            [208, 200, 2015, 0, 0],
            [2278, 1, 3, 288, 4],
            [171, 0, 13, 2, 14]
        ]),
        'classification_report': {
            'benign': {'precision': 0.71, 'recall': 0.97, 'f1-score': 0.82, 'support': 9711},
            'dos': {'precision': 0.96, 'recall': 0.84, 'f1-score': 0.89, 'support': 7636},
            'probe': {'precision': 0.90, 'recall': 0.83, 'f1-score': 0.86, 'support': 2423},
            'r2l': {'precision': 0.99, 'recall': 0.11, 'f1-score': 0.20, 'support': 2574},
            'u2r': {'precision': 0.74, 'recall': 0.07, 'f1-score': 0.13, 'support': 200},
            'accuracy': 0.805225,
            'macro avg': {'precision': 0.86, 'recall': 0.56, 'f1-score': 0.58, 'support': 22544},
            'weighted avg': {'precision': 0.85, 'recall': 0.81, 'f1-score': 0.77, 'support': 22544}
        }
    }
]

def generate_confusion_matrix_image(cm, model_name):
    """Generate a confusion matrix plot and return as base64 encoded image"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    classes = ['benign', 'dos', 'probe', 'r2l', 'u2r']
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'{model_name} Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    fmt = 'd'
    thresh = np.max(cm) / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i][j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    
    fig.tight_layout()
    
    # Save plot to a temporary buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return image_base64

# Generate confusion matrix images for each model
for model in model_metrics:
    model['confusion_matrix_img'] = generate_confusion_matrix_image(
        model['confusion_matrix'], 
        model['name']
    )
    # Add weighted average f1-score to the model dictionary for easy access
    model['weighted_avg_f1'] = model['classification_report']['weighted avg']['f1-score']

@app.route('/')
def index():
    # Sort models by accuracy (descending)
    sorted_models = sorted(model_metrics, key=lambda x: x['accuracy'], reverse=True)
    
    # Convert to DataFrame for better table display
    metrics_df = pd.DataFrame(sorted_models)
    metrics_df = metrics_df.drop(columns=['confusion_matrix_img', 'confusion_matrix', 'classification_report'])
    
    # Format percentages
    metrics_df['accuracy'] = metrics_df['accuracy'].map('{:.2%}'.format)
    metrics_df['error'] = metrics_df['error'].map('{:.2%}'.format)
    metrics_df['weighted_avg_f1'] = metrics_df['weighted_avg_f1'].map('{:.2%}'.format)
    
    return render_template('index.html', 
                         models=sorted_models,
                         metrics_table=metrics_df.to_html(
                             classes='table table-striped table-hover', 
                             index=False,
                             columns=['name', 'accuracy', 'error', 'weighted_avg_f1', 'training_time']
                         ))

@app.route('/model/<model_name>')
def model_details(model_name):
    model = next((m for m in model_metrics if m['name'] == model_name), None)
    if not model:
        return "Model not found", 404
    
    # Convert classification report to DataFrame
    report = model['classification_report']
    report_data = []
    
    # Add class-specific metrics
    for label, metrics in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        report_data.append({
            'Class': label,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1-score'],
            'Support': metrics['support']
        })
    
    # Add averages
    report_data.append({
        'Class': 'Macro Avg',
        'Precision': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall'],
        'F1-Score': report['macro avg']['f1-score'],
        'Support': report['macro avg']['support']
    })
    report_data.append({
        'Class': 'Weighted Avg',
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score'],
        'Support': report['weighted avg']['support']
    })
    
    report_df = pd.DataFrame(report_data)
    
    # Format numbers
    report_df['Precision'] = report_df['Precision'].map('{:.2f}'.format)
    report_df['Recall'] = report_df['Recall'].map('{:.2f}'.format)
    report_df['F1-Score'] = report_df['F1-Score'].map('{:.2f}'.format)
    report_df['Support'] = report_df['Support'].map('{:.0f}'.format)
    
    return render_template('model_details.html',
                         model=model,
                         classification_report=report_df.to_html(
                             classes='table table-striped table-hover', 
                             index=False
                         ))

if __name__ == '__main__':
    app.run(debug=True)