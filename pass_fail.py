import pandas as pd
import numpy as np
import copy
import re
import os
import shap
import matplotlib.pyplot as plt


from itertools import combinations
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

from sklearn.metrics import ConfusionMatrixDisplay

import shutil
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import PatternFill

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Border, Side


# ------------------ Data Loading ------------------ #
def load_data(file_path, sheet_name, engine='openpyxl'):
    return pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)

# ------------------ Preprocessing ------------------ #
def preprocess_data(df, target_column='Total Mark', binning=True):
    df['Unit Class'] = df['Unit Class'].astype('category')
    df['Course Code'] = df['Course Code'].astype('category')
    df['Att Type'] = df['Att Type'].astype('category')
    df['Fee Category'] = df['Fee Category'].astype('category')

    
    # # Recalculate total mark using new weighting
    # df['Total Mark'] = 0.2 * df['Test 1'] + 0.2 * df['Test 2'] + 0.6 * df['Exam']

    if binning:
        bins = [0, 49.5, 101]
        grade_labels = ['Fail', 'Pass']

        # bins = [0, 49.5, 69.5, 101]
        # grade_labels = ['F', 'P', 'D']

        # bins = [0, 49.5, 59.5, 69.5, 79.5, 101]
        # grade_labels = ['F', 'P', 'C', 'D', 'HD']

        df['Grade'] = pd.cut(df[target_column], bins=bins, labels=grade_labels, right=False, include_lowest=True)
        y_cat = df['Grade'].astype('category')
        y = y_cat.cat.codes
        grade_mapping = dict(enumerate(y_cat.cat.categories))
    else:
        y = df[target_column]
        grade_mapping = None

    # One-hot encode categorical variables
    X = pd.get_dummies(df[['Unit Class', 'Course Code', 'Att Type', 'Fee Category', 'Test 1', 'Test 2']], drop_first=False) # Include all categories to avoid missing columns in prediction
    return X, y, grade_mapping

# ------------------ Model Definitions ------------------ #
def get_classification_models(scale_pos):
    return {
        'LR': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
        'SVC': Pipeline([('scaler', StandardScaler()), ('model', SVC(gamma='auto', C=1, probability=True, class_weight='balanced'))]),
        'KNN': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=5))]),
        'NB': Pipeline([('scaler', StandardScaler()), ('model', GaussianNB())]),
        'DTC': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'RFC': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGB': XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos, max_depth=3, n_estimators=50), # max depth=3, n_estimators=50
        'LGBM': LGBMClassifier(class_weight='balanced', random_state=42, min_child_samples=5, max_depth=3, n_estimators=50) # min_child_samples=5, max depth=3 num_leaves=30, n_estimators=50
        # 'XGB': XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos, max_depth=5, n_estimators=100), # max depth=3, n_estimators=50
        # 'LGBM': LGBMClassifier(class_weight='balanced', random_state=42, min_child_samples=10, max_depth=5, n_estimators=100) # min_child_samples=5, max depth=3 num_leaves=30, n_estimators=50
    }

# ------------------ Ensemble Evaluation ------------------ #
def evaluate_ensemble_combinations(base_estimators, X, y):
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for r in range(1, len(base_estimators) + 1):
        for combo in combinations(base_estimators.items(), r):
            name_combo = [name for name, _ in combo]
            estimators_combo = [(name, copy.deepcopy(model)) for name, model in combo]
            # Create a VotingClassifier (or other ensemble) for this combo
            vot_model = VotingClassifier(estimators=estimators_combo, voting='soft')
            metrics = evaluate_model(vot_model, X, y, cv)
            results.append({'Models': ', '.join(name_combo), **metrics})

    return pd.DataFrame(results)

# ------------------ Evaluation ------------------ #
def evaluate_model(model, X, y, cv):
    roc_auc_scores, acc_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train_CV, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_CV, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_CV, y_train_CV)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        all_classes = np.unique(y)
        y_val_bin = label_binarize(y_val, classes=all_classes)
        if y_val_bin.shape[1] == 1:
            y_val_bin = np.hstack([1 - y_val_bin, y_val_bin])

        roc_auc_scores.append(roc_auc_score(y_val_bin, y_prob, average='macro'))
        acc_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, average='macro'))
        recall_scores.append(recall_score(y_val, y_pred, average='macro'))
        f1_scores.append(f1_score(y_val, y_pred, average='macro'))

    return {
        'Mean ROC-AUC': round(np.mean(roc_auc_scores), 4),
        'Mean Accuracy': round(np.mean(acc_scores), 4),
        'Mean Precision': round(np.mean(precision_scores), 4),
        'Mean Recall': round(np.mean(recall_scores), 4),
        'Mean F1-Score': round(np.mean(f1_scores), 4)
    }

# ------------------ Best Model Selection ------------------ #
def select_best_ensemble(results_df, base_estimators):
    if 'Models' not in results_df.columns:
        raise KeyError("Expected 'Models' column in results_df but it was not found.")

    results_df.sort_values(by='Mean ROC-AUC', ascending=False, inplace=True)
    best_row = results_df.iloc[0]
    best_models_names = [m.strip() for m in best_row['Models'].split(',')]
    best_estimators_combo = [(name, copy.deepcopy(base_estimators[name])) for name in best_models_names]
    return VotingClassifier(estimators=best_estimators_combo, voting='soft')

# ------------------ Final Evaluation ------------------ #
def evaluate_on_test_set(model, X_test, y_test, grade_mapping):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    classes = model.classes_
    n_classes = len(classes)
    target_names = [str(grade_mapping[c]) for c in classes]

    if n_classes == 2:
        pos_code = next(k for k, v in grade_mapping.items() if v == 'P')
        pos_idx = int(np.where(classes == pos_code)[0][0])
        p_pos = y_prob[:, pos_idx]
        threshold = 0.50
        y_true_bin = (y_test == pos_code).astype(int)
        y_pred_bin = (p_pos >= threshold).astype(int)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        return cm, acc, prec, rec, f1, target_names
    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        return cm, acc, prec, rec, f1, target_names
    
# ------------------ Write Results to Excel ------------------ #
def write_ensemble_results_to_excel(results_df, cm_df, summary_df, target_names, excel_file, data_source="MAT1114 Data More Attributes.xlsx"):
    # Create a new workbook and sheet if the file doesn't exist, otherwise append a new sheet
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet_title = f"Run {timestamp.replace(':', '-').replace(' ', '-')}"

    if os.path.exists(excel_file):
        wb = load_workbook(excel_file)
        ws = wb.create_sheet(title=sheet_title)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_title


    # Define styles for the header i.e., bold font and borders
    bold_font = Font(bold=True)
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Header
    header = f"Data Source: {data_source} | Test Name: Ensemble Evaluation Run | Timestamp: {timestamp}"
    ws.cell(row=1, column=1, value=header)

    # Statistics table
    stats = results_df[['Mean ROC-AUC', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1-Score']].agg(['mean', 'median', 'max', 'min'])
    stats.reset_index(inplace=True)
    stats.rename(columns={'index': 'Statistic'}, inplace=True)

    stat_start_row = 3
    for r_idx, row in enumerate(dataframe_to_rows(stats, index=False, header=True), start=stat_start_row):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == stat_start_row:  
                cell.font = bold_font
                cell.border = thin_border


    # Model metrics table
    model_metrics = results_df[['Mean ROC-AUC', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1-Score']].copy()
    if 'Models' in results_df.columns:
        model_metrics.insert(0, 'Models', results_df['Models'].astype(str))
    else:
        model_metrics.insert(0, 'Models', results_df.index.astype(str))

    model_start_row = stat_start_row + len(stats) + 2
    for r_idx, row in enumerate(dataframe_to_rows(model_metrics, index=False, header=True), start=model_start_row):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == model_start_row:  
                cell.font = bold_font
                cell.border = thin_border

    # Summary table
    summary_start_row = model_start_row
    for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), start=summary_start_row):
        for c_idx, value in enumerate(row, start=8):  # Start from column H
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == summary_start_row:  
                cell.font = bold_font
                cell.border = thin_border

    # Confusion matrix table
    # Remove index name to avoid extra row
    cm_df.index.name = None
    cm_start_row = summary_start_row + 2 + len(summary_df)
    for r_idx, row in enumerate(dataframe_to_rows(cm_df, index=True, header=True), start=cm_start_row):
        for c_idx, value in enumerate(row, start=8):  # Start from column H
            cell = ws.cell(row=r_idx, column=c_idx, value=value)

            # Apply bold font and border to header row and first column
            if r_idx == cm_start_row or c_idx == 8:  # First row or first column
                cell.font = bold_font
                cell.border = thin_border

    # Create & save the confusion matrix image
    cm = cm_df.values  # Extract the raw confusion matrix as a NumPy array

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig = disp.plot(values_format='d').figure_
    fig.set_size_inches(3, 2)
    fig.tight_layout()
    img_path = f"{sheet_title}_{'binary' if len(cm_df) == 2 else 'multiclass'}_cm.png"
    # img_path = "confusion_matrix.png"
    fig.savefig(img_path, dpi=200)
    plt.close(fig)
    
    ws.add_image(XLImage(img_path), "H16") # Fixed position for simplicity


    # Save workbook
    wb.save(excel_file)



# ------------------ SHAP Analysis ------------------ #
def generate_shap_plots(best_ensemble, X_train, y_train, shap_dir='shap_plots_v1', top_n=10):
    """
    Generate SHAP summary plots for each base estimator in a VotingClassifier,
    and produce an overall feature importance summary plot.
    Supports tree-based and linear models.
    """

    os.makedirs(shap_dir, exist_ok=True)
    X_train_numeric = pd.get_dummies(X_train)
    feature_importances = pd.DataFrame(0, index=X_train_numeric.columns, columns=['importance'])
    timestamp = datetime.now().strftime("Run %Y-%m-%d %H-%M-%S")

    for name, model in best_ensemble.named_estimators_.items():
        try:
            model.fit(X_train, y_train)

            # Handle pipelines
            if hasattr(model, 'named_steps'):
                # final_model = model.named_steps.get('classifier', model.steps[-1][1])
                final_model = model.named_steps.get('model', model.steps[-1][1])
                if 'preprocessor' in model.named_steps:
                    X_transformed = model.named_steps['preprocessor'].transform(X_train)
                    try:
                        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
                    except:
                        X_transformed = pd.DataFrame(X_transformed)
                else:
                    final_model = model
                    X_transformed = X_train_numeric
            else:
                final_model = model
                X_transformed = X_train_numeric

            # Select appropriate SHAP explainer
            model_type = name.lower()
            if 'xgb' in model_type or 'lgbm' in model_type or 'tree' in model_type or 'rf' in model_type:
                explainer = shap.TreeExplainer(final_model)
            elif 'logistic' in model_type or 'lr' in model_type:
                explainer = shap.LinearExplainer(final_model, X_transformed)
            else:
                explainer = shap.Explainer(
                    final_model.predict_proba if hasattr(final_model, 'predict_proba') else final_model.predict,
                    X_transformed
                )

            shap_values = explainer(X_transformed)

            # Save individual SHAP plot
            plt.figure()
            shap.summary_plot(shap_values, X_transformed, show=False)
            fig = plt.gcf()  # get SHAP’s figure
            fig.savefig(os.path.join(shap_dir, f"{timestamp} {name}_shap.png"), bbox_inches='tight')
            plt.close(fig)

            # Aggregate mean absolute SHAP values for overall plot
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            temp_df = pd.DataFrame(mean_abs, index=X_transformed.columns, columns=['importance'])
            feature_importances = feature_importances.add(temp_df, fill_value=0)

        except Exception as e:
            print(f"SHAP failed for {name}: {e}")

    # Normalize by number of base models
    feature_importances /= len(best_ensemble.named_estimators_)

    # Plot overall SHAP summary
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    feature_importances.head(top_n).plot(kind='barh', legend=False)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP value|")
    plt.title("Overall SHAP Feature Importance (VotingClassifier)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{timestamp} Overall_shap_summary.png"))
    plt.close()
    print(f"Overall SHAP summary saved to {os.path.join(shap_dir, '{timestamp} Overall_shap_summary.png')}")


# ------------------ Prediction for New Students ------------------ #
def predict_student_grade(vot_model, X_train_columns, unit_class, course_code, test1, test2, att_type, fee_category,
                          grade_mapping, at_risk_grades=['Fail'], confidence_threshold=0.7):
    
    new_student = pd.DataFrame(np.zeros((1, len(X_train_columns))), columns=X_train_columns, dtype=float)

    # print(X_train_columns)

    if 'Test 1' in new_student.columns:
        new_student.at[0, 'Test 1'] = test1
    if 'Test 2' in new_student.columns:
        new_student.at[0, 'Test 2'] = test2
    for col_prefix, value in [('Unit Class', unit_class), ('Course Code', course_code), ('Att Type', att_type), ('Fee Category', fee_category)]:
        col_name = f"{col_prefix}_{value}"
        if col_name in new_student.columns:
            new_student.at[0, col_name] = 1

    # pd.set_option('display.max_columns', None)
    # print("Input to model:\n", new_student)

    pred_numeric = vot_model.predict(new_student)[0]
    pred_proba = vot_model.predict_proba(new_student)[0]
    # print("Predicted class:", pred_numeric)
    # print("Probabilities:", pred_proba)
    max_prob = pred_proba.max()
    predicted_grade = grade_mapping.get(pred_numeric, "Unknown")
    at_risk = predicted_grade in at_risk_grades and max_prob >= confidence_threshold
    return predicted_grade, max_prob, at_risk


# ------------------ Load Prediction Sheet ------------------ #
def load_prediction_data(local_file, sheet_name):
    wb = load_workbook(local_file) if os.path.exists(local_file) else None
    if wb is None:
        raise FileNotFoundError(f"{local_file} not found.")
    ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.create_sheet(sheet_name)
    try:
        df = pd.read_excel(local_file, sheet_name=sheet_name, engine='openpyxl')
    except ValueError:
        df = pd.DataFrame(columns=['Test 1', 'Test 2', 'Unit Class', 'Course Code', 'Att Type', 'Fee Category',
                                   'Predicted Grade', 'Confidence', 'At-Risk'])
    for col in ['Predicted Grade', 'Confidence', 'At-Risk']:
        if col not in df.columns:
            df[col] = ""
    return wb, ws, df

# ------------------ Apply Predictions ------------------ #
def apply_predictions(df, model, X_train_columns, grade_mapping):
    for idx, row in df.iterrows():
        test1 = row['Test 1']
        test2 = row['Test 2']
        unit_class = row['Unit Class']
        course_code = row['Course Code']
        att_type = row['Att Type']
        fee_category = row['Fee Category']
        pred_grade, confidence, is_at_risk = predict_student_grade(
            vot_model=model,
            X_train_columns=X_train_columns,
            unit_class=unit_class,
            course_code=course_code,
            test1=test1,
            test2=test2,
            att_type=att_type,
            fee_category=fee_category,
            grade_mapping=grade_mapping
        )
        df.at[idx, 'Predicted Grade'] = pred_grade
        df.at[idx, 'Confidence'] = confidence
        df.at[idx, 'At-Risk'] = "Yes" if is_at_risk else "No"
    return df

# ------------------ Write Predictions to Excel ------------------ #
def write_predictions_to_excel(df, ws):
    for col_idx, col_name in enumerate(df.columns, start=1):
        ws.cell(row=1, column=col_idx, value=col_name)
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row)
    dark_red_fill = PatternFill(start_color="FF3333", end_color="FF3333", fill_type="solid")
    red_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")
    green_fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")
    at_risk_df = df[df['At-Risk'] == "Yes"]
    top_10_percent = at_risk_df['Confidence'].quantile(0.8) if not at_risk_df.empty else 1.0
    at_risk_col = df.columns.get_loc('At-Risk') + 1
    conf_col = df.columns.get_loc('Confidence') + 1
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        for col_idx, value in enumerate(row, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)
        if row[at_risk_col - 1] == "Yes":
            fill = dark_red_fill if row[conf_col - 1] >= top_10_percent else red_fill
        else:
            fill = green_fill
        for col_idx in range(1, len(df.columns) + 1):
            ws.cell(row=row_idx, column=col_idx).fill = fill

# ------------------ Run Prediction Workflow ------------------ #
def run_prediction_workflow(model, X_train_columns, grade_mapping, local_file, cloud_file, sheet_name="Prediction"):
    wb, ws, df = load_prediction_data(local_file, sheet_name)
    df = apply_predictions(df, model, X_train_columns, grade_mapping)
    write_predictions_to_excel(df, ws)
    wb.save(local_file)
    shutil.copy(local_file, cloud_file)
    print(f"Predictions added and formatted with top 10% at-risk highlighted in '{cloud_file}'")

def export_full_results_to_excel(results_df, cm_df, summary_df, excel_file, data_source="MAT1114 Data More Attributes.xlsx"):
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    from datetime import datetime

    wb = Workbook()
    ws = wb.active
    timestamp = datetime.now().strftime("Run %Y-%m-%d %H-%M-%S")
    ws.title = timestamp

    # Header
    header = f"Data Source: {data_source} | Test Name: Ensemble Evaluation Run | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws.cell(row=1, column=1, value=header)

    # Statistics
    stats = results_df[['Mean ROC-AUC', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1-Score']].agg(['mean', 'median', 'max', 'min'])
    stats.reset_index(inplace=True)
    stats.rename(columns={'index': 'Statistic'}, inplace=True)

    stat_start = 3
    for r_idx, row in enumerate(dataframe_to_rows(stats, index=False, header=True), start=stat_start):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # Model metrics
    model_metrics = results_df[['Mean ROC-AUC', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1-Score']].copy()
    model_metrics.insert(0, 'Model', results_df.index.astype(str))

    model_start = stat_start + len(stats) + 3
    for r_idx, row in enumerate(dataframe_to_rows(model_metrics, index=False, header=True), start=model_start):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # Confusion matrix
    cm_start = model_start + len(model_metrics) + 3
    for r_idx, row in enumerate(dataframe_to_rows(cm_df, index=False, header=True), start=cm_start):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # Summary
    summary_start = cm_start + len(cm_df) + 3
    for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), start=summary_start):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    wb.save(excel_file)

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# ------------------ Main Workflow ------------------ #

def main():
    # Load and preprocess data
    file_path = r"C:\Users\OneDrive\Data.xlsx"
    sheet_name = "Combined All Three"

    df = load_data(file_path, sheet_name)
    X, y, grade_mapping = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Handle class imbalance on training data only
    n_pos = np.sum(y_train == 1)
    # Avoid division by zero
    n_neg = np.sum(y_train == 0)
    # If there are no positive samples, set scale_pos to 1
    scale_pos = n_neg / n_pos   # Compute weights based on class frequencies 

    # Get base models
    base_estimators = get_classification_models(scale_pos)

    # Evaluate ensemble combinations using ONLY training data
    results_df = evaluate_ensemble_combinations(base_estimators, X_train, y_train)
    # Sort results by Mean ROC-AUC
    results_df.sort_values(by='Mean ROC-AUC', ascending=False, inplace=True)

    # Select best ensemble
    best_ensemble = select_best_ensemble(results_df, base_estimators)

    # Train best ensemble on full training data
    best_ensemble.fit(X_train, y_train)

    # Evaluate on the held-out test set. NOTE: line 546 to 601 is similar to evaluate_on_test_set function (to be done later)
    y_pred_test = best_ensemble.predict(X_test)
    # Get probabilities for ROC-AUC and other metrics
    y_prob_test = best_ensemble.predict_proba(X_test)

    # Prepare confusion matrix and summary
    classes = best_ensemble.classes_
    n_classes = len(classes)
    target_names = [str(grade_mapping[c]) for c in classes]

    if n_classes == 2:
        pos_code = next(k for k, v in grade_mapping.items() if v == 'Pass')
        pos_idx = int(np.where(classes == pos_code)[0][0])
        p_pos = y_prob_test[:, pos_idx]
        #  take threshold of 0.5 for binary classification as the passing mark is 50%
        threshold = 0.50
        y_true_bin = (y_test == pos_code).astype(int)
        y_pred_bin = (p_pos >= threshold).astype(int)

        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=['Actual Fail', 'Actual Pass'], columns=['Pred Fail', 'Pred Pass'])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        summary_df = pd.DataFrame([{
            'Case': 'Pass vs Fail',
            'Positive': 'Pass',
            'Threshold': threshold,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
        }])
    else:
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

        cm = confusion_matrix(y_test, y_pred_test, labels=classes)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {grade_mapping[c]}" for c in classes],
            columns=[f"Pred {grade_mapping[c]}" for c in classes]
        )
        summary_df = pd.DataFrame([{
            'Case': 'Multiclass',
            'Accuracy': round(acc, 4),
            'Precision (macro)': round(prec, 4),
            'Recall (macro)': round(rec, 4),
            'F1 (macro)': round(f1, 4),
        }])

    # Export to Excel in the correct format
    write_ensemble_results_to_excel(
        results_df=results_df,
        cm_df=cm_df,
        summary_df=summary_df,
        target_names=target_names,
        excel_file="Ensemble Combo Results 20251017.xlsx",
        data_source="MAT1114 Data More Attributes.xlsx"
    )

    # ------------------ SHAP Plots ------------------ #
    # generate_shap_plots({"best_model": best_ensemble}, X_train, y_train) # This give three plots for LR, XGB, LGBM

    generate_shap_plots(best_ensemble, X_train, y_train)  # This gives individual plots + overall plot


    # ------------------ Predict New Students ------------------ #
    local_file = r"C:\Temp\Prediction 252.xlsx"
    cloud_file = r"C:\Users\OneDrive\Prediction 252.xlsx"
    run_prediction_workflow(
        model=best_ensemble,
        X_train_columns=X_train.columns,
        grade_mapping=grade_mapping,
        local_file=local_file,
        cloud_file=cloud_file,
        sheet_name="Prediction"
    )

if __name__ == "__main__":
    main()
