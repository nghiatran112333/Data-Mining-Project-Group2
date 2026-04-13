import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Thử import XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# ==============================================================================
# 2. MODEL TRAINING & EVALUATION (TỐI ƯU)
# ==============================================================================

def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocess, use_xgb: bool):
    """
    Huấn luyện và đánh giá 5 mô hình ML.
    Hỗ trợ lưu/tải mô hình bằng joblib để tăng tốc độ.
    """
    model_file_path = "credit_scoring_models.pkl"

    # -- 1. NẾU ĐÃ CÓ FILE MÔ HÌNH THÌ LOAD LUÔN --
    if os.path.exists(model_file_path):
        try:
            saved_data = joblib.load(model_file_path)
            models = saved_data["models"]
            metrics_df = saved_data["metrics_df"]
            feat_names = saved_data["feat_names"]
            
            # Cần predict lại để lấy preds/probs cho tập test hiện tại (vì X_test có thể thay đổi do shuffle)
            preds = {}
            probs = {}
            
            # Re-evaluate
            for name, pipe in models.items():
                y_pred = pipe.predict(X_test)
                preds[name] = y_pred
                try:
                    if hasattr(pipe.named_steps["clf"], "predict_proba"):
                        probs[name] = pipe.predict_proba(X_test)[:, 1]
                    else:
                        probs[name] = None
                except:
                    probs[name] = None
                    
            return models, metrics_df, preds, probs, feat_names
        except Exception as e:
            st.warning(f"Không tải được mô hình cũ ({e}). Đang huấn luyện lại...")

    # -- 2. NẾU CHƯA CÓ THÌ TRAIN MỚI --
    models = {
        "LogisticRegression": Pipeline(steps=[
            ("pre", preprocess),
            ("clf", LogisticRegression(
                max_iter=500,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            ))
        ]),
        "RandomForest": Pipeline(steps=[
            ("pre", preprocess),
            ("clf", RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ))
        ])
    }

    if use_xgb and XGBClassifier is not None:
        models["XGBoost"] = Pipeline(steps=[
            ("pre", preprocess),
            ("clf", XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            ))
        ])
    else:
        # Fallback if XGB is not available
        models["KNeighbors"] = Pipeline(steps=[
            ("pre", preprocess),
            ("clf", KNeighborsClassifier(n_neighbors=11))
        ])

    metrics = []
    preds = {}
    probs = {}

    progress_text = "Đang huấn luyện các mô hình..."
    my_bar = st.progress(0, text=progress_text)
    total_models = len(models)

    for i, (name, pipe) in enumerate(models.items(), start=1):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        preds[name] = y_pred

        try:
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                probs[name] = pipe.predict_proba(X_test)[:, 1]
            else:
                probs[name] = None
        except Exception:
            probs[name] = None

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }

        if probs[name] is not None:
            fpr, tpr, _ = roc_curve(y_test, probs[name])
            row["ROC-AUC"] = auc(fpr, tpr)

        metrics.append(row)

        my_bar.progress(i / total_models, text=f"Đang huấn luyện mô hình: {name} ({i}/{total_models})")

    my_bar.empty()

    metrics_df = pd.DataFrame(metrics).sort_values(
        by="F1", ascending=False
    ).reset_index(drop=True)

    feat_names = preprocess.get_feature_names_out().tolist()

    # -- 3. LƯU LẠI MÔ HÌNH --
    try:
        joblib.dump({
            "models": models,
            "metrics_df": metrics_df,
            "feat_names": feat_names
        }, model_file_path)
    except Exception as e:
        st.error(f"Không thể lưu mô hình: {e}")

    return models, metrics_df, preds, probs, feat_names


# ==============================================================================
# 2b. DATA MINING (CLUSTERING & ANOMALY DETECTION)
# ==============================================================================

def perform_data_mining(X_transformed, n_clusters=3):
    """
    Thực hiện phân cụm K-Means và phát hiện bất thường Isolation Forest.
    Input: X_transformed (Dữ liệu đã qua Preprocess - chuẩn hóa & onehot)
    """
    # 1. K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_transformed)

    # 2. PCA để giảm chiều về 2D (cho visual)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed)
    
    # 3. Anomaly Detection (Isolation Forest)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(X_transformed)
    # quy đổi: -1 là bất thường, 1 là bình thường -> đổi thành 1 (bất thường) và 0 (bình thường) cho dễ hiểu
    anomalies = np.where(anomalies == -1, 1, 0)
    
    return clusters, X_pca, anomalies
