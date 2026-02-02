import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ==============================================================================
# 2c. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def display_eda_dashboard(df: pd.DataFrame):
    """
    Hiển thị các biểu đồ phân tích dữ liệu (EDA).
    """
    st.markdown("## Khám phá dữ liệu (EDA)")
    
    # 1. Tỷ lệ Risk
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Tỷ lệ Rủi ro (Good vs Bad)")
        risk_counts = df['Risk'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
        ax1.axis('equal')
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.markdown("### Thống kê cơ bản")
        st.dataframe(df.describe(), use_container_width=True)

    # 2. Phân phối biến số
    st.markdown("### Phân phối các biến quan trọng theo Risk")
    features_to_plot = ['Age', 'Credit_Amount', 'Duration']
    cols = st.columns(3)
    
    for i, feature in enumerate(features_to_plot):
        if feature in df.columns:
            with cols[i]:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.histplot(data=df, x=feature, hue="Risk", kde=True, element="step", palette=['#ff9999', '#66b3ff'], ax=ax)
                ax.set_title(f"Phân phối {feature}")
                st.pyplot(fig)
                plt.close(fig)

    # 3. Biến phân loại
    st.markdown("### Phân tích biến phân loại")
    cat_cols_plot = ['Housing', 'Purpose', 'Job']
    cols_cat = st.columns(3)
    for i, col in enumerate(cat_cols_plot):
        if col in df.columns:
            with cols_cat[i]:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.countplot(data=df, x=col, hue="Risk", palette="Set2", ax=ax)
                ax.set_title(f"Risk theo {col}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)
    
    # 4. Correlation
    if st.checkbox("Hiển thị Ma trận Tương quan (Correlation Matrix)"):
        st.markdown("### Ma trận Tương quan")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)
            plt.close(fig_corr)


# ==============================================================================
# 3. DASHBOARD VISUALIZATION
# ==============================================================================

def display_dashboard(models, metrics_df, preds, probs, X_test, y_test, feat_names):
    st.subheader("Bảng so sánh mô hình (Đánh giá trên tập Test)")
    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        })
    )

    # Ma trận nhầm lẫn
    st.markdown("### Ma trận nhầm lẫn")
    model_names = list(models.keys())
    cols_cm = st.columns(3)

    for i, name in enumerate(model_names):
        with cols_cm[i % 3]:
            y_pred = preds[name]
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Bad (0)', 'Good (1)'],
                yticklabels=['Bad (0)', 'Good (1)'],
                ax=ax
            )
            ax.set_title(f"CM — {name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

    # Đường cong ROC
    st.markdown("### Đường cong ROC")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        if p is not None:
            fpr, tpr, _ = roc_curve(y_test, p)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color='gray')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Đường cong ROC")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

    # Độ quan trọng đặc trưng
    st.markdown("### Độ quan trọng đặc trưng (Top 20)")

    if feat_names:
        cols_imp = st.columns(2)
        imp_counter = 0
        for name, pipe in models.items():
            clf = pipe.named_steps.get("clf")
            importances = None

            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                coef = clf.coef_
                importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

            if importances is not None:
                imp_df = pd.DataFrame({
                    "feature": feat_names,
                    "importance": importances
                })
                imp_df["feature"] = imp_df["feature"].apply(
                    lambda x: x.split('__')[-1].replace('_', ' ').title()
                )
                imp_df = imp_df.sort_values(
                    "importance", ascending=False
                ).head(20)

                with cols_imp[imp_counter % 2]:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.barplot(
                        x='importance', y='feature',
                        data=imp_df, palette='viridis', ax=ax2
                    )
                    ax2.set_title(f"Feature Importance — {name}")
                    ax2.set_xlabel("Độ quan trọng / |hệ số|")
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

                imp_counter += 1


# ==============================================================================
# 4. PREDICTION FORM
# ==============================================================================

def display_prediction_form(X, models, numeric_cols, categorical_cols, df_raw):
    st.subheader("Nhập dữ liệu để dự đoán")

    cols_input = st.columns(3)
    input_data = {}
    col_index = 0

    for col in X.columns:
        current_col = cols_input[col_index % 3]
        label = col.replace('_', ' ')

        if col in numeric_cols:
            default_val = float(X[col].median())
            input_data[col] = current_col.number_input(
                label, value=default_val, step=1.0
            )
        else:
            if col in df_raw.columns:
                choices = sorted(df_raw[col].dropna().astype(str).unique().tolist())
            else:
                choices = sorted(X[col].dropna().astype(str).unique().tolist())

            if choices:
                input_data[col] = current_col.selectbox(label, options=choices, index=0)
            else:
                input_data[col] = current_col.text_input(label, value="")

        col_index += 1

    st.markdown("---")
    model_choice = st.selectbox(
        "Chọn mô hình để dự đoán",
        options=list(models.keys()),
        index=1 if "RandomForest" in models else 0
    )

    if st.button("Dự đoán", help="Nhấn để thực hiện dự đoán với mô hình đã chọn."):
        input_for_model = {k: v for k, v in input_data.items() if k in X.columns}
        row_df = pd.DataFrame([input_for_model])

        pred = models[model_choice].predict(row_df)[0]
        proba = None
        try:
            proba = models[model_choice].predict_proba(row_df)[0, 1]
        except Exception:
            pass

        label = "✅ TỐT / Được duyệt (1)" if pred == 1 else "❌ XẤU / Không duyệt (0)"

        if pred == 1:
            st.success(
                f"Kết quả dự đoán: **{label}**"
                + (f" — Xác suất TỐT: **{proba:.3f}**" if proba is not None else "")
            )
        else:
            st.error(
                f"Kết quả dự đoán: **{label}**"
                + (f" — Xác suất TỐT: **{proba:.3f}**" if proba is not None else "")
            )
            
            # --- TÍNH NĂNG GIẢI THÍCH (EXPLAINABILITY) ---
            st.markdown("#### 💡 Tại sao hồ sơ này bị đánh giá Rủi Ro?")
            
            # So sánh với trung bình khách hàng tốt
            good_customers = df_raw[df_raw['Risk'] == 1]
            if not good_customers.empty:
                comparison = []
                # So sánh các biến số
                for col in numeric_cols:
                    val = input_data.get(col, 0)
                    avg_good = good_customers[col].mean()
                    
                    # Nếu giá trị tệ hơn mức trung bình đáng kể
                    if col == 'Credit_Amount' and val > avg_good * 1.2:
                        comparison.append(f"- **Số tiền vay ({val:,.0f})** cao hơn trung bình (Good: {avg_good:,.0f}).")
                    elif col == 'Duration' and val > avg_good * 1.2:
                        comparison.append(f"- **Kỳ hạn vay ({val:.0f} tháng)** dài hơn trung bình (Good: {avg_good:.0f} tháng).")
                    elif col == 'Age' and val < avg_good * 0.8:
                        comparison.append(f"- **Tuổi ({val:.0f})** trẻ hơn mức trung bình của nhóm uy tín ({avg_good:.0f}).")
                
                # So sánh biến phân loại
                for col in categorical_cols:
                    val = input_data.get(col, '')
                    # Tính tỷ lệ bad của nhóm này
                    subset = df_raw[df_raw[col] == val]
                    if not subset.empty:
                        bad_rate = (subset['Risk'] == 0).mean()
                        avg_bad_rate = (df_raw['Risk'] == 0).mean()
                        if bad_rate > avg_bad_rate * 1.3: 
                            comparison.append(f"- Nhóm **{col} = {val}** thường có tỷ lệ nợ xấu cao ({bad_rate*100:.1f}%).")

                if comparison:
                    for line in comparison:
                        st.write(line)
                    
                    # Thêm phân tích nhóm (Group analysis)
                    st.info("💡 **Gợi ý:** Hồ sơ này có nhiều đặc điểm tương đồng với nhóm khách hàng có tỷ lệ rủi ro cao. Bạn nên kiểm tra kỹ lịch sử tín dụng hoặc yêu cầu thêm tài sản đảm bảo.")
                else:
                    st.write("Hồ sơ có tổng điểm tín dụng thấp dựa trên sự kết hợp của nhiều yếu tố.")


def display_mining_dashboard(X, clusters, X_pca, anomalies, n_clusters, df_original, algorithm_name="K-Means", silhouette_score=-1):
    st.markdown(f"## Kết quả phân cụm: {algorithm_name}")
    
    # Hiển thị Silhouette Score nếu có
    if silhouette_score > 0:
        st.metric("Chỉ số Silhouette (Độ tách biệt cụm)", f"{silhouette_score:.3f}")
        if silhouette_score > 0.5:
            st.success("Cụm được phân tách rất tốt.")
        elif silhouette_score > 0.2:
            st.info("Cụm có sự phân tách khá.")
        else:
            st.warning("Cụm có sự chồng lấn cao. Nên cân nhắc đổi tham số hoặc thuật toán.")

    # --- A. CLUSTER PROFILING ---
    st.markdown("### Đặc điểm từng nhóm (Cluster Profiling)")
    st.info("Bảng dưới đây hiển thị đặc điểm trung bình của từng nhóm khách hàng sau khi phân cụm.")
    
    # Gán nhãn tạm vào df gốc để tính toán
    df_temp = df_original.copy()
    df_temp['Cluster'] = clusters
    
    # Group by và tính mean các cột số
    profile = df_temp.groupby('Cluster')[['Age', 'Credit_Amount', 'Duration']].mean().reset_index()
    # Tính số lượng và tỷ lệ Risk cho từng cụm
    cluster_counts = df_temp.groupby('Cluster').size().reset_index(name='Số Lượng KH')
    
    # Phân tích Business - Cluster vs Risk
    risk_by_cluster = df_temp.groupby(['Cluster', 'Risk']).size().unstack(fill_value=0)
    # Map Risk nếu cần (nếu Risk là 0/1 thì 1=Good, 0=Bad)
    if 1 in risk_by_cluster.columns and 0 in risk_by_cluster.columns:
        risk_by_cluster['Tỷ lệ Bad (%)'] = (risk_by_cluster[0] / (risk_by_cluster[0] + risk_by_cluster[1]) * 100).round(1)
    
    profile = profile.merge(cluster_counts, on='Cluster')
    if not risk_by_cluster.empty:
        profile = profile.merge(risk_by_cluster[['Tỷ lệ Bad (%)']], on='Cluster', how='left')
    
    # Đổi tên cột cho đẹp
    profile = profile.rename(columns={
        'Age': 'Tuổi TB', 
        'Credit_Amount': 'Tín Dụng TB', 
        'Duration': 'Kỳ Hạn TB'
    })
    
    st.dataframe(profile.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # --- B. VISUALIZATION ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Biểu đồ phân bố tập trung")
        # Tạo DF cho Plotly
        df_vis = pd.DataFrame(X_pca[:, :2], columns=['PCA1', 'PCA2'])
        df_vis['Cluster'] = clusters.astype(str)
        df_vis['Anomaly'] = anomalies.astype(str)
        df_vis['Risk'] = df_original['Risk'].astype(str)
        df_vis['Credit Amount'] = df_original['Credit_Amount']
        
        fig = px.scatter(
            df_vis, x='PCA1', y='PCA2', 
            color='Cluster', symbol='Anomaly',
            hover_data=['Risk', 'Credit Amount'],
            title=f"Bản Đồ Phân Cụm ({algorithm_name})",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_dark"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Tỷ lệ rủi ro theo cụm")
        if not risk_by_cluster.empty:
            # Vẽ biểu đồ cột chồng tỷ lệ Risk
            fig_risk = px.bar(
                risk_by_cluster.drop(columns=['Tỷ lệ Bad (%)'], errors='ignore'),
                barmode='stack',
                title="Số lượng Good/Bad theo cụm",
                labels={'value': 'Số lượng', 'variable': 'Risk'},
                template="plotly_dark"
            )
            fig_risk.update_layout(height=450, showlegend=True)
            st.plotly_chart(fig_risk, use_container_width=True)
    
    # --- C. ANOMALIES ---
    st.markdown("### Phát hiện bất thường (Anomaly Detection)")
    n_anom = np.sum(anomalies)
    if n_anom > 0:
        st.warning(f"Hệ thống phát hiện **{n_anom}** hồ sơ có dấu hiệu bất thường (được đánh dấu trên biểu đồ).")
        with st.expander("Khám phá danh sách hồ sơ bất thường"):
            st.write("Các hồ sơ này có đặc điểm rất khác biệt so với phần còn lại của tập dữ liệu.")
            st.dataframe(df_original[anomalies == 1].head(20))
    else:
        st.success("Không phát hiện điểm bất thường đáng kể nào.")
