# app.py - ML Dashboard (5 Mô hình) – BẢN TỐI ƯU TỐC ĐỘ (MODULARIZED)

import streamlit as st
import pandas as pd
import modules.data as data
import modules.model as model
import modules.visualize as visualize
from modules.clustering import ClusteringModule

# ==============================================================================
# 5. MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(page_title="Credit Data Analysis - Group 2", layout="wide")
    
    # Premium CSS for Modern Look
    st.markdown("""
        <style>
        /* Toàn bộ nền ứng dụng */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
        }
        
        /* Cấu hình lại Block Container */
        .block-container { 
            padding-top: 3rem; 
            max-width: 1200px;
        }

        /* Tiêu đề chính */
        h1 {
            background: linear-gradient(to right, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.8rem !important;
            letter-spacing: -0.05em;
            margin-bottom: 2rem;
            text-align: center;
        }

        /* Card cho Metrics */
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 25px !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(96, 165, 248, 0.5);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            gap: 12px;
            padding: 10px 0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(51, 65, 85, 0.4);
            border-radius: 8px 8px 8px 8px;
            color: #94a3b8;
            padding: 12px 24px;
            font-weight: 600;
            border: 1px solid transparent;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6 !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Thành phần nội dung */
        .stMarkdown div p {
            color: #cbd5e1;
            font-size: 1.05rem;
        }
        
        div[data-testid="stExpander"] {
            background-color: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>Hệ Thống Phân Tích Tín Dụng</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lĩnh vực", "Khai thác dữ liệu", "Phân cụm")
    with col2:
        st.metric("Thực hiện", "Nhóm 2", "Tiểu luận")
    with col3:
        st.metric("Phiên bản", "2026", "Final")

    st.write("---")

    # 1. Tải & làm sạch dữ liệu
    with st.sidebar.expander("Cấu hình Dữ liệu & Mô hình", expanded=False):
        uploaded = st.file_uploader("Tải lên CSV (tuỳ chọn)", type=["csv"])
    
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                df = data.clean_column_names(df)
                st.sidebar.caption(f"Dữ liệu đã được tải từ **{uploaded.name}**")
            except Exception as e:
                st.error(f"Lỗi: {e}")
                df = data.load_and_create_data()
        else:
            df = data.load_and_create_data()

    # 2. Tiền xử lý
    df_used, X, y, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, preprocess = data.process_data(df)

    # Hiển thị thống kê gọn trong sidebar
    st.sidebar.info(
        f"**Tổng hồ sơ:** {len(df):,}\n"
        f"**Huấn luyện:** {len(df_used):,}\n"
        f"**Tỷ lệ Rủi ro:** {int(y.sum())} tốt / {int(len(y) - y.sum())} xấu"
    )
    # Detect XGB availability for passing to function
    USE_XGB = True
    try:
        from xgboost import XGBClassifier
    except ImportError:
        USE_XGB = False

    # 3. Training
    with st.spinner("Đang huấn luyện mô hình dự đoán nền tảng..."):
        models, metrics_df, preds, probs, feat_names = model.train_and_evaluate_models(
            X_train, y_train, X_test, y_test, preprocess, USE_XGB
        )

    # 4. Data Mining Prep
    preprocess.fit(X)
    X_transformed = preprocess.transform(X)

    # 5. TABS
    tab_eda, tab_mining, tab_dashboard, tab_predict = st.tabs([
        "Khám phá dữ liệu",
        "Phân cụm dữ liệu", 
        "Đánh giá mô hình", 
        "Hệ thống dự đoán"
    ])

    with tab_eda:
        visualize.display_eda_dashboard(df)

    with tab_mining:
        st.sidebar.markdown("---")
        st.sidebar.header("Cấu hình Phân Cụm")
        algo_choice = st.sidebar.selectbox(
            "Chọn thuật toán phân cụm",
            ["K-Means", "DBSCAN", "Hierarchical", "GMM", "Mean Shift"]
        )
        k_val = st.sidebar.slider("Số lượng cụm (K)", 2, 8, 3)
        
        # Tiền xử lý dữ liệu cho phân cụm (có thể dùng data gốc để ClusteringModule tự xử lý hoặc X_transformed)
        cm = ClusteringModule(df_used) # ClusteringModule sẽ tự lấy cột số và scale
        
        # Chạy thuật toán được chọn
        if algo_choice == "K-Means":
            clusters = cm.run_kmeans(n_clusters=k_val)
        elif algo_choice == "DBSCAN":
            eps = st.sidebar.slider("DBSCAN Eps", 0.1, 2.0, 0.5)
            min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 10, 5)
            clusters = cm.run_dbscan(eps=eps, min_samples=min_samples)
        elif algo_choice == "Hierarchical":
            clusters = cm.run_hierarchical(n_clusters=k_val)
        elif algo_choice == "GMM":
            clusters = cm.run_gmm(n_components=k_val)
        elif algo_choice == "Mean Shift":
            clusters = cm.run_meanshift()
            
        # Lấy kết quả đánh giá
        summary = cm.get_summary()
        current_summary = summary.get(algo_choice, {})
        
        # Lấy thông tin Visual (PCA) và Anomalies từ model module để đồng bộ
        # Chúng ta vẫn dùng Isolation Forest cho anomalies vì nó độc lập với clustering
        _, X_pca, anomalies = model.perform_data_mining(X_transformed, n_clusters=k_val)
        
        # Hiển thị Dashboard Mining đã nâng cấp
        visualize.display_mining_dashboard(
            X, clusters, X_pca, anomalies, k_val, df_used, 
            algorithm_name=algo_choice, 
            silhouette_score=current_summary.get('silhouette_score', -1)
        )
        
        # Thêm bảng so sánh tất cả thuật toán
        if st.checkbox("So sánh hiệu năng tất cả thuật toán (Silhouette Score)"):
            with st.spinner("Đang tính toán so sánh..."):
                # Đảm bảo tất cả đã chạy để lấy score
                cm.run_kmeans(3); cm.run_dbscan(); cm.run_hierarchical(3); cm.run_gmm(3); cm.run_meanshift()
                all_summary = cm.get_summary()
                compare_df = pd.DataFrame(all_summary).T.reset_index().rename(columns={'index': 'Thuật toán'})
                st.write("### 🏆 Bảng so sánh các thuật toán")
                st.dataframe(compare_df.style.highlight_max(axis=0, subset=['silhouette_score'], color='lightgreen'))

    with tab_dashboard:
        visualize.display_dashboard(models, metrics_df, preds, probs, X_test, y_test, feat_names)

    with tab_predict:
        visualize.display_prediction_form(X, models, numeric_cols, categorical_cols, df_used)


if __name__ == "__main__":
    main()
