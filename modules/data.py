import pandas as pd
import numpy as np
import re
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ==============================================================================
# 1. DATA LOADING & CLEANING
# ==============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch tên cột: bỏ khoảng trắng, ký tự đặc biệt, chuẩn hoá tên phổ biến."""
    col_mapping = {}
    for col in df.columns:
        # Standardize known columns from credit_risk_dataset.csv
        clean_col = re.sub(r'[^A-Za-z0-9_]+', ' ', col.strip()).strip().replace(' ', '_')
        
        col_lower = clean_col.lower()
        if 'person_age' in col_lower:
            clean_col = 'Age'
        elif 'loan_amnt' in col_lower:
            clean_col = 'Credit_Amount'
        elif 'person_home_ownership' in col_lower:
            clean_col = 'Housing'
        elif 'loan_intent' in col_lower:
            clean_col = 'Purpose'
        elif 'loan_status' in col_lower:
            clean_col = 'Risk'
        elif 'person_income' in col_lower:
            clean_col = 'Income'
        elif 'person_emp_length' in col_lower:
            clean_col = 'Job_Tenure'
        elif 'loan_int_rate' in col_lower:
            clean_col = 'Interest_Rate'
            
        # Old German data mapping (for backward compatibility)
        if 'Credit amount' in col or 'Credit_Amount' in col:
            clean_col = 'Credit_Amount'
        elif 'Saving accounts' in col or 'Saving_Accounts' in col:
            clean_col = 'Saving_Accounts'
        elif 'Checking account' in col or 'Checking_Account' in col:
            clean_col = 'Checking_Account'
            
        col_mapping[col] = clean_col

    df = df.rename(columns=col_mapping)
    return df


@st.cache_data
def load_and_create_data() -> pd.DataFrame:
    """
    Đọc dữ liệu từ credit_risk_dataset.csv.
    Nếu không tìm thấy file thì thảo luận tạo dữ liệu mẫu.
    """
    # Prefer the newly uploaded Kaggle dataset
    possible_paths = [
        "credit_risk_dataset.csv",
        os.path.join("mayhoc", "synthetic_credit_approval_data.csv"),
        "synthetic_credit_approval_data.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = clean_column_names(df)
            
            # Special logic for the new Kaggle dataset mapping Risk 0/1
            # In Kaggle dataset: 1 = default (Bad), 0 = non-default (Good)
            # We want: 1 = Good, 0 = Bad
            if 'Risk' in df.columns and set(df['Risk'].unique()).issubset({0, 1}):
                df['Risk'] = df['Risk'].map({0: 1, 1: 0})
            
            return df

    # Fallback: tạo dữ liệu mẫu CÓ CẤU TRÚC
    n_samples = 50000
    np.random.seed(42)
    
    # Tạo 3 nhóm nòng cốt để simulation có cụm rõ ràng
    # Nhóm 1: Người già, Giàu, An toàn (30%)
    n1 = int(n_samples * 0.3)
    d1 = {
        'Age': np.random.randint(45, 75, n1),
        'Sex': np.random.choice(['male', 'female'], n1),
        'Job': np.random.choice([2, 3], n1), # skilled or high qual
        'Housing': np.random.choice(['own'], n1),
        'Saving accounts': np.random.choice(['rich', 'quite rich'], n1),
        'Checking account': np.random.choice(['rich', 'moderate'], n1),
        'Credit amount': np.random.randint(5000, 20000, n1),
        'Duration': np.random.randint(12, 48, n1),
        'Purpose': np.random.choice(['car', 'business', 'repairs'], n1),
    }
    
    # Nhóm 2: Người trẻ, Ít tiền, Rủi ro cao (40%)
    n2 = int(n_samples * 0.4)
    d2 = {
        'Age': np.random.randint(18, 30, n2),
        'Sex': np.random.choice(['male', 'female'], n2),
        'Job': np.random.choice([0, 1], n2), # unskill
        'Housing': np.random.choice(['rent', 'free'], n2),
        'Saving accounts': np.random.choice(['little', 'moderate'], n2),
        'Checking account': np.random.choice(['little', 'moderate'], n2),
        'Credit amount': np.random.randint(500, 5000, n2),
        'Duration': np.random.randint(6, 24, n2),
        'Purpose': np.random.choice(['radio/TV', 'education', 'domestic appliances'], n2),
    }

    # Nhóm 3: Trung lưu, Bình thường (30%)
    n3 = n_samples - n1 - n2
    d3 = {
        'Age': np.random.randint(30, 50, n3),
        'Sex': np.random.choice(['male', 'female'], n3),
        'Job': np.random.choice([1, 2], n3),
        'Housing': np.random.choice(['own', 'rent'], n3),
        'Saving accounts': np.random.choice(['little', 'moderate', 'quite rich'], n3),
        'Checking account': np.random.choice(['little', 'moderate', 'rich'], n3),
        'Credit amount': np.random.randint(2000, 10000, n3),
        'Duration': np.random.randint(12, 36, n3),
        'Purpose': np.random.choice(['furniture/equipment', 'car', 'education'], n3),
    }

    # Gom lại
    data = {}
    for k in d1.keys():
        data[k] = np.concatenate([d1[k], d2[k], d3[k]])
        
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    score = (df['Credit amount'] / 15000) * 0.4 + \
            (df['Duration'] / 72) * 0.3 + \
            (1 - df['Age'] / 70) * 0.15 + \
            np.random.rand(n_samples) * 0.15

    df['Risk'] = (score < 0.4).map({True: 'bad', False: 'good'})
    df = clean_column_names(df)
    return df


def process_data(df: pd.DataFrame):
    """
    Tiền xử lý dữ liệu, giới hạn số dòng, chia train/test và tạo ColumnTransformer.
    df ĐÃ được clean_column_names trước khi truyền vào.
    """
    df = df.copy()

    # ===== Giới hạn số dòng để train cho nhanh =====
    # Ẩn vào expander để đỡ rối
    with st.sidebar.expander("⚙️ Tham số huấn luyện (Nâng cao)", expanded=False):
        max_rows = st.number_input(
            "Số dòng tối đa dùng để huấn luyện",
            min_value=2000, max_value=min(100000, len(df)),
            value=min(15000, len(df)), step=1000
        )
        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=42).reset_index(drop=True)
            st.caption(f"ℹ️ Đang lấy mẫu {max_rows} dòng để tối ưu tốc độ. Bạn xem đầy đủ dữ liệu tại tab 'Khám phá dữ liệu'.")
            
        test_size = st.slider("Tỷ lệ test (%)", 10, 40, value=20, step=5) / 100.0
    if 'Risk' in df.columns:
        target_col = 'Risk'
    else:
        target_col = st.selectbox(
            "Chọn cột mục tiêu (target):",
            options=list(df.columns)
        )

    # 2. Mapping target (ưu tiên good=1, bad=0)
    y_raw = df[target_col].astype(str).str.lower().str.strip().fillna('')
    unique_values = y_raw.unique()

    class_1_val = next((v for v in unique_values if 'good' in v or v == '1'), None)
    class_0_val = next((v for v in unique_values if 'bad' in v or v == '0'), None)

    if class_1_val is None or class_0_val is None:
        if len(unique_values) >= 2:
            counts = y_raw.value_counts().nlargest(2)
            class_1_val = counts.index[0]
            class_0_val = counts.index[1]
        else:
            st.error(
                f"❌ Cột mục tiêu '{target_col}' chỉ có một giá trị sau khi xử lý. "
                f"Vui lòng dùng dữ liệu có đủ 2 lớp."
            )
            st.stop()

    y = y_raw.apply(lambda x: 1 if x == class_1_val else 0)

    if y.sum() < len(y) * 0.01:
        st.error(
            f"❌ Dữ liệu target '{target_col}' quá mất cân bằng (lớp 1 < 1%). "
            f"Có thể bạn đang dùng file kết quả dự đoán thay vì dữ liệu gốc."
        )
        st.stop()

    # 3. Tách X
    X = df.drop(columns=[target_col], errors='ignore')

    # 4. Xử lý missing & phân loại cột
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        X[col] = X[col].fillna(X[col].mode()[0])

    # 5. Chia train/test
    # test_size đã được define ở trên
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # 6. ColumnTransformer
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return df, X, y, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, preprocess
