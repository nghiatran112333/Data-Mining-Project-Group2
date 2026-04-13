import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class ClusteringModule:
    """
    Module cung cấp 5 thuật toán phân cụm phổ biến để phân khúc dữ liệu.
    """
    def __init__(self, data):
        """
        Khởi tạo với dữ liệu đầu vào (DataFrame hoặc ndarray).
        """
        self.data = data
        self.scaled_data = None
        self.models = {}
        self.labels = {}
        
        # Tiền xử lý dữ liệu (Scaling)
        self._preprocess()

    def _preprocess(self):
        """
        Chuẩn hóa dữ liệu trước khi phân cụm.
        """
        scaler = StandardScaler()
        if isinstance(self.data, pd.DataFrame):
            # Chỉ lấy các cột số để phân cụm
            numeric_data = self.data.select_dtypes(include=[np.number])
            # Điền giá trị thiếu bằng median trước khi scale
            numeric_data = numeric_data.fillna(numeric_data.median())
            self.scaled_data = scaler.fit_transform(numeric_data)
        else:
            data = np.where(np.isnan(self.data), np.nanmedian(self.data, axis=0), self.data)
            self.scaled_data = scaler.fit_transform(data)

    def run_kmeans(self, n_clusters=3):
        """
        1. K-Means Clustering: Phân cụm dựa trên trọng tâm.
        """
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.labels['K-Means'] = model.fit_predict(self.scaled_data)
        self.models['K-Means'] = model
        return self.labels['K-Means']

    def run_dbscan(self, eps=0.5, min_samples=5):
        """
        2. DBSCAN: Phân cụm dựa trên mật độ. 
        Tự động phát hiện nhiễu (noise) và không cần biết trước số cụm.
        """
        model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels['DBSCAN'] = model.fit_predict(self.scaled_data)
        self.models['DBSCAN'] = model
        return self.labels['DBSCAN']

    def run_hierarchical(self, n_clusters=3):
        """
        3. Agglomerative Hierarchical Clustering: Phân cụm phân cấp.
        Xây dựng cây phân cụm (dendrogram).
        """
        model = AgglomerativeClustering(n_clusters=n_clusters)
        self.labels['Hierarchical'] = model.fit_predict(self.scaled_data)
        self.models['Hierarchical'] = model
        return self.labels['Hierarchical']

    def get_summary(self):
        """
        Trả về tóm tắt kết quả phân cụm và điểm đánh giá.
        """
        summary = {}
        for name, label in self.labels.items():
            # Không tính noise (-1) trong DBSCAN cho số lượng cụm
            unique_labels = np.unique(label)
            n_clusters = len(unique_labels[unique_labels != -1])
            
            # Tính Silhouette Score (nếu có ít nhất 2 cụm và không phải tất cả là noise)
            score = -1
            if 1 < n_clusters < len(self.scaled_data):
                # Sample dữ liệu nếu quá lớn để tính cho nhanh
                if len(self.scaled_data) > 2000:
                    indices = np.random.choice(len(self.scaled_data), 2000, replace=False)
                    score = silhouette_score(self.scaled_data[indices], label[indices])
                else:
                    score = silhouette_score(self.scaled_data, label)
            
            summary[name] = {
                "n_clusters": n_clusters,
                "noise_points": np.sum(label == -1) if name == 'DBSCAN' else 0,
                "silhouette_score": score
            }
        return summary

    def visualize_clusters(self, method_name, n_components=2):
        """
        Trực quan hóa kết quả phân cụm bằng PCA.
        """
        if method_name not in self.labels:
            print(f"Chưa chạy thuật toán {method_name}")
            return None

        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(self.scaled_data)
        
        df_viz = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(n_components)])
        df_viz['Cluster'] = self.labels[method_name]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_viz, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.7)
        plt.title(f"Phân cụm bằng {method_name} (PCA Projection)")
        return plt
