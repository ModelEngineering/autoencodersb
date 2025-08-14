import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.datasets import make_classification, load_digits, load_wine, load_breast_cancer # type: ignore # type:ignore
from sklearn.decomposition import PCA # type:ignore
from sklearn.preprocessing import StandardScaler # type:ignore
from sklearn.model_selection import train_test_split # type:ignore
from sklearn.ensemble import RandomForestClassifier # type:ignore
from sklearn.linear_model import LogisticRegression # type:ignore
from sklearn.metrics import accuracy_score, classification_report # type:ignore
from sklearn.neighbors import NearestNeighbors # type:ignore
from sklearn.metrics.pairwise import euclidean_distances # type:ignore
from scipy.stats import spearmanr # type:ignore
import warnings
warnings.filterwarnings('ignore')
import example.pca_analyzer as pca_analyzer

if __name__ == "__main__":
    # Run demonstration
    pca_analyzer.demo_pca_evaluation()
    
    print("\n" + "="*60)
    print("PCA EVALUATION COMPLETE")
    print("="*60)
    print("The PCAQualityEvaluator class provides:")
    print("1. Variance explained analysis")
    print("2. Reconstruction error metrics")
    print("3. Distance preservation evaluation")
    print("4. Neighborhood preservation assessment")
    print("5. Classification performance comparison")
    print("6. Comprehensive visualization")
    print("7. Detailed text reports")