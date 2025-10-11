# tests/test_data.py
"""
Data validation tests for CI/CD pipeline
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

class TestDataQuality:
    """Test data quality and schema validation"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data paths"""
        self.raw_data_path = Path("data/raw")
        self.processed_data_path = Path("data/processed")
        self.features_data_path = Path("data/features")
        self.feature_file = Path("data/features/train_FD001_features.csv")
    
    def test_data_directories_exist(self):
        """Test that required data directories exist"""
        assert self.raw_data_path.exists(), "Raw data directory missing"
        assert self.processed_data_path.exists(), "Processed data directory missing"
        assert self.features_data_path.exists(), "Features data directory missing"
    
    def test_feature_file_exists(self):
        """Test that feature file exists and is readable"""
        assert self.feature_file.exists(), f"Feature file not found: {self.feature_file}"
        
        try:
            df = pd.read_csv(self.feature_file)
            assert len(df) > 0, "Feature file is empty"
        except Exception as e:
            pytest.fail(f"Cannot read feature file: {str(e)}")
    
    def test_data_schema(self):
        """Test that data has expected schema"""
        if not self.feature_file.exists():
            pytest.skip("Feature file not found")
        
        df = pd.read_csv(self.feature_file)
        
        # Test required columns
        required_columns = ['RUL']  # Target column
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing"
        
        # Test data types
        assert pd.api.types.is_numeric_dtype(df['RUL']), "RUL should be numeric"
        
        # Test that we have sensor columns
        numeric_columns = [col for col in df.columns if str(col).isdigit()]
        metadata_columns = ['engine_id', 'cycle', 'max_cycle', 'RUL']

        assert len(numeric_columns) > 0, f"No numeric sensor columns found. Found columns: {df.columns.tolist()}"
        
        # Test that all feature columns are numeric
        feature_cols = [col for col in df.columns if col != 'RUL']
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"Feature column '{col}' should be numeric"
    
    def test_data_quality(self):
        """Test data quality issues"""
        if not self.feature_file.exists():
            pytest.skip("Feature file not found")
        
        df = pd.read_csv(self.feature_file)
        
        # Test for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
        
        assert missing_percentage < 10, \
            f"Too many missing values: {missing_percentage:.2f}%"
        
        # Test for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        assert duplicate_percentage < 5, \
            f"Too many duplicate rows: {duplicate_percentage:.2f}%"
        
        # Test for constant columns
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        
        constant_percentage = len(constant_columns) / len(df.columns) * 100
        assert constant_percentage < 50, f"Too many constant columns ({constant_percentage:.1f}%): {constant_columns}"
        
        # Test target variable distribution
        rul_stats = df['RUL'].describe()
        assert rul_stats['min'] >= 0, "RUL should not be negative"
        assert rul_stats['max'] <= 1000, "RUL values seem unreasonably high"
        assert rul_stats['std'] > 0, "RUL has no variation"
    
    def test_data_ranges(self):
        """Test that data values are within expected ranges"""
        if not self.feature_file.exists():
            pytest.skip("Feature file not found")
        
        df = pd.read_csv(self.feature_file)
        
        # Test RUL ranges
        assert df['RUL'].min() >= 0, "RUL cannot be negative"
        assert df['RUL'].max() <= 1000, "RUL seems unreasonably high"
        
        # Test for extreme outliers using IQR method
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outliers as points beyond 3*IQR from Q1/Q3
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(df)) * 100
            
            assert outlier_percentage < 5, \
                f"Too many outliers in column '{col}': {outlier_percentage:.2f}%"
    
    def test_data_consistency(self):
        """Test data consistency across time"""
        if not self.feature_file.exists():
            pytest.skip("Feature file not found")
        
        df = pd.read_csv(self.feature_file)
        
        # Test that we have reasonable amount of data
        assert len(df) >= 1000, f"Dataset too small: {len(df)} rows"
        
        # Test feature correlation with target
        feature_cols = [col for col in df.columns if col != 'RUL']
        correlations = df[feature_cols].corrwith(df['RUL']).abs()
        
        # At least some features should be correlated with target
        high_corr_features = correlations[correlations > 0.1]
        assert len(high_corr_features) > 0, \
            "No features show correlation with target variable"
        
        # No feature should be perfectly correlated (potential data leakage)
        perfect_corr = correlations[correlations > 0.99]
        assert len(perfect_corr) == 0, \
            f"Features with perfect correlation found: {perfect_corr.index.tolist()}"

class TestDataVersioning:
    """Test data versioning and tracking"""
    
    def test_dvc_files_exist(self):
        """Test that DVC tracking files exist"""
        dvc_yaml = Path("dvc.yaml")
        assert dvc_yaml.exists(), "dvc.yaml file missing"
        
        dvc_lock = Path("dvc.lock")
        if dvc_lock.exists():
            # If dvc.lock exists, validate it's readable
            try:
                import yaml
                with open(dvc_lock, 'r') as f:
                    lock_content = yaml.safe_load(f)
                assert isinstance(lock_content, dict), "Invalid dvc.lock format"
            except Exception as e:
                pytest.fail(f"Cannot read dvc.lock: {str(e)}")
    
    def test_data_pipeline_stages(self):
        """Test that all data pipeline stages are defined"""
        dvc_yaml = Path("dvc.yaml")
        if not dvc_yaml.exists():
            pytest.skip("dvc.yaml not found")
        
        try:
            import yaml
            with open(dvc_yaml, 'r') as f:
                pipeline = yaml.safe_load(f)
            
            assert 'stages' in pipeline, "No stages defined in dvc.yaml"
            
            stages = pipeline['stages']
            expected_stages = ['preprocess', 'features']
            
            for stage in expected_stages:
                assert stage in stages, f"Missing pipeline stage: {stage}"
                
                # Check stage structure
                stage_config = stages[stage]
                assert 'cmd' in stage_config, f"Stage {stage} missing 'cmd'"
                assert 'deps' in stage_config, f"Stage {stage} missing 'deps'"
                assert 'outs' in stage_config, f"Stage {stage} missing 'outs'"
        
        except Exception as e:
            pytest.fail(f"Error validating dvc.yaml: {str(e)}")

class TestFeatureEngineering:
    """Test feature engineering quality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for feature tests"""
        self.feature_file = Path("data/features/train_FD001_features.csv")
    
    def test_feature_scaling(self):
        """Test that features are properly scaled"""
        if not self.feature_file.exists():
            pytest.skip("Feature file not found")
        
        df = pd.read_csv(self.feature_file)
        feature_cols = [col for col in df.columns if col != 'RUL']
        
        # Check if features seem to be on similar scales
        feature_stats = df[feature_cols].describe()
        
        # Calculate coefficient of variation for each feature
        cv_values = feature_stats.loc['std'] / feature_stats.loc['mean']
        cv_values = cv_values.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Most features should have reasonable coefficient of variation
        reasonable_cv = cv_values[(cv_values > 0.01) & (cv_values < 10)]
        cv_percentage = len(reasonable_cv) / len(cv_values) * 100
        
        assert cv_percentage > 70, \
            f"Only {cv_percentage:.1f}% of features have reasonable variation"
    
    def test_feature_correlation(self):
        """Test feature correlation matrix"""
        if not self.feature_file.exists():
            pytest.skip("Feature file not found")
        
        df = pd.read_csv(self.feature_file)
        feature_cols = [col for col in df.columns if col != 'RUL']
        
        if len(feature_cols) > 1:
            corr_matrix = df[feature_cols].corr()
            
            # Check for highly correlated features (potential redundancy)
            # Get upper triangle of correlation matrix
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find highly correlated pairs
            high_corr = upper_tri[(upper_tri > 0.95) | (upper_tri < -0.95)]
            high_corr_pairs = [(col, idx) for col in high_corr.columns 
                             for idx in high_corr.index if pd.notna(high_corr.loc[idx, col])]
            
            # Warning if too many highly correlated features
            if len(high_corr_pairs) > len(feature_cols) * 0.1:  # More than 10% of features
                pytest.fail(f"Too many highly correlated feature pairs: {len(high_corr_pairs)}")

if __name__ == "__main__":
    pytest.main([__file__])