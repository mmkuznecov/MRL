import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinearMetricsCalculator:
    """Class for computing and caching linear metrics of embedding matrices."""
    
    def __init__(self):
        """Initialize the calculator with empty cache."""
        self.svd_cache = {}
        
    def compute_metrics(self, embeddings: np.ndarray, cache_key: str = "default") -> Dict[str, float]:
        """
        Compute various linear metrics for embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, embedding_dim)
            cache_key: Identifier for cached SVD results
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Check if we have enough samples
        n_samples, n_dim = embeddings.shape
        logger.debug(f"Computing linear metrics for embeddings shape: {embeddings.shape}")
        
        # Compute or reuse SVD decomposition
        S, updated_cache = self._compute_or_reuse_svd(embeddings, cache_key)
        self.svd_cache.update(updated_cache)
        
        # Compute condition number (ratio of largest to smallest non-zero singular value)
        non_zero_S = S[S > 1e-10]
        if len(non_zero_S) > 0:
            metrics['condition_number'] = float(non_zero_S[0] / non_zero_S[-1])
        else:
            metrics['condition_number'] = float('inf')
        
        # Compute effective rank using entropy of normalized singular values
        if len(S) > 0 and np.sum(S) > 0:
            S_norm = S / np.sum(S)
            # Avoid log(0) by adding small epsilon
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
            metrics['effective_rank'] = float(np.exp(entropy))
        else:
            metrics['effective_rank'] = 0.0
        
        # Number of significant singular values (explaining 95% of variance)
        if len(S) > 0:
            total_variance = np.sum(S**2)
            if total_variance > 0:
                cumulative_variance = np.cumsum(S**2) / total_variance
                significant_dims = np.sum(cumulative_variance < 0.95) + 1
                metrics['significant_dims_95pct'] = int(significant_dims)
            else:
                metrics['significant_dims_95pct'] = 0
        else:
            metrics['significant_dims_95pct'] = 0
        
        # Explained variance ratios for top 5, 10, 50 dimensions
        if len(S) > 0:
            total_variance = np.sum(S**2)
            if total_variance > 0:
                for k in [5, 10, 50]:
                    if k < len(S):
                        metrics[f'explained_variance_top{k}'] = float(np.sum(S[:k]**2) / total_variance)
                    else:
                        metrics[f'explained_variance_top{k}'] = 1.0
            else:
                for k in [5, 10, 50]:
                    metrics[f'explained_variance_top{k}'] = 0.0
        else:
            for k in [5, 10, 50]:
                metrics[f'explained_variance_top{k}'] = 0.0
        
        # Compute "soft" rank - number of singular values that are at least 1% of the largest
        if len(S) > 0 and S[0] > 0:
            soft_rank = np.sum(S >= 0.01 * S[0])
            metrics['soft_rank'] = int(soft_rank)
        else:
            metrics['soft_rank'] = 0
        
        # Compute average correlation between dimensions
        try:
            # Calculate correlation matrix and take average of absolute off-diagonal elements
            corr_matrix = np.corrcoef(embeddings, rowvar=False)
            if not np.all(np.isnan(corr_matrix)):
                # Create a mask for the diagonal
                mask = np.ones(corr_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                # Calculate average absolute correlation
                avg_corr = np.nanmean(np.abs(corr_matrix[mask]))
                metrics['avg_correlation'] = float(avg_corr)
            else:
                metrics['avg_correlation'] = 0.0
        except Exception as e:
            logger.warning(f"Error computing correlation: {str(e)}")
            metrics['avg_correlation'] = 0.0
        
        return metrics
        
    def _compute_or_reuse_svd(self, embeddings: np.ndarray, cache_key: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute SVD if needed or reuse cached results.
        
        Args:
            embeddings: Embedding matrix
            cache_key: Identifier for cache entry
            
        Returns:
            Tuple of (singular_values, updated_cache_dict)
        """
        # Compute a simple hash of the embeddings to check if they've changed
        current_hash = hash(str(embeddings.shape) + str(np.sum(embeddings)) + str(np.sum(embeddings**2)))
        
        # Check if we can reuse cached SVD
        if cache_key in self.svd_cache and self.svd_cache[cache_key].get('embeddings_hash') == current_hash:
            # Use cached SVD results
            logger.debug(f"Using cached SVD decomposition for {cache_key}")
            S = self.svd_cache[cache_key]['S']
            return S, {}
        
        # Compute new SVD
        try:
            logger.debug(f"Computing new SVD decomposition for {cache_key}")
            # Compute SVD
            U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
            # Create new cache entry
            new_cache = {cache_key: {'U': U, 'S': S, 'Vt': Vt, 'embeddings_hash': current_hash}}
            return S, new_cache
        except np.linalg.LinAlgError as e:
            logger.warning(f"SVD computation failed: {str(e)}")
            # Handle SVD computation failure by creating placeholder values
            n_samples, n_dim = embeddings.shape
            S = np.ones(min(n_samples, n_dim))
            new_cache = {cache_key: {'U': None, 'S': S, 'Vt': None, 'embeddings_hash': None}}
            return S, new_cache
    
    def clear_cache(self):
        """Clear the SVD cache."""
        self.svd_cache.clear()
        
    def get_all_metrics_names(self) -> list:
        """Get names of all computed metrics."""
        return [
            'condition_number', 'effective_rank', 'significant_dims_95pct',
            'explained_variance_top5', 'explained_variance_top10', 'explained_variance_top50',
            'soft_rank', 'avg_correlation'
        ]


# Singleton instance for reuse
calculator = LinearMetricsCalculator()

def compute_linear_metrics(embeddings: np.ndarray, cache_key: str = "default") -> Dict[str, float]:
    """
    Compute linear metrics for embeddings using the singleton calculator.
    
    Args:
        embeddings: Input embeddings matrix
        cache_key: Identifier for caching SVD results
        
    Returns:
        Dictionary of metrics
    """
    return calculator.compute_metrics(embeddings, cache_key)