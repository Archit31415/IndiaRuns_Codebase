import numpy as np

def precision_at_k(actual_relevance: list, k: int) -> float:
    """
    Calculates Precision@K.
    
    Args:
        actual_relevance (list): A list of binary relevance scores (1 for relevant, 0 for not) 
                                 ordered by the model's predicted ranking.
        k (int): The cutoff rank.
        
    Returns:
        float: The proportion of relevant items in the top K results.
    """
    if k == 0:
        return 0.0
    
    top_k = actual_relevance[:k]
    return sum(top_k) / k

def average_precision(actual_relevance: list) -> float:
    """
    Calculates Average Precision (AP) for a single query.
    
    Args:
        actual_relevance (list): A list of binary relevance scores ordered by the model's ranking.
        
    Returns:
        float: The average precision score.
    """
    relevant_count = 0
    cumulative_precision = 0.0

    for i, is_relevant in enumerate(actual_relevance):
        if is_relevant > 0:
            relevant_count += 1
            cumulative_precision += relevant_count / (i + 1)

    if relevant_count == 0:
        return 0.0

    return cumulative_precision / relevant_count

def mean_average_precision(actual_relevance_lists: list) -> float:
    """
    Calculates Mean Average Precision (MAP) across multiple queries/tests.
    
    Args:
        actual_relevance_lists (list of lists): A list containing relevance arrays for multiple tests.
        
    Returns:
        float: The MAP score.
    """
    ap_scores = [average_precision(rel) for rel in actual_relevance_lists]
    return np.mean(ap_scores) if ap_scores else 0.0

def dcg_at_k(actual_relevance: list, k: int) -> float:
    """
    Calculates Discounted Cumulative Gain (DCG) at K.
    Supports graded relevance (e.g., 0, 1, 2, 3).
    
    Args:
        actual_relevance (list): Graded relevance scores ordered by the model's ranking.
        k (int): The cutoff rank.
        
    Returns:
        float: The DCG score.
    """
    actual_relevance = np.asarray(actual_relevance)[:k]
    if actual_relevance.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, actual_relevance.size + 2))
    return np.sum(actual_relevance / discounts)

def ndcg_at_k(actual_relevance: list, k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) at K.
    
    Args:
        actual_relevance (list): Graded relevance scores ordered by the model's ranking.
        k (int): The cutoff rank.
        
    Returns:
        float: The NDCG score bounded between 0.0 and 1.0.
    """
    dcg_val = dcg_at_k(actual_relevance, k)
    
    ideal_relevance = sorted(actual_relevance, reverse=True)
    idcg_val = dcg_at_k(ideal_relevance, k)
    
    if idcg_val == 0.0:
        return 0.0
        
    return dcg_val / idcg_val