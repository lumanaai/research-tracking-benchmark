import numpy as np
import torch

def match_from_keys(query_embedding, key_embeddings, top_k: int = 1):
    """A greedy algorithm to match query and camera images via embeddings.
    Arguments:
    query_embeddings -- embeddings of query (d)
    key_embeddings -- embeddings of other reids (N x d)
    Returns:
    return index of query, if not found, returns Non
    """
    dot_matrix = torch.matmul(key_embeddings, query_embedding)  # (N x 1)
    dot_matrix = dot_matrix.squeeze()  # (N)

    argmax = torch.argmax(dot_matrix)

    if top_k > 1:
        ind = np.argpartition(dot_matrix, -top_k)[-top_k:]
        top_ind = ind[np.argsort(dot_matrix[ind])]
    elif top_k == 1:
        top_ind = [argmax]
    else:
        top_ind = np.argsort(-1 * dot_matrix)

    return argmax, dot_matrix, top_ind


def match_from_keys_list(query_embedding, key_embeddings_list):
    """
    For multi camera, many lists
    """
    return [match_from_keys(query_embedding, key_embeddings) for key_embeddings in key_embeddings_list]
