"""Small CPU/GPU forward/backward test; it does not load the real datasets."""

import torch
import torch.nn.functional as F

from anomaly_gfm_complete_model import AnomalyGFMModel, create_model_config


def run_mode(prototype_mode: str):
    torch.manual_seed(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, time_steps, num_nodes, feature_dim = 1, 3, 5, 6
    config = create_model_config(feature_dim, time_steps)
    config.update({
        'projection_dim': 4,
        'hidden_dim': 8,
        'output_dim': 4,
        'tcn_hidden_dims': [8, 4],
        'gnn_hidden_dims': [8, 4],
        'projection_hidden_dim': 8,
        'dropout': 0.0,
        'prototype_mode': prototype_mode,
    })
    model = AnomalyGFMModel(config).to(device)
    features = torch.randn(
        batch_size, time_steps, num_nodes, feature_dim, device=device
    )
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0, 0, 1, 2, 3, 4],
    ], dtype=torch.long, device=device)
    adjacency = [
        {'edge_index': edge_index, 'edge_attr': None, 'num_nodes': num_nodes}
        for _ in range(time_steps)
    ]
    output = model(features, adjacency, features[:, -1])
    assert output['anomaly_logits'].shape == (batch_size, num_nodes)
    assert output['anomaly_scores'].shape == (batch_size, num_nodes)
    assert output['consistency_representations'].shape[:3] == (
        batch_size, time_steps, num_nodes
    )
    labels = torch.tensor(
        [[0, 1, 0, 1, 0]], dtype=torch.float32, device=device
    )
    loss = F.binary_cross_entropy_with_logits(output['anomaly_logits'], labels)
    loss.backward()
    assert model.normal_prototype.grad is not None
    if prototype_mode == 'single':
        assert model.abnormal_prototype is None
    else:
        assert model.abnormal_prototype is not None
        assert model.abnormal_prototype.grad is not None
    model.eval()
    with torch.no_grad():
        inference_output = model(features, adjacency, features[:, -1])
    assert inference_output['recency_neighbor_inconsistency'].shape == (
        batch_size, num_nodes
    )
    assert torch.isfinite(inference_output['anomaly_scores']).all()
    print(f'{prototype_mode}: OK on {device.type}, loss={loss.item():.6f}')


if __name__ == '__main__':
    run_mode('single')
    run_mode('dual')
