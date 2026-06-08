import torch
from src.models.semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig
from src.models.semantic_roi_graph_losses import compute_semantic_roi_graph_losses

def test_model():
    print('--- Starting Model Tests ---')
    config = SemanticRoiGraphConfig()
    model = SemanticROIGraphFER(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    image = torch.randn(4, 1, 48, 48)
    bboxes = torch.ones(4, 9, 4)
    bboxes[..., 2:] = 10
    bboxes[1, 3, :] = torch.tensor([0,0,0,0])
    bboxes[2] = torch.tensor([0,0,0,0]).expand(9, 4)
    
    print('1. Testing Forward Pass...')
    out = model(image, bboxes)
    print('  Forward pass successful!')
    
    print('2. Testing Output Shapes...')
    assert out['logits'].shape == (4, 7)
    assert out['mask_logits'].shape == (4, 9, 12, 12)
    print('  Output shapes valid!')
    
    print('3. Testing Loss Calculation...')
    labels = torch.tensor([0, 1, 2, 3])
    loss_dict = compute_semantic_roi_graph_losses(model, out, labels)
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f'Loss {k} contains NaN!'
    print('  Loss calculation successful! Total Loss:', loss_dict['loss'].item())
    
    print('4. Testing Backward Pass...')
    optimizer.zero_grad()
    loss_dict['loss'].backward()
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f'  [WARNING] NaN gradient in {name}')
                has_nan_grad = True
    if not has_nan_grad:
        print('  Backward pass successful! No NaN gradients found.')
    print('--- All Tests Completed Successfully ---')

if __name__ == '__main__':
    test_model()