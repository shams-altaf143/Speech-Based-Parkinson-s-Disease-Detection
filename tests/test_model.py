import torch

from auditorynet.augment import spec_augment
from auditorynet.model import AuditoryNet, A1Block, BeltBlock, count_parameters


def test_block_shapes():
    assert A1Block(1, 32)(torch.randn(2, 1, 64, 128)).shape == (2, 32, 64, 128)
    assert A1Block(32, 64)(torch.randn(2, 32, 32, 64)).shape == (2, 64, 32, 64)
    assert BeltBlock(64)(torch.randn(2, 64, 16, 32)).shape == (2, 64, 16, 32)


def test_forward_and_params():
    m = AuditoryNet()
    out = m(torch.rand(4, 1, 64, 128))
    assert out.shape == (4, 2)
    assert count_parameters(m) > 0


def test_spec_augment_masks_but_keeps_shape():
    torch.manual_seed(0)
    x = torch.ones(1, 64, 128)
    y = spec_augment(x, p=1.0)
    assert y.shape == x.shape and y.sum() < x.sum() and x.sum() == 64 * 128
    assert torch.equal(spec_augment(x, p=0.0), x)
