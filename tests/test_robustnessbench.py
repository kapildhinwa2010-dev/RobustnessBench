import pytest
import torch
from robustnessbench import RobustnessBench, create_robustnessbench

class TestRobustnessBench:
    def test_init(self):
        m = RobustnessBench(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = RobustnessBench(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_robustnessbench()
        assert isinstance(m, RobustnessBench)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
