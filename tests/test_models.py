"""
Tests for model implementations
"""

import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ResNet18, create_model


class TestResNet18(unittest.TestCase):
    """Test ResNet-18 model implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 10
        self.batch_size = 4
        self.input_shape = (3, 32, 32)

    def test_model_creation(self):
        """Test model creation"""
        model = ResNet18(num_classes=self.num_classes)
        self.assertIsInstance(model, ResNet18)

        # Test factory function
        model2 = create_model(num_classes=self.num_classes)
        self.assertIsInstance(model2, ResNet18)

    def test_forward_pass(self):
        """Test forward pass"""
        model = ResNet18(num_classes=self.num_classes)
        model.eval()

        # Create dummy input
        x = torch.randn(self.batch_size, *self.input_shape)

        # Forward pass
        output = model(x)

        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)

        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_parameter_operations(self):
        """Test parameter get/set operations"""
        model = ResNet18(num_classes=self.num_classes)

        # Get parameters
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)

        # Modify parameters
        modified_params = {}
        for name, param in params.items():
            modified_params[name] = param + 0.1

        # Set parameters
        model.set_parameters(modified_params)

        # Verify parameters were updated
        new_params = model.get_parameters()
        for name in params.keys():
            self.assertFalse(torch.equal(params[name], new_params[name]))

    def test_model_training_mode(self):
        """Test model training/evaluation modes"""
        model = ResNet18(num_classes=self.num_classes)

        # Test training mode
        model.train()
        self.assertTrue(model.training)

        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)

    def test_different_num_classes(self):
        """Test model with different number of classes"""
        for num_classes in [5, 10, 100]:
            model = ResNet18(num_classes=num_classes)
            x = torch.randn(self.batch_size, *self.input_shape)
            output = model(x)

            expected_shape = (self.batch_size, num_classes)
            self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()