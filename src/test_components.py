import unittest
from pathlib import Path
from src.models import build_convnext_small
from src.preprocessing import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

class ModelTests(unittest.TestCase):
    def test_architecture_keys(self):
        import torch
        m=build_convnext_small(2,head_style="legacy")
        keys=m.state_dict().keys()
        self.assertIn("classifier.2.1.weight",keys)
        self.assertEqual(tuple(m.classifier[-1][-1].weight.shape),(2,768))
        del m
    def test_preprocessing_spec(self):
        self.assertEqual(IMAGE_SIZE,(224,224))
        self.assertEqual(len(IMAGENET_MEAN),3)
        self.assertEqual(len(IMAGENET_STD),3)

if __name__=="__main__":
    unittest.main()
