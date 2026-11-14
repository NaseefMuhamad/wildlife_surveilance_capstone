"""Regression for low-confidence detections inflating the displayed confusion matrix."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pipeline_cli import runtime, operating_matrix


class ConfusionThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch,_=runtime('cpu')

    def test_matrix_confidence_does_not_inherit_ap_floor(self):
        t=self.torch
        matrix=operating_matrix({0:'animal',1:'human'},.25,.5)
        batch={'cls':t.tensor([0.]),'bboxes':t.tensor([[0.,0.,10.,10.]])}
        detections={'cls':t.tensor([0.,1.]),'conf':t.tensor([.9,.01]),
                    'bboxes':t.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])}
        matrix.process_batch(detections,batch,conf=.001)
        self.assertEqual(matrix.matrix[0,0],1)
        self.assertEqual(matrix.matrix[1,2],0)

    def test_matrix_uses_documented_matching_iou(self):
        t=self.torch
        matrix=operating_matrix({0:'animal',1:'human'},.25,.5)
        batch={'cls':t.tensor([0.]),'bboxes':t.tensor([[0.,0.,10.,10.]])}
        detections={'cls':t.tensor([0.]),'conf':t.tensor([.9]),
                    'bboxes':t.tensor([[3.5,0.,13.5,10.]])}
        matrix.process_batch(detections,batch,conf=.001,iou_thres=.45)
        self.assertEqual(matrix.matrix[0,0],0)
        self.assertEqual(matrix.matrix[0,2],1)
        self.assertEqual(matrix.matrix[2,0],1)

if __name__ == '__main__':
    unittest.main()
