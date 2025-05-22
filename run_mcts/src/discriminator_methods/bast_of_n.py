#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from run_mcts.src.discriminator_methods import BasicDiscriminator

class BoN(BasicDiscriminator):
    def __init__(self, args, device):
        super().__init__(args, device)
        
    def inference(self, paths):
        pred_answer, candidates = None, None
        
        pred_answer, candidates