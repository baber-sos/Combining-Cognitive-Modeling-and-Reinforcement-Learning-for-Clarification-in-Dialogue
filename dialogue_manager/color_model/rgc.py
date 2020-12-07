from magis.algorithms import rsa_monroepotts, rsa_mcdowellgoodman, rgc_only
from magis.models.model_b import ModelB, ModelBWithRGC
from magis.settings import REPO_ROOT

import torch
import os


class RGC:
    def __init__(self):
        self.rsa_mg_alpha=15.0
        self.rsa_mg = self.rsa_mg = rsa_mcdowellgoodman.LockedAlpha(
            ModelB.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B1")), 
            rsa_alpha=self.rsa_mg_alpha
        )

        self.rsa_ooc = rsa_monroepotts.ContextFreeAlgorithm(
            ModelB.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B1"))
        )
        
        self.d_rgc = rgc_only.RGCAlgorithm(
            ModelBWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B1"))
        )
        
        self.model_order = ["rsa_mg", "rsa_ooc", "d_rgc"]

        self.speaker_probas = None
        self.listener_probas = None
        self.rgc_speaker = None

        rsa_mg_prior = 0.194
        rsa_ooc_prior = 0.469
        rgc_prior = 0.337

        self.prior = torch.FloatTensor([
            rsa_mg_prior, rsa_ooc_prior, rgc_prior
        ]).view(3, 1)
    
    def set_new_context(self, context):
        self.reset()
        self.model_contexts = [context.copy() for _ in self.model_order]
        speaker_probas = []
        listener_probas = []
        for model_name, model_context in zip(self.model_order, self.model_contexts):
            model = getattr(self, model_name)
            model(model_context)
            if model_name == "rsa_ooc":
                speaker_probas.append(model_context.S1_probabilities[:, :3])
                listener_probas.append(model_context.L1_probabilities[:, :3])
            elif model_name == "rsa_mg":
                speaker_probas.append(model_context.S1_probabilities[:, :3])
                listener_probas.append(model_context.L2_probabilities[:, :3])
            elif model_name == "d_rgc":
                self.rgc_speaker = model_context.S0_probabilities
                speaker_probas.append(model_context.S0_probabilities)
                listener_probas.append(model_context.L0_probabilities)
            else:
                raise Exception(f"{model_name} not recognized")

        self.listener_probas = torch.stack(listener_probas)
        self.listener_probas = self.listener_probas * self.prior.view(-1, 1, 1, 1)
        self.listener_probas = self.listener_probas.sum(dim=0)

        self.speaker_probas = torch.stack(speaker_probas)
        self.speaker_probas = self.speaker_probas * self.prior.view(-1, 1, 1, 1)
        self.speaker_probas = self.speaker_probas.sum(dim=0)
    
    def reset(self):
        self.model_contexts = None
        self.speaker_probas = None
        self.listener_probas = None
        self.rgc_speaker = None



