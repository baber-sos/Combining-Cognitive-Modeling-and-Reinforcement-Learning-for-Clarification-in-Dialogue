from magis.algorithms import rgc_only, rsa_mcdowellgoodman, rsa_monroepotts
from magis.models.model_b import ModelBWithRGC
from magis.models.model_c import ModelCWithRGC
from magis.settings import REPO_ROOT
from magis.utils.data import Context
import numpy as np
import torch
import os

class CompositeModel:
    def __init__(self, rsa_mg_alpha, s_rgc_num_samples):
        self.rsa_mg = rsa_mcdowellgoodman.LockedAlpha(
            ModelBWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B1")), 
            rsa_alpha=rsa_mg_alpha
        )

        self.rsa_ooc = rsa_monroepotts.ContextFreeAlgorithm(
            ModelBWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B1"))
        )
        
        self.s_rgc = rgc_only.RGCSamplingAlgorithm(
            ModelCWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2C1")), 
            num_samples=s_rgc_num_samples
        )
        self.model_order = ["rsa_mg", "rsa_ooc", "s_rgc"]
        self._initial_prior = torch.FloatTensor([0.207, 0.359, 0.434]).view(3, 1)
        
        self.prior = self._initial_prior
        self.speaker_probas = None
        self.listener_probas = None
        self.weighted_speaker_probas = None
        self.weighted_listener_probas = None
        self.speaker_marginals = None
        self.listener_marginals = None
        
    def reset(self):
        self.prior = self._initial_prior
        self.speaker_probas = None
        self.listener_probas = None
        self.weighted_speaker_probas = None
        self.weighted_listener_probas = None
        self.speaker_marginals = None
        self.listener_marginals = None
        self.model_contexts = None
        
    def first_turn_flow(self, context, utterance_indices):
        self.compute_model_probas(context)
        self.compute_marginals()
        self.update_model_prior(utterance_indices)
        
    def second_turn_flow(self, utterance_indices):
        self.compute_marginals()
        self.update_model_prior(utterance_indices)
        
    def compute_model_probas(self, context):
        model_results = [
            getattr(self, model_name)(context.copy())
            for model_name in self.model_order
        ]
        
        speaker_probas = []
        listener_probas = []
        model_contexts = []
        for model_name in self.model_order:
            model = getattr(self, model_name)
            model_context = model(context)
            model_contexts.append(model_context)
            if model_name == "rsa_ooc":
                speaker_probas.append(model_context.S1_probabilities[:, :3])
                listener_probas.append(model_context.L1_probabilities[:, :3])
            elif model_name == "rsa_mg":
                speaker_probas.append(model_context.S1_probabilities[:, :3])
                listener_probas.append(model_context.L2_probabilities[:, :3])
            elif model_name == "s_rgc":
                speaker_probas.append(model_context.S0_probabilities)
                listener_probas.append(model_context.L0_probabilities)
            else:
                raise Exception(f"{model_name} not recognized")
        # shape = (model, batch, object, word)
        self.speaker_probas = torch.stack(speaker_probas)
        self.listener_probas = torch.stack(listener_probas)
        self.model_contexts = model_contexts
        
    def get_listener_proba_for_utterances(self, utterance_indices):
        num_models, num_batch, num_objects, _ = self.listener_probas.size()
        # shape = (batch, object)
        return {
            "full": self.listener_probas.gather(
                dim=3, 
                index=utterance_indices.expand(num_objects, num_batch, num_objects, 1)
            ).squeeze(dim=3),
            "marginal": self.listener_marginals.gather(
                dim=2, 
                index=utterance_indices.expand(num_batch, num_objects, 1)
            ).squeeze(dim=2)
        } 
    
    def compute_marginals(self):
        self.weighted_speaker_probas = self.speaker_probas * self.prior.view(-1, 1, 1, 1)
        self.weighted_listener_probas = self.listener_probas * self.prior.view(-1, 1, 1, 1)
        
        # shape = (batch, object, word)
        MODEL_DIM = 0
        self.speaker_marginals = self.weighted_speaker_probas.sum(dim=MODEL_DIM)
        self.listener_marginals = self.weighted_listener_probas.sum(dim=MODEL_DIM)
        
    def update_model_prior(self, utterance_indices):
        raise NotImplemented

class CompositeInterface:
    def __init__(self, composite_model, vocab):
        self.composite_model = composite_model
        self.composite_model.reset()
        self.vocab = vocab
        self.prior_history = [self.get_model_probas()]
        self.context = None
    
    def set_new_context(self, context, clear_utterance_index=True):
        self.context = context
        if clear_utterance_index:
            context.utterance_index = torch.ones_like(context.utterance_index) * -1
        self.composite_model.reset()
        self.composite_model.compute_model_probas(self.context)
        self.composite_model.compute_marginals()
        
    def _vectorize_utterance(self, utterance):
        return torch.LongTensor([self.vocab[utterance]])
        
    def _get_utterance_distribution(self, target_index=0, for_model=None, listener_filter_threshold=0.0):
        assert self.context is not None
        for_model = self._resolve_for_model(for_model)
        if for_model is None:
            speaker_proba_tensor = self.composite_model.speaker_marginals
            listener_proba_tensor = self.composite_model.listener_marginals
        else:
            speaker_proba_tensor = self.composite_model.speaker_probas[for_model]
            listener_proba_tensor = self.composite_model.listener_probas[for_model]
            
        probas = speaker_proba_tensor[0, target_index].cpu().detach().numpy()
        interps = listener_proba_tensor[0, target_index].cpu().detach().numpy()
        assert len(probas.shape) == 1
        assert len(probas) == len(self.vocab)
        sorted_indices = probas.argsort()[::-1]
        return [(self.vocab.lookup_index(i), probas[i]) for i in sorted_indices if interps[i] > listener_filter_threshold]
    
    def _resolve_for_model(self, for_model):
        if for_model is None:
            return for_model
        if isinstance(for_model, str):
            assert for_model in self.composite_model.model_order
            for_model = self.composite_model.model_order.index(for_model)
        assert for_model < len(self.composite_model.model_order)
        return for_model
        
    def generate_utterance(self, target_index=0, mode="best", for_model=None, listener_filter_threshold=0.0, k_index=-1, include_interps=True):
        for_model = self._resolve_for_model(for_model)
        utterance_distribution = self._get_utterance_distribution(
            target_index=target_index, 
            for_model=for_model,
            listener_filter_threshold=listener_filter_threshold
        )
        if mode == "best":
            utterance, proba = utterance_distribution[0]
        elif mode == "sample":
            choices, probas = list(zip(*utterance_distribution))
            probas = (lambda x: x/x.sum())(np.array(probas))
            choice_index = np.random.choice(np.arange(len(choices)), size=1, p=probas).item()
            utterance, proba = utterance_distribution[choice_index]
        elif mode == "kth": 
            assert k_index >= 0
            utterance, proba = utterance_distribution[k_index]
            
        probas_source = ("marginal" if for_model is None else self.composite_model.model_order[for_model])
        output = {
            "utterance": utterance,
            "proba": proba,
            "mode": mode,
            "proba_source": probas_source
        }
        if include_interps:
            output["interpretations"] = tuple(self.interpret_utterance(utterance).tolist())
        return output
    
    def generate_k_utterances(self, top_k=5, target_index=0, for_model=None, listener_filter_threshold=0.0, include_interps=True):
        output = []
        for k in range(top_k):
            output.append(self.generate_utterance(
                target_index=target_index, 
                mode="kth", 
                for_model=for_model,
                listener_filter_threshold=listener_filter_threshold,
                k_index=k,
                include_interps=include_interps
            ))
        return output
    
    def interpret_utterance(self, utterance, update_model_priors=False, for_model=None):
        for_model = self._resolve_for_model(for_model)
        utterance_index = self._vectorize_utterance(utterance)
        object_probas = self.composite_model.get_listener_proba_for_utterances(utterance_index)
        if for_model is None:
            object_probas = object_probas["marginal"]
        else:
            object_probas = object_probas["full"][for_model]
        object_probas = object_probas.squeeze().cpu().detach().numpy()
        
        # doing this last so there's no misunderstanding that the model prior isn't updated 
        # before returned object probas
        if update_model_priors:
            raise NotImplemented
            self.composite_model.update_model_prior(utterance_index)
       
        return object_probas
        
    def get_model_probas(self):
        model_probas = self.composite_model.prior.detach().cpu().numpy().flatten()
        return dict(zip(self.composite_model.model_order, model_probas))
    
    def __call__(self, context):
        # batch_dict = dict()
        # batch_dict['x_colors'] = torch.cat(colors, dim=0).view(1, 3, -1)
        # batch_dict['y_utterance'] = -1
        self.set_new_context(context)
        return self.composite_model.speaker_marginals[0], self.composite_model.listener_marginals[0]
