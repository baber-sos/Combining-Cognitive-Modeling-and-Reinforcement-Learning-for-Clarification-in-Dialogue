import os
from magis.models.model_b import ModelBWithRGC
from magis.settings import REPO_ROOT

from color_model.composite import CompositeModel
from color_model.composite import CompositeInterface
from magis.models.composite_sigdial2020 import CompositeInterface as DComposite
from color_model.conservative import Conservative
from color_model.rgc import RGC

def get_model_interface(vocab):
    imprecision_model = ModelBWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B"))
    if os.getenv('COLOR_MODEL') == 'XKCD':
        return (ModelBWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B")), imprecision_model)
    elif os.getenv('COLOR_MODEL') == 'COMPOSITE':
        interface = CompositeInterface(
            composite_model=CompositeModel(s_rgc_num_samples=75, rsa_mg_alpha=14.0),
            vocab=vocab
        )
        return (interface, imprecision_model)
    elif os.getenv('COLOR_MODEL') == 'DCOMPOSITE':
        interface = DComposite.published_defaults()
        return (interface, imprecision_model)
    elif os.getenv('COLOR_MODEL') == 'CONSERVATIVE':
        interface = Conservative()
        return (interface, imprecision_model)
    elif os.getenv('COLOR_MODEL') == 'RGC':
        interface = RGC()
        return (interface, imprecision_model)
def use_xkcd(*args): #arguments need to be filled
    return ModelBWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B"))

def use_composite(*args): #arguments need to be filled
    pass
