import torch
import random
import os
from magis.utils.data import Context
import numpy as np

def get_term_probs(model, colors, simulation=False, turn=None):
    model_probs = None
    if simulation:
        # print(simulation)
        colors = [colors[0]] if random.random() < float(os.getenv('SIMU_IMPREC_THRESH')) else colors
    
    if len(colors) == 1:
        model = model[1]
        model_probs = model(*colors)
        target_key = 'S0_probability'
        return model_probs[target_key][0]
    #find the index
    for i in range(3):
        if all(model[0]['x_colors'][i].view(-1) == colors[0].view(-1)):
            break
    # print('Generation Shape:', model[0][i]['S'].shape)
    if turn == 'L' and os.getenv('COLOR_MODEL') == 'CONSERVATIVE':
        # print('Invoking the conversative speaker.')
        return model[0][i]['CS'][0]
    elif turn == 'L' and os.getenv('COLOR_MODEL') == 'RGC':
        # print('Invoking the RGC speaker.')
        return model[0][i]['RS'][0]
    else:
        return model[0][i]['S'][0]

def get_patch_probs(model, colors, clr_term, vocab):
    predictions = None
    # print('This is the color term:', vocab.lookup_index(int(clr_term)))
    # model = model[0]
    for i in range(3):
        if all(colors[0].view(-1) == model[0]['x_colors'][i].view(-1)):
            break
    predictions = model[0][i]['L']
    patch_probs = lookup_listener_probability(predictions, vocab, clr_term, index=True)
    
    patch_probs = [x.cpu() for x in patch_probs]

    next_color = 0 if i == 1 else (i + 1) % 3
    # last_color = 2 if next_color == 0 else (next_color + 1) % 3

    if all(model[0]['x_colors'][next_color].view(-1) == colors[1].view(-1)):
        pass
    else:
        # print('These are the probs before:', patch_probs)
        patch_probs = [patch_probs[0], patch_probs[2], patch_probs[1]]
        # print('These are the probs after:', patch_probs)
    return np.array(patch_probs)

def get_listener_predictions_cs(model, color0, color1, color2):
    """Get the Listener probabilities for the 3 colors in RGB representation
    
    Args:
        model (ModelB): the pre-trained model
        color0 : fft features for the color
        color1 : fft features for the color
        color2 : fft features for the color
    
    Returns:
        np.ndarray: [shape=(3, 829)] Probabilities for objects given one of 
            the 829 words were heard.  Probabilities are normalized across column. 
    """

    # each patch gets its turn as the target
    # print(color0.shape)
    model_output_x0_target = model(color0, color1, color2)
    model_output_x1_target = model(color1, color0, color2)
    model_output_x2_target = model(color2, color0, color1)
    
    # single batch is why the [0]
    # print(model_output_x0_target['psi_value'])
    psi_values = torch.stack([
        model_output_x0_target['psi_value'][0],
        model_output_x1_target['psi_value'][0],
        model_output_x2_target['psi_value'][0]
    ], dim=0).cpu().detach().numpy()
    
    # normalize over objects
    listener_predictions = psi_values / psi_values.sum(axis=0)
    # print(listener_predictions[0, 638])
    return listener_predictions


def get_listener_predictions_cf(model, color0, color1, color2):
    """Get the Listener probabilities for the 3 colors in RGB representation
    
    Args:
        model (ModelB): the pre-trained model
        color0 : fft features for the color
        color1 : fft features for the color
        color2 : fft features for the color
    
    Returns:
        np.ndarray: [shape=(3, 829)] Probabilities for objects given one of 
            the 829 words were heard.  Probabilities are normalized across column. 
    """
    
    # each patch gets its turn as the target
    model_output_x0_target = model(color0)
    model_output_x1_target = model(color1)
    model_output_x2_target = model(color2)
    
    # single batch is why the [0]
    psi_values = torch.stack([
        model_output_x0_target['psi_value'][0],
        model_output_x1_target['psi_value'][0],
        model_output_x2_target['psi_value'][0]
    ], dim=0).cpu().detach().numpy()
    
    # normalize over objects
    listener_predictions = psi_values / psi_values.sum(axis=0)
    return listener_predictions

def lookup_listener_probability(listener_predictions, color_vocab, word, index=False):
    """Return the probability distribution over objects given word
    
    Args:
        listener_predictions (np.ndarray): [shape=(3, 829)] 
            The probabilities for objects given the 829 color words. 
            This should be computed using the `get_listener_predictions` function
        color_vocab (magis.Vocabulary): the vocabulary data structure with XKCD color descriptions
        word (str): the observed color word
    Returns:
        np.ndarray: [shape=(3,)] The probability for each object given `word`
    Raises:
        Exception: if word is not in color_vocab
    """
    # print('**********This is the word:', \
    #     word, color_vocab.lookup_token(word), type(listener_predictions))
    word_index = -1
    if not index:
        word_index = color_vocab.lookup_token(word)
    else:
        word_index =int(word)
    return listener_predictions[:, word_index]
