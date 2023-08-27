import re
import torch.nn.functional as F

def postprocess(preds, converter, cfg=None):
    if cfg is not None:
        sensitive = cfg.get('sensitive', True)
        character = cfg.get('character', '')
    else:
        sensitive = True
        character = ''

    probs = F.softmax(preds, dim=2)
    max_probs, indexes = probs.max(dim=2)
    preds_str = []
    preds_prob = []
    for i, pstr in enumerate(converter.decode(indexes)):
        str_len = len(pstr)
        if str_len == 0:
            prob = 0
        else:
            prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
        preds_prob.append(prob)
        if not sensitive:
            pstr = pstr.lower()

        if character:
            pstr = re.sub('[^{}]'.format(character), '', pstr)

        preds_str.append(pstr)

    return preds_str, preds_prob