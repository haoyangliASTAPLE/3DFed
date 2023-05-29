import logging
import random
from utils.parameters import Params
logger = logging.getLogger('logger')

def adaptive_tuning(params: Params, accept, alpha, k, weakDP):
    if weakDP:
        logger.warning("3DFed: disable adaptive tuning")
        for i in range(len(alpha)):
            alpha[i] = 0.1
        return alpha, k

    group_size = params.fl_adv_group_size
    num_adv = params.fl_number_of_adversaries
    accept_adv = accept[:num_adv]
    accept_dec = accept[num_adv:]
    logger.warning(f'3DFed: acceptance status {accept}')
    k -= accept[num_adv:].count('a')
    k = max(k, 0)
    if 'a' not in  accept_adv and \
            'c' not in accept_adv and \
                not accept_dec.count('a') > 0:
        k += 1
    logger.info(f'3DFed: number of decoy models {k}')

    # Adaptively decide alpha
    alpha_candidate = []
    for i in range(int(num_adv / group_size)):
        count = accept[i*group_size:(i+1)*group_size].count('a')
        if count >= group_size * 0.8:
            alpha_candidate.append(alpha[i])
    alpha_candidate.sort()

    for i in range(int(num_adv / group_size)):
        # if the attacker has only one group
        if int(num_adv / group_size) <= 1:
            if len(alpha_candidate) <= 0:
                for j in range(len(alpha)):
                    alpha[j] = random.uniform(params.noise_mask_alpha, 1.)
            break

        # if all the groups are accepted
        if len(alpha_candidate) == int(num_adv / group_size):
            alpha[i] = random.uniform(alpha_candidate[0], alpha_candidate[1])
        # if partial groups are accepted
        elif len(alpha_candidate) > 0:
            alpha[i] = random.uniform(alpha_candidate[0], alpha_candidate[0]+0.1)
        # if no group is accepted
        else:
            alpha[i] = random.uniform(params.noise_mask_alpha, 1.) # += 0.1
    # revise the alpha range
    for i in range(len(alpha)):
        if alpha[i] >= 1:
            alpha[i] = 0.99
        elif alpha[i] <= 0:
            alpha[i] = 0.01
    return alpha, k