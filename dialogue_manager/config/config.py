import os
import math
from state.state import INTENTS

os.environ['EPS_END'] = str(0)
os.environ['EPS_START'] = str(0.95)

os.environ['MAX_CONV_LEN'] = str(15)
os.environ['AVG_CONV_LEN'] = str(4)

# os.environ['REPLAY_CAPACITY'] = str(500)
os.environ['REPLAY_CAPACITY'] = str(50000)
os.environ['NUM_LAYERS'] = str(2)
# os.environ['BATCH_SIZE'] = str(32)
os.environ['BATCH_SIZE'] = str(64)
os.environ['NUM_CONVERSATIONS'] = str(5000)

# os.environ['NUM_CONVERSATIONS'] = str(4000)
os.environ['VAL_CONVERSATIONS'] = str(1300)
os.environ['TEST_CONVERSATIONS'] = str(500)
os.environ['NUM_EPOCHS'] = str(15)
os.environ['ITEMS_PER_EPOCH'] = '18000'
# os.environ['LEARN_RATE'] = '0.000001'
# os.environ['LEARN_RATE'] = '0.000075'
os.environ['LEARN_RATE'] = '7.5*10**(-5)'
# os.environ['LEARN_RATE'] = '10**(-9)'
os.environ['WEIGHT_DECAY'] = '0'
os.environ['LR_DECAY'] = '0.1'

os.environ['NUM_MODELS'] = '50'
os.environ['RS'] = '0.95'
os.environ['RF'] = '-1.0'
os.environ['RA'] = '-0.05'
os.environ['RB'] = '-0.10'
os.environ['RC'] = '-0.15'
os.environ['RL'] = '-0.1'
#os.environ['RA'] = '-0.04'
#os.environ['RB'] = '-0.17'
#os.environ['RC'] = '-0.25'
#os.environ['RL'] = '-0.1'
os.environ['RINV'] = '-1'

os.environ['OLD_TERM'] = '0'
os.environ['NEW_TERM'] = '0.05'

#os.environ['RESP_LOADA'] = '1.86'
#os.environ['RESP_LOADB'] = '1.20'
#os.environ['RESP_LOADC'] = '1.31'
#os.environ['RESP_LOADA'] = '1.0'
#os.environ['RESP_LOADB'] = '0.64'
#os.environ['RESP_LOADC'] = '0.70'
#os.environ['BASEA'] = '0.1'
#os.environ['BASEA'] = '0.15'

os.environ['PARSE_MIS'] = '0.1'
os.environ['MIS_ID1'] = '0.43'
os.environ['MIS_ID2'] = '0.29'
os.environ['MIS_ID3'] = '0.24'

os.environ['SELECT_MISTAKE'] = '0.0'

os.environ['SIMULATOR'] = 'MODEL'

#could be LIST_SAMPLE or LIST_BEST
# os.environ['MATCHER_STRAT'] = 'LIST_SAMPLE'
os.environ['MATCHER_STRAT'] = 'LIST_BEST'
os.environ['BREAK_CONDITION'] = '4'
os.environ['THRESH_REWARD'] = '0.0'
os.environ['MODE'] = 'NORM'
os.environ['NUM_ACTIONS'] = '4'
#this can be used for dimension reduction of inputs
# os.environ['REDUCED_DIM'] = str(0.5 * int(os.environ['MAX_CONV_LEN'])*len(INTENTS) + 3)
os.environ['REDUCED_DIM'] = '50'
# os.environ['REDUCED_DIM'] = str(math.ceil(math.sqrt(int(os.environ['MAX_CONV_LEN'])*len(INTENTS))))
os.environ['FEATURE_REP'] = 'actions'
os.environ['FEATURE_TRANSFORM'] = 'fft'

os.environ['EPS_DECAY'] = str((1.5 * int(os.environ['AVG_CONV_LEN']) * \
    int(os.environ['NUM_CONVERSATIONS'])))

os.environ['TARGET_TRAIL'] = str(1)

os.environ['DEVICE'] = 'cpu'

os.environ['GAMMA'] = '0.99'
os.environ['ALPHA'] = '0.1'
os.environ['SIMU_IMPREC_THRESH'] = '0.0'
# os.environ['SIMU_SAMPLE_THRESH'] = '1.0'
# os.environ['SIMULATOR'] = 'SAMPLE'

###could be 'boltzmann or epsilon'
os.environ['ACTION_SAMPLE_MODE'] = 'epsilon'
os.environ['temp_start'] = '50'
os.environ['temp_end'] = '0.001'

os.environ['CFG'] = '/home/sos/CIC/system/' + \
    'dialogue_manager/chart_parser/syntaxparsingrules_V3'
#os.environ['CFG'] = '/ilab/users/bk456/Dialogue_Research/color_in_context/system/dialogue_manager/' + \
#   'chart_parser/syntaxparsingrules_V3'
    
os.environ['INTENT_MAP'] = '/home/sos/CIC/system/' + \
    'dialogue_manager/chart_parser/gram_intent_map'
#/home/sos/Dialogue_Research/color_in_context/system
#os.environ['INTENT_MAP'] = '/ilab/users/bk456/Dialogue_Research/color_in_context/system/dialogue_manager/' + \
#    'chart_parser/gram_intent_map'

os.environ['MODELF'] = '/home/sos/CIC/system/' + \
    'dialogue_manager/model/weights/'
os.environ['ProbsD'] = '/home/sos/CICLoadedProbs'
#os.environ['MODELF'] = '/ilab/users/bk456/Dialogue_Research/color_in_context/system/dialogue_manager/' + \
#   'model/weights/'

# os.environ['LOAD_MODEL'] = 'COMPOSITE0.4.pt'
# os.environ['LOAD_MODEL'] = 'LIST_SAMPLEMODELCONSERVATIVE0.0MISS.pt'
# os.environ['LOAD_MODEL'] = 'LIST_SAMPLEMODELRGC0.0MISS.pt'
os.environ['LOAD_MODEL'] = 'LIST_SAMPLEMODELDCOMPOSITE0.0.pt'
# os.environ['LOAD_MODEL'] = 'False'

os.environ['START_INDEX'] = '0'

os.environ['VERBOSE'] = 'False'
os.environ['MTURK'] = 'False'
# os.environ['COLOR_MODEL'] = 'COMPOSITE'
# os.environ['COLOR_MODEL'] = 'DCOMPOSITE'
# os.environ['COLOR_MODEL'] = 'CONSERVATIVE'
os.environ['COLOR_MODEL'] = 'DCOMPOSITE'
os.environ['AVG_ITERATIONS'] = '50'

os.environ['TRIAL_MODE'] = 'True'
# os.environ['TRIAL_MODE'] = 'False'
os.environ['EVAL_MODE'] = 'NORMAL'