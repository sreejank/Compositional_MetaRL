import optuna
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from stable_baselines import logger
from stable_baselines.common.policies import *
from stable_baselines.a2c import A2C
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from grid_env import *
import sys
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.tf_layers import conv,conv_to_fc

rules=sys.argv[2] 
register_grid_env('grid-v0',rules,hold_out=1)  
register_grid_env('test-v0',rules,hold_out=-1)   
  
#register_trace_env('trace-v0',4,['chain'],20,use_precomputed=True)
#register_battleship_env('ship-v0',4,['chain'],20)

gamma=0.9
n_steps=8
lr_schedule='linear'
lr=0.0023483181861598565
ent_coef=0.0006747109316677081
vf_coef=0.00635090082912515
num_layers=2
#n_lstm=15 
n_lstm=120 



def cnn(input_tensor,**kwargs):
    visual_input=tf.slice(input_tensor,[0,0],[-1,49],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,49],[-1,50],'prev_outputs')
    visual_input=tf.reshape(visual_input,(-1,7,7,1))
    activ=tf.nn.relu

    layer_1 = activ(conv(visual_input, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)) 
    #layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #layer_3=conv_to_fc(layer_2)
    layer_2=conv_to_fc(layer_1)
    visual_output=activ(linear(layer_2,'fc1',n_hidden=49,init_scale=np.sqrt(2)))
    total_output=tf.concat([visual_output,prev_output],1)  
    return total_output 


def mlp(input_tensor, **kwargs): 
    """
    

    :param input_tensor: (TensorFlow Tensor) Observation input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    net_arch=[49 for _ in range(num_layers)]
    visual_output=tf.slice(input_tensor,[0,0],[-1,49],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,49],[-1,50],'prev_outputs')
    activ = tf.tanh
    for i, layer_size in enumerate(net_arch):
        visual_output = activ(linear(visual_output, 'pi_fc' + str(i), n_hidden=layer_size,init_scale=np.sqrt(2)))
    total_output=tf.concat([visual_output,prev_output],1) 
    #total_output=visual_output
    return total_output 


def make_env(env_id, rank, seed=0,board=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param board: (numpy array) pre-determined board for env. 
    """
    if board is not None: 
        def _init(): 
            env = gym.make(env_id)
            env.seed(seed + rank)
            env.reset_task(board)
            return env 
    else:
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
    set_global_seeds(seed)
    return _init


if __name__=='__main__':
    num_episodes=int(sys.argv[1])

    env=make_vec_env('grid-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    
    policy_kwargs={'cnn_extractor':mlp,'n_lstm':n_lstm}
    #policy_kwargs={'cnn_extractor':cnn,'n_lstm':n_lstm}  
    model=A2C(policy='CnnLstmPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
        n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)

    model.learn(num_episodes)  
    model.save("7x7_"+rules+"_"+"metalearning.zip") 
