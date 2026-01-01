import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部, 1=Info, 2=Warning, 3=Error

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.cluster import KMeans

true_num_states=3
tfd = tfp.distributions

def generateSeq1D(n_steps):
    # The estimation of the initial probs would be too noisy if only a single sequence is
    # used as the training data
    true_init_probs = tf.constant([0.6, 0.3, 0.1])
    true_transition_probs = tf.constant([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ])
    true_observation_locs = tf.constant([0.1, 1.0, 2.5])
    true_observation_scale = tf.constant([0.1, 0.3, 0.5])
    true_HMM=tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(probs=true_init_probs),
        transition_distribution=tfd.Categorical(probs=true_transition_probs),
        observation_distribution=tfd.Normal(loc=true_observation_locs, scale=true_observation_scale),
        num_steps=n_steps,
    )
    observations=true_HMM.sample(seed=42)
    return observations

def generateDataset(n_sequences,min_len=10,max_len=50):
    observations_list=[]
    for _ in range(n_sequences):
        obs=generateSeq1D(np.random.randint(min_len,max_len+1))
        observations_list.append(obs)
    return observations_list

@tf.function
def train_step(vars_to_opt,optimizer,observations):
    init_logits,transition_logits,obs_locs,obs_scales=vars_to_opt
    with tf.GradientTape() as tape:
        hmm=tfd.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(logits=init_logits),
            transition_distribution=tfd.Categorical(logits=transition_logits),
            observation_distribution=tfd.Normal(loc=obs_locs,
                                                scale=tf.nn.softplus(obs_scales)+1e-6), # ensure positivity
            num_steps=tf.shape(observations)[0],
        )
        loss=-hmm.log_prob(observations)
    gradients=tape.gradient(loss, vars_to_opt)
    optimizer.apply_gradients(zip(gradients, vars_to_opt))
    return loss

@tf.function
def batch_train_step(vars_to_opt,optimizer,observations_list):
    def compute_loss(seq_obs):
        init_logits, transition_logits, obs_locs, obs_scales = vars_to_opt
        hmm=tfd.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(logits=init_logits),
            transition_distribution=tfd.Categorical(logits=transition_logits),
            observation_distribution=tfd.Normal(loc=obs_locs,
                                                scale=tf.nn.softplus(obs_scales)+1e-6), # ensure positivity
            num_steps=tf.shape(seq_obs)[0],
        )
        return -hmm.log_prob(seq_obs)

    with tf.GradientTape() as tape:
        log_probs=tf.map_fn(compute_loss,elems=observations_list,dtype=tf.float32)
        avg_loss=tf.reduce_mean(log_probs)
    gradients=tape.gradient(avg_loss, vars_to_opt)
    optimizer.apply_gradients(zip(gradients, vars_to_opt))
    return avg_loss

@tf.function
def loop_train_step(vars_to_opt,optimizer,observations_list):
    init_logits, transition_logits, obs_locs, obs_scales = vars_to_opt
    total_loss=0.0
    with tf.GradientTape() as tape:
        for seq_obs in observations_list:
            hmm=tfd.HiddenMarkovModel(
                initial_distribution=tfd.Categorical(logits=init_logits),
                transition_distribution=tfd.Categorical(logits=transition_logits),
                observation_distribution=tfd.Normal(loc=obs_locs,
                                                    scale=tf.nn.softplus(obs_scales)+1e-6), # ensure positivity
                num_steps=tf.shape(seq_obs)[0],
            )
            loss=-hmm.log_prob(seq_obs)
            total_loss+=loss
        avg_loss=total_loss/len(observations_list)
    gradients=tape.gradient(avg_loss, vars_to_opt)
    optimizer.apply_gradients(zip(gradients, vars_to_opt))
    return avg_loss



class HMM:
    def __init__(self, init_probs, transition_probs, obs_locs, obs_scales):
        self.init_probs = init_probs # shape (n_states,), (K,)
        self.transition_probs = transition_probs # shape (n_states, n_states), (K,K)
        self.obs_locs = obs_locs # shape (n_states,n_features), (K,D)
        self.obs_scales = obs_scales # shape (n_states,n_features,n_features), (K,D,D)
        self.hmm=None

    def sample(self,n_steps):
        if self.hmm is None:
            raise ValueError("HMM model has not been fitted.")
        return self.hmm.sample(n_steps)

    def predict(self,observations):
        if self.hmm is None:
            raise ValueError("HMM model has not been fitted.")
        self.hmm = tfd.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(probs=self.init_probs),
            transition_distribution=tfd.Categorical(probs=self.transition_probs),
            observation_distribution=tfd.Normal(loc=self.obs_locs, scale=self.obs_scales),
            num_steps=tf.shape(observations)[0],
        )
        return self.hmm.posterior_modes(observations)

    def fit(self,sequences, n_epochs=100,n_interval=10):
        # Placeholder for fitting method
        log_p=tf.reduce_mean(self.log_likelihood(sequences))
        for epoch in range(n_epochs):
            gammas,xis=self.Estep(sequences)
            self.Mstep(sequences,gammas,xis)
            if epoch%n_interval==0:
                new_log_p=tf.reduce_mean(self.log_likelihood(sequences))
                print(f'Epoch {epoch}, Log Likelihood: {new_log_p.numpy():.4f}')
                if tf.abs(new_log_p - log_p)/(log_p+1e-8) < 1e-4:
                    break
                log_p=new_log_p


    def Estep(self,X):
        '''
        All sequences are padded to the same length with zeros,
        X shape: (n_sequences, n_steps, n_features), (N,T,D)
        :param sequences: a list of observation sequences
        :return: gammas: shape (n_sequences, n_steps, n_states), (N,T,K)
                 xis: shape (n_sequences, n_states, n_states), (N,K,K)
                 xis=\sum_t xi^n_t(i,j) for the n-th sequence
        '''
        with tf.GradientTape() as tape:
            self.hmm=tfd.HiddenMarkovModel(
                initial_distribution=tfd.Categorical(probs=self.init_probs),
                transition_distribution=tfd.Categorical(probs=self.transition_probs),
                observation_distribution=tfd.Normal(loc=self.obs_locs, scale=self.obs_scales),
                num_steps=tf.shape(X)[1],
            )
            transition_logits=self.hmm.transition_distribution.logits
            log_probs=self.hmm.log_prob(X)
        gammas=self.hmm.posterior_marginals(X)
        # Compute xis
        xis=tape.gradient(log_probs, transition_logits)
        return gammas,xis

    def Mstep(self,X,gammas,xis):
        '''
        :param X: padded observation sequences, shape (n_sequences, n_steps, n_features), (N,T,D)
        :param gammas: posterior state probabilities, shape (n_sequences, n_steps, n_states), (N,T,K)
         0<=gammas[n,t,k]<=1, sum_k gammas[n,t,k]=1
        :param xis: summed posterior marginal joint probabilities,
                    shape (n_sequences, n_states, n_states), (N,K,K)
        '''
        self.init_probs=tf.reduce_sum(gammas[:,0,:],axis=0)
        self.init_probs=self.init_probs/tf.reduce_sum(self.init_probs)
        self.transition_probs=tf.reduce_sum(xis,axis=0)
        self.transition_probs=self.transition_probs/tf.reduce_sum(self.transition_probs,axis=1,keepdims=True)

        # Update observation parameters
        gamma_sum=tf.reduce_sum(gammas,axis=[0,1]) # shape (n_states,)
        weighted_locs= tf.einsum('ntk,ntd->kd', gammas, X)
        self.obs_locs=weighted_locs/tf.reshape(gamma_sum,(-1,1))

        diff=X[:, :, tf.newaxis, :] - self.obs_locs[tf.newaxis, tf.newaxis, :, :]  # shape (N,T,K,D)
        weighted_scales=tf.einsum('ntk,ntkd->kd', gammas, tf.square(diff)) # shape (K,D)
        self.obs_scales=tf.sqrt(weighted_scales/tf.reshape(gamma_sum,(-1,1))+1e-6)


    def log_likelihood(self,sequences):
        pass


if __name__=='__main__':
    n_steps=50
    n_epochs=2000
    observations=generateSeq1D(n_steps)

    print("Observations:", observations.numpy()[:10])

    # define HMM model to fit
    init_logits=tf.Variable(tf.random.uniform(shape=[true_num_states]))
    init_probs=tf.Variable(tf.nn.softmax(init_logits))
    transition_logits=tf.Variable(tf.random.uniform(shape=[true_num_states,true_num_states]))
    transition_probs=tf.Variable(tf.nn.softmax(transition_logits,axis=-1))

    # initialize observation parameters using KMeans
    # the mean locations from KMeans will be used as initial locs
    # the std given by KMeans will be used as initial scales
    obs_np=observations.numpy().reshape(-1,1) # for 1D observations
    kmeans=KMeans(n_clusters=true_num_states,random_state=42).fit(obs_np)
    obs_locs=tf.Variable(tf.sort(tf.constant(kmeans.cluster_centers_.flatten(),dtype=tf.float32)))
    obs_scales_list=[]
    for s in range(true_num_states):
        cluster_points=obs_np[kmeans.labels_==s]
        if len(cluster_points)>1:
            obs_scales_list.append(np.std(cluster_points))
        else:
            obs_scales_list.append(1.0)
    obs_scales=tf.Variable(tf.constant(obs_scales_list,dtype=tf.float32))
    print("Initial observation locs:", obs_locs.numpy())
    print("Initial observation scales:", obs_scales.numpy())

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005)
    vars_to_opt=[init_logits,transition_logits,obs_locs,obs_scales]

    '''
    for epoch in range(n_epochs):
        loss=train_step(vars_to_opt,optimizer,observations)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy():.4f}')
    print("Trained initial probs:", tf.nn.softmax(init_logits.numpy()))
    print("Trained transition probs:", tf.nn.softmax(transition_logits,axis=-1).numpy())
    print("Trained observation locs:", obs_locs.numpy())
    print("Trained observation scales:", tf.nn.softplus(obs_scales).numpy())
    print('\n')
    '''


    # Batch training using multiple sequences
    n_sequences = 100
    dataset = generateDataset(n_sequences, min_len=10, max_len=50)

    # Re-initialize parameters
    transition_logits=tf.Variable(tf.random.uniform(shape=[true_num_states,true_num_states]))

    all_obs=np.concatenate([seq.numpy() for seq in dataset]).reshape(-1,1)
    kmeans=KMeans(n_clusters=true_num_states,random_state=42).fit(all_obs)
    obs_locs=tf.Variable(tf.sort(tf.constant(kmeans.cluster_centers_.flatten(),dtype=tf.float32)))
    obs_scales_list=[]
    for s in range(true_num_states):
        cluster_points=all_obs[kmeans.labels_==s]
        if len(cluster_points)>1:
            obs_scales_list.append(np.std(cluster_points))
        else:
            obs_scales_list.append(1.0)
    obs_scales=tf.Variable(tf.constant(obs_scales_list,dtype=tf.float32))

    '''
    first_obs = np.array([seq.numpy()[0] for seq in dataset]).reshape(-1, 1)  # 提取每个序列的第一个观测
    first_states = kmeans.predict(first_obs)
    init_state_counts = np.bincount(first_states, minlength=true_num_states)+1
    init_probs_est=tf.Variable(tf.constant(init_state_counts / np.sum(init_state_counts),dtype=tf.float32))
    init_logits=tf.Variable(tf.math.log(init_probs_est+1e-6),dtype=tf.float32)
    
    transition_counts = np.ones((true_num_states, true_num_states))
    for seq in dataset:
        seq_np = seq.numpy().reshape(-1, 1)
        states = kmeans.predict(seq_np)
        for i in range(len(states) - 1):
            transition_counts[states[i], states[i + 1]] += 1
    transition_probs_est = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    transition_logits = tf.Variable(tf.math.log(tf.constant(transition_probs_est,dtype=tf.float32)), name="trans_logits")
    '''

    print("\n===== Batch Training Initialization =====")
    #print("Estimated initial probs from KMeans:", init_probs_est)
    #print("Estimated transition probs from KMeans:\n", transition_probs_est)
    print("Initial observation locs:", obs_locs.numpy())
    print("Initial observation scales:", tf.nn.softplus(obs_scales).numpy())


    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
    vars_to_opt=[init_logits,transition_logits,obs_locs,obs_scales]
    n_epochs=500
    for epoch in range(n_epochs):
        loss=loop_train_step(vars_to_opt,optimizer,dataset)
        if epoch % 50 == 0:
            print(f'[Batch] Epoch {epoch}, Loss: {loss.numpy():.4f}')

    print("Trained initial probs:", tf.nn.softmax(init_logits).numpy())
    print("Trained transition probs:", tf.nn.softmax(transition_logits, axis=-1).numpy())
    print("Trained observation locs:", obs_locs.numpy())
    print("Trained observation scales:", (tf.nn.softplus(obs_scales) + 1e-6).numpy())





