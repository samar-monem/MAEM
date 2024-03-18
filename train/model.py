!pip install tensorflow == 2.13.0
!pip install spektral
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Minimum,Maximum,Add,Maximum,PReLU, Flatten,Reshape,Dropout, Input,Dense,Add,concatenate,BatchNormalization, Activation,Lambda#,MultiHeadAttention,AdditiveAttention
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from spektral.transforms import LayerPreprocess
from sklearn.metrics import confusion_matrix, mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_score,cohen_kappa_score,precision_recall_curve,average_precision_score,roc_auc_score
from tensorflow.keras import regularizers
np.random.seed(10)

def scaled_dot_product_attention(q, k, v, mask=None):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits)#, axis=-1)#tf.nn.linear()#, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output,attention_weights 

#multi-head

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model,activation='relu',use_bias='true')
    self.wk = tf.keras.layers.Dense(d_model,activation='relu',use_bias='true')
    self.wv = tf.keras.layers.Dense(d_model,activation='relu',use_bias='true')

    self.dense = tf.keras.layers.Dense(d_model)#,activation='relu',use_bias='true')

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
#     q=PReLU()(q)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
#     k=PReLU()(k)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
#     v=PReLU()(v)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output#, attention_weights



class CrossStitch(tf.keras.layers.Layer):

    """Cross-Stitch implementation according to arXiv:1604.03539
    Implementation adapted from https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning"""

    def __init__(self, num_tasks, *args, **kwargs):
        """initialize class variables"""
        self.num_tasks = num_tasks
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        """initialize the kernel and set the instance to 'built'"""
        self.kernel = self.add_weight(name="kernel",
                                      shape=(self.num_tasks,
                                             self.num_tasks),
                                      initializer='identity',
                                      trainable=True)
        super(CrossStitch, self).build(input_shape)

    def call(self, xl):
        """
        called by TensorFlow when the model gets build. 
        Returns a stacked tensor with num_tasks channels in the 0 dimension, 
        which need to be unstacked.
        """
        if (len(xl) != self.num_tasks):
            # should not happen
            raise ValueError()

        out_values = []
        for this_task in range(self.num_tasks):
            this_weight = self.kernel[this_task, this_task]
            out = tf.math.scalar_mul(this_weight, xl[this_task])
            for other_task in range(self.num_tasks):
                if this_task == other_task:
                    continue  # already weighted!
                other_weight = self.kernel[this_task, other_task]
#                 out += tf.math.scalar_mul(other_weight, xl[other_task])
            out_values.append(out)
        # HACK!
        # unless we stack, and then unstack the tensors, TF (2.0.0) can't follow
        # the graph, so it aborts during model initialization.
        # return tf.stack(out_values, axis=0)
        return out_values[0],out_values[1]

    def compute_output_shape(self, input_shape):
        return [self.num_tasks] + input_shape

    def get_config(self):
        """implemented so keras can save the model to json/yml"""
        config = {
            "num_tasks": self.num_tasks
        }
        base_config = super(CrossStitch, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

    
 

def generate_network_att1(drug1,drug2,cell):
    # fill the architecture params from dict

    
    cell_layers = [1024,512,128]
    drop=0.0
#     snp_layers = [512,128]

#     ddi_layers=[1024,512,256]
    t_layers=[567,128]
    g_layers=[254,128]
#     g_layers=[327,128]
    dsn1_layers = [394,128]
    dsn2_layers = [394,128]
    f_layers=[142,128]
    
    c_layers=[1024,512,256]
    
    m_layers=[209,128]
    p_layers=[1024,512,256]
    
    l2_reg = 1e-3  # L2 regularization rate

#     p_layers=[drug1.shape[1],512,128]

    dropout=0.0

   
    # contruct two parallel networks
   
    drug1=Input(shape=(drug1.shape[1],))
    
    for layer in range(len(p_layers)):
      if layer == 0:

        train = Dense(int(p_layers[layer]), activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(drug1)
        train = Dropout(float(drop))(train)
        
      elif layer == (len(p_layers)-1):
        drug11 = Dense(int(p_layers[layer]), activation='linear',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(train)

      else:
        train = Dense(int(p_layers[layer]), activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(train)
        train = Dropout(float(drop))(train)
        
        
    drug2=Input(shape=(drug2.shape[1],))
    
    for layer in range(len(p_layers)):
      if layer == 0:

        train = Dense(int(p_layers[layer]), activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(drug2)
        train = Dropout(float(drop))(train)
        
      elif layer == (len(p_layers)-1):
        drug22 = Dense(int(p_layers[layer]), activation='linear',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(train)

      else:
        train = Dense(int(p_layers[layer]), activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(train)
        train = Dropout(float(drop))(train)
        
        
    cell=Input(shape=(cell.shape[1],),name='train')
    
    for layer in range(len(c_layers)):
      if layer == 0:

        train = Dense(int(c_layers[layer]), activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(cell)
        train = Dropout(float(drop))(train)
        
      elif layer == (len(c_layers)-1):
        cell1 = Dense(int(c_layers[layer]), activation='linear',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(train)

      else:
        train = Dense(int(c_layers[layer]), activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(train)
        train = Dropout(float(drop))(train)
 

  
    concatModel=concatenate([drug11,drug22,cell1])
 
    
    layer1 =MultiHeadAttention(d_model=concatModel.shape[1], num_heads=4)
    a_task1= layer1(concatModel,concatModel,concatModel)
    layer2 = MultiHeadAttention(d_model=concatModel.shape[1], num_heads=4)
    a_task2= layer2(concatModel,concatModel,concatModel)
    task11 = Reshape([a_task1.shape[2]])(a_task1)
    task22 = Reshape([a_task2.shape[2]])(a_task2)
    task1=concatenate([task11,concatModel])
    task2=concatenate([task22,concatModel])
    
    r_task1,r_task2 = CrossStitch(2)([task1,task2])

    
    r_task1=Dense(2048,activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(r_task1)
    
    r_task2=Dense(2048, activation='relu',use_bias=True,kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(r_task2)

                             
    r_task1,r_task2 = CrossStitch(2)([r_task1,r_task2])

    
    r_task1=concatenate([r_task1,task1])
    r_task2=concatenate([r_task2,task2])


    r_task1 = Dense(1024, activation='linear',kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(r_task1)
    r_task1=PReLU()(r_task1)
    r_task2 = Dense(1024, activation='relu',kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001))(r_task2)
    
    r_task1 = Dense(128, activation='linear',kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001),name='fsynergy1')(r_task1)
    r_task1=PReLU()(r_task1)
    r_task2 = Dense(128, activation='relu',kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001),name='fclass1')(r_task2)

   
    r_task1 = Dense(64, activation='linear',kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001),name='fsynergy2')(r_task1)
    r_task1=PReLU()(r_task1)
    r_task2 = Dense(64, activation='relu',kernel_initializer="he_normal",kernel_regularizer=regularizers.L1(0.001),
                     activity_regularizer=regularizers.L2(0.001),name='fclass2')(r_task2)


    
    snp_output1 = Dense(1, activation='linear',name='synergy')(r_task1)
    snp_output2 = Dense(3, activation='sigmoid',name='class')(r_task2)
    

    model = Model(inputs=[drug1,drug2,cell],outputs= [snp_output1,snp_output2])

    print(model.summary())
    return model


def trainer_att1(model, train_d1,train_d2,train_c,train_synergy,train_class, epo, batch_size, earlyStop,test_d1,test_d2,test_c,test_synergy,test_class1):

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001)


    model.compile(optimizer=optimizer,loss={'synergy':'mse','class':'categorical_crossentropy'})

    model.fit([train_d1,train_d2,train_c],[train_synergy,train_class],shuffle=True, epochs=epo, batch_size=batch_size,verbose=1 )
              

    return model

