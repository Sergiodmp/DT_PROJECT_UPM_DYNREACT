

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################################################SELF-ATTENTION MODULE
class MaskedCausalAttention(nn.Module):                                                    #En esta clase se desarrolla el mecanismo de atención
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        #1. Convertimos nuestra entrada (como embedding) a keys,queries y values. Cada una de estas es un perceptrón. Le metemos una dimensión determinada (dimensión embedding se suele llamar)
        
        
        self.n_heads = n_heads                                                              #multihead attention
        self.max_T = max_T                                                                  #embedding size
        self.q_net = nn.Linear(h_dim, h_dim).double()                                       # Take h_dim and map to h_dim         Queries
        self.k_net = nn.Linear(h_dim, h_dim).double()                                       # Convert to double          keys
        self.v_net = nn.Linear(h_dim, h_dim).double()                                       # Convert to double          Values
        #self_fully_connected_out=nn.Linear(max_T, max_t)


        # Esto es lo que hace el mecanismo de atención. Aprender que keys, queries y values nos van a dar el mejor resultado.



        self.proj_net = nn.Linear(h_dim, h_dim).double()  # Convert to double               #Como usamos un multi-head attention, para mejorar la precisión, necesitamos proyectarlo a la dimensión original. Es lo que hacemos aquí
        self.att_drop = nn.Dropout(drop_p)                                                  #Dropout is applied to avoid overfitting, avoiding the fluctuations
        self.proj_drop = nn.Dropout(drop_p)
        ones = torch.ones((max_T, max_T), dtype=torch.double)
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
                                                                                            # register buffer makes sure mask does not get updated
                                                                                            # during backpropagation
        self.register_buffer('mask', mask)


    def forward(self, x):
        B, T, C = x.shape                                                               # batch size, seq length, h_dim * n_heads (nº features, en una imagen sería el número de pixeles, aquí es como la cantidad de datos)

        N, D = self.n_heads, C // self.n_heads                                          # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)                       
        q = self.q_net(x.double()).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x.double()).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x.double()).view(B, T, N, D).transpose(1, 2)                     #sergio 2206
                                                                                        #calculamos queries,keys y values con las funciones que hemos definido en el init


        weights = q @ k.transpose(2,3) / math.sqrt(D)                                   # causal mask applied to weights ATTENTION FORMULA BEFORE SOFTMAX                                                                                 #@ is for matrix multiplication 
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))         # normalize weights, all -inf -> 0 after softmax 
        normalized_weights = F.softmax(weights, dim=-1)                                 # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v) 
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)                #calculo de la matriz self_attention. Transponemos para que esté en orden
        out = self.proj_drop(self.proj_net(attention))
        return out

#Bloque transformer: Layer norms, multihead-attention and MLP (perceptrón multicapa)
class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):                                  #h_dim: dimensión embedding: revisar
                                                                                        #max_T: sequence length
                                                                                        #n_heads: number of blocks
        super().__init__()
        ################################################################################# SELF ATTENTION BLOCK
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        ################################################################################# FEED FORWARD BLOCK
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim).double(),
                nn.GELU(),                                        #se prueba RELU Y GELU
                                                                  #función de activación para las capas ocultas de la red para introducir no linealidad. Usamos GELU en vez de RELU
                nn.Linear(4*h_dim, h_dim).double(),
                nn.Dropout(drop_p),
            )
        ################################################################################## NORMALIZATION
        self.ln1 = nn.LayerNorm(h_dim, elementwise_affine=False).double()               # Layer norm.Convert to Double type sergio 2206
        self.ln2 = nn.LayerNorm(h_dim, elementwise_affine=False).double()               # Convert to Double type

                                                                                        #Luego se aplicará la función de pérdida en la capa de salida evaluar qué tan bien está realizando la tarea de aprendizaje supervisado.
                                                                                        #Estas funciones calculan la discrepancia entre las salidas predichas por la red neuronal y los valores de salida esperados, es decir los calculados y los de coste mínimo vaya.
                                                                                        #Las más habituales MSE (error cuadrático medio) y cross entropy


    def forward(self, x):                                                               #Reproducimos en esta función el encoder de la arquitectura transformer
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x=x.type(torch.double)
        x = x + self.attention(x)                                                       #Multi-head attention
        x = self.ln1(x)                                                                 #Add & Norm
        x = x + self.mlp(x)                                                             #Feed forward
        x = self.ln2(x)                                                                 #Add & Norm
        

        return x

#Aquí se hace el transformer como tal, en este caso el decision transformer
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)                                                               #Crea una secuencia de bloques transformer
        
                                                                                                                #embeddings are the vectors transformed from the inputs
        self.embed_ln = nn.LayerNorm(h_dim, elementwise_affine=False).double()                                  
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)                                                 #Tratamiento del Tiempo como Categórico. En el contexto de un Decision Transformer, el tiempo se considera a menudo como una variable categórica. Cada paso de tiempo en una secuencia (por ejemplo, cada momento en una serie temporal) es único y tiene su propio conjunto de características asociadas. Al usar nn.Embedding para los pasos de tiempo, el modelo puede aprender representaciones únicas para cada momento específico, lo que puede ayudar a capturar la dinámica temporal de los datos.
                                                                                                                #El embedding pasa [batch_size, sequence_length] a un tensor de salida de forma [batch_size, sequence_length, embedding_dim]
        self.embed_rtg = torch.nn.Linear(1, h_dim).double()                                                     
        self.embed_state = torch.nn.Linear(state_dim, h_dim)                                                    #matriz de pesos. La capa lineal pasa de [batch_size, input_features] a untensor de salida de forma [batch_size, output_features]
        self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        use_action_tanh = False      
        #prediction heads                                                                                       # True for continuous actions                                                                                                            ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1).double()
        self.predict_state = torch.nn.Linear(h_dim, state_dim).double()
        self.predict_action = nn.Sequential(                                                                    #se aplicaría una función de activación tangente hiperbolica si las acciones no fuesen discretas
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        ).double()

    def forward(self, timesteps, states, actions, returns_to_go, bandera):         
        B, T, _ = states.shape                                                                                  #B tamaño del batch y T tamaño de la secuencia


# Recorrer cada elemento en el conjunto de datos
  
        if bandera==0:
            time_embeddings = self.embed_timestep(timesteps)
            
        else:
            timesteps = timesteps.long()
            time_embeddings = self.embed_timestep(timesteps)
        states=states.float()
        
        actions = actions.long()  # Convierte a intTensor
        if actions.max().item() >= self.act_dim:
            print("Error: Action index out of range")
        action_embeddings = self.embed_action(actions)
        action_embeddings = action_embeddings + time_embeddings
        state_embeddings = self.embed_state(states)
        state_embeddings += time_embeddings                              #esto es lo que hacemos al principio en el encoder: sumamos input embedding con positional encoding.
                                                                         #El input emmbedding (states), va a ser simplemente una capa lineal
                                                                         #Hacemos lo mismo con las acciones y los rewards. Luego los juntamos
        returns_embeddings = self.embed_rtg(returns_to_go.double()) + time_embeddings
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        h = h.double()                                      
        h = self.embed_ln(h)
        # transformer and prediction
        #h = self.transformer(h)
        h = self.transformer(h.double())                                 #Aplicamos el transformer a el vector de estados, acciones y rewards
        '''# get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.'''
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)                                                                                                # Auí hacemos el paso final. Predicciones. Get predictions
        return_preds = self.predict_rtg(h[:,2].double())                    # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2].double())                   # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1].double())                 # predict action given r, s
        return state_preds, action_preds, return_preds
    
