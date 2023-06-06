'''
Networks architectures
'''
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
import transformers

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN_LSTM(nn.Module):
    '''Encoder
    Embedding -> Convolutional layer -> One-layer LSTM
    '''
    def __init__(self, input_dim, embedding_dim, ker_size, n_filters, n_rnn_layers, hidden_dim):
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.lstm_hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(ker_size, embedding_dim))
        self.lstm = nn.LSTM(input_size=n_filters, hidden_size=hidden_dim, num_layers=n_rnn_layers)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
   
    def forward(self, text, h_0, c_0):
        # CNN and LSTM network
        '''
        At every time step, the model reads one chunk which has a size of 20 words.

        --- input & output dimension ---
        
        Input text: 1 * 20
        
        **Embedding**
        1.Input: 1 * 20
        2.Output: 1 * 20 * 100
        
        **CNN**
        1. Input(minibatch×in_channels×iH×iW): 1 * 1 * 20 * 100
        2. Output(minibatch×out_channels×oH×oW): 1 * 128 * 16 * 1
        
        **LSTM**
        1. Inputs: input, (h_0, c_0)
        input(seq_len, batch, input_size): (16, 1 , 128)
        c_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        h_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        2. Outputs: output, (h_n, c_n)
        output:
        h_n(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        

        '''
        embedded = self.embedding(text)
        #print(embeded.size())
        conved = self.relu(self.conv(embedded.unsqueeze(1)))  # 1 * 128 * 16 * 1
        #print(conved.size())
        # conv -> relu -> dropout
        batch = conved.size()[0]
        conved = self.dropout(conved)
        conved = conved.squeeze(3)  # 1 * 128 * 16
        conved = torch.transpose(conved, 1, 2)  # 1 * 16 * 128
        conved = torch.transpose(conved, 1, 0)  # 16 * 1 * 128
        output, (hidden, cell) = self.lstm(conved, (h_0, c_0))
        ht = hidden.squeeze(0)  # 1 * 128
        return ht, cell


class Transformer_LSTM(nn.Module):
    '''Encoder
    Embedding -> Transformer Encoder layer -> One-layer LSTM
    '''
    def __init__(self, input_dim, embedding_dim, n_trns_layers, n_rnn_layers, hidden_dim):
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.lstm_hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=n_trns_layers)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_rnn_layers)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy = torch.randint(0, 5, (1, 20))
            h0 = torch.zeros((self.n_rnn_layers, 1, self.lstm_hidden_dim))
            c0 = torch.zeros((self.n_rnn_layers, 1, self.lstm_hidden_dim))
            print(f"dummy start: {dummy.shape}")
            dummy = self.embedding(dummy)
            print(f"Embedding: {dummy.shape}")
            dummy = self.transformer_encoder(dummy)
            print(f"Transformer: {dummy.shape}")
            dummy = torch.permute(dummy, (1, 0, 2))
            print(f"permute: {dummy.shape}")
            dummy, (hidden, cell) = self.lstm(dummy, (h0, c0))
            print(f"lstm out: {dummy.shape}, hidden: {hidden.shape}, cell: {cell.shape}")
            ht = hidden.squeeze(0)
            print(ht.shape)

   
    def forward(self, text, h_0, c_0):
        # transformer and LSTM network
        '''
        At every time step, the model reads one chunk which has a size of 20 words.

        --- input & output dimension ---
        
        Input text: 1 * 20
        
        **Embedding**
        1.Input: 1 * 20
        2.Output: 1 * 20 * 100
        
        **Transformer Encoder**
        1. Input(minibatch×seq_len×embedding_dim): 1 * 20 * 100
        2. Output(minibatch×seq_len×embedding_dim): 1 * 20 * 100
        
        **LSTM**
        1. Inputs: input, (h_0, c_0)
        input(seq_len, batch, input_size): (20, 1 , 128)
        c_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        h_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        2. Outputs: output, (h_n, c_n)
        output:
        h_n(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        

        '''
        embedded = self.embedding(text)
        trns = self.transformer_encoder(embedded)
        trns = torch.permute(trns, (1, 0, 2))
        output, (hidden, cell) = self.lstm(trns, (h_0, c_0))
        ht = hidden.squeeze(0)  # 1 * 128
        return ht, cell


class Distilbert_LSTM(nn.Module):
    def __init__(self, n_rnn_layers: int, lstm_hidden_dim: int, bert_checkpoint: str = "distilbert-base-uncased", freeze_bert: bool = False):
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.distilbert = transformers.AutoModel.from_pretrained(bert_checkpoint)

        if freeze_bert:
            self.distilbert = self.distilbert.requires_grad_(False)
        
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

        dummy = self.calculate_lm_output()
        self.lstm_input_dim = dummy.shape[-1]
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=lstm_hidden_dim, num_layers=n_rnn_layers)
        self.assert_forward()

    def calculate_lm_output(self) -> torch.Tensor:
        with torch.no_grad():
            dummy = torch.randint(0, 5, (1, 20))
            dummy_mask = torch.randint(0, 1, (1, 20))
            print(f"dummy start: {dummy.shape}")
            dummy = self.distilbert(dummy, dummy_mask).last_hidden_state
            print(f"Transformer: {dummy.shape}")
            dummy = torch.permute(dummy, (1, 0, 2))
            print(f"permute: {dummy.shape}")

            return dummy
    
    def assert_forward(self):
        dummy = self.calculate_lm_output()
        h0 = torch.zeros((self.n_rnn_layers, 1, self.lstm_hidden_dim))
        c0 = torch.zeros((self.n_rnn_layers, 1, self.lstm_hidden_dim))
        with torch.no_grad():
            dummy2, (hidden, cell) = self.lstm(dummy, (h0, c0))
            print(f"lstm out: {dummy2.shape}, hidden: {hidden.shape}, cell: {cell.shape}")
            ht = hidden.squeeze(0)
            print(ht.shape)

   
    def forward(self, text, text_mask, h_0, c_0):
        # Distilbert and LSTM network
        '''
        At every time step, the model reads one chunk which has a size of 20 words and chunk mask of size 20.

        --- input & output dimension ---
        
        Input text: 1 * 20, mask: 1 * 20
        
        **DistilBERT**
        1. Input(minibatch×seq_len, minibatch×seq_len): 1 * 20 * 100
        2. Output(seq_len×minibatch×out_dim): 20 * 1 * 768
        
        **LSTM**
        1. Inputs: input, (h_0, c_0)
        input(seq_len, batch, input_size): (20, 1 , 768)
        c_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        h_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        2. Outputs: output, (h_n, c_n)
        output:
        h_n(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        

        '''
        embedded = self.distilbert(text, text_mask).last_hidden_state
        trns = torch.permute(embedded, (1, 0, 2))
        output, (hidden, cell) = self.lstm(trns, (h_0, c_0))
        ht = hidden.squeeze(0)  # 1 * 128
        return ht, cell


class Alpaca_LSTM(nn.Module):
    def __init__(self, n_rnn_layers: int, lstm_hidden_dim: int, freeze_bert: bool = False):
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.distilbert = transformers.AutoModel.from_pretrained("chavinlo/alpaca-native")

        if freeze_bert:
            self.distilbert = self.distilbert.requires_grad_(False)
        
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

        dummy = self.calculate_lm_output()
        self.lstm_input_dim = dummy.shape[-1]
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=lstm_hidden_dim, num_layers=n_rnn_layers)
        self.assert_forward()

    def calculate_lm_output(self) -> torch.Tensor:
        with torch.no_grad():
            dummy = torch.randint(0, 5, (1, 20))
            dummy_mask = torch.randint(0, 1, (1, 20))
            print(f"dummy start: {dummy.shape}")
            dummy = self.distilbert(dummy, dummy_mask).last_hidden_state
            print(f"Transformer: {dummy.shape}")
            dummy = torch.permute(dummy, (1, 0, 2))
            print(f"permute: {dummy.shape}")

            return dummy
    
    def assert_forward(self):
        dummy = self.calculate_lm_output()
        h0 = torch.zeros((self.n_rnn_layers, 1, self.lstm_hidden_dim))
        c0 = torch.zeros((self.n_rnn_layers, 1, self.lstm_hidden_dim))
        with torch.no_grad():
            dummy2, (hidden, cell) = self.lstm(dummy, (h0, c0))
            print(f"lstm out: {dummy2.shape}, hidden: {hidden.shape}, cell: {cell.shape}")
            ht = hidden.squeeze(0)
            print(ht.shape)

   
    def forward(self, text, text_mask, h_0, c_0):
        # Distilbert and LSTM network
        '''
        At every time step, the model reads one chunk which has a size of 20 words and chunk mask of size 20.

        --- input & output dimension ---
        
        Input text: 1 * 20, mask: 1 * 20
        
        **DistilBERT**
        1. Input(minibatch×seq_len, minibatch×seq_len): 1 * 20 * 100
        2. Output(seq_len×minibatch×out_dim): 20 * 1 * 768
        
        **LSTM**
        1. Inputs: input, (h_0, c_0)
        input(seq_len, batch, input_size): (20, 1 , 768)
        c_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        h_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        2. Outputs: output, (h_n, c_n)
        output:
        h_n(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        

        '''
        embedded = self.distilbert(text, text_mask).last_hidden_state
        trns = torch.permute(embedded, (1, 0, 2))
        output, (hidden, cell) = self.lstm(trns, (h_0, c_0))
        ht = hidden.squeeze(0)  # 1 * 128
        return ht, cell


class Policy_S(nn.Module):
    '''Stopping module

    Three hidden-layer MLP with 128 hidden units per layer.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()    
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
          
    def forward(self, ht):
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)
        return out
    

class Policy_N(nn.Module):
    '''Re-reading and skipping module
    
    Three hidden-layer MLP with 128 hidden units per layer.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()    
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
         
    def forward(self, ht):
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out
    

class Policy_C(nn.Module):
    '''Classifier
    
    Single-layer MLP with 128 hidden units.
    '''

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        #self.fc = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
         
    def forward(self, ht):
        #return self.fc(ht)
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ValueNetwork(nn.Module):
    '''Baseline
    Reduce the variance.

    Single-layer MLP with 128 hidden units.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        #self.fc = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, ht):
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

################################################################################################
########################## Gated TransformerXL ################################################

class Module(nn.Module):
    """nn.Module is extended by functions to compute the norm and the mean of this module's parameters."""
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        """Concatenates the gradient of this module's parameters and then computes the norm.
        Returns:
            {float}: Returns the norm of the gradients of this model's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        """Concatenates the gradient of this module's parameters and then computes the mean.
        Returns:
            {float}: Returns the mean of the gradients of this module's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None


class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension
            num_heads {int} -- Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        """
        The forward pass of the multi head attention layer.
        
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
            
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Get number of training examples and sequence lengths
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # print("*-*- Inside MHA *-*-")
        # print(f"mems: {values.shape}, query: {query.shape}")
        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)
        # print(f"MHA 1 -- values: {values.shape}, keys: {keys.shape} query: {query.shape}, mask: {mask.shape}")

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)
        # print(f"MHA 2 -- values: {values.shape}, keys: {keys.shape} query: {query.shape}, mask: {mask.shape}")


        # Einsum does matrix mult. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)
        # print(f"MHA 3 -- energy: {energy.shape}")
        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN
        # print(f"MHA 4 -- energy: {energy.shape}")
        # Normalize energy values and apply softmax wo retreive the attention scores
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # print(f"MHA 5 -- attention: {attention.shape}")
        # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        # print(f"MHA 6 -- out: {out.shape}")
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Forward projection
        out = self.fc_out(out)
        # print(f"MHA 7 -- energy: {out.shape}")
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_dim)

        return out, attention
        
class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, config):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.
        Arguments:
            embed_dim {int} -- Size of the embeddding dimension
            num_heads {int} -- Number of attention headds
            config {dict} -- General config
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Setup GTrXL if used
        self.use_gtrxl = config["gtrxl"] if "gtrxl" in config else False
        if self.use_gtrxl:
            self.gate1 = GRUGate(embed_dim, config["gtrxl_bias"])
            self.gate2 = GRUGate(embed_dim, config["gtrxl_bias"])

        # LayerNorms
        self.layer_norm = config["layer_norm"]
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.layer_norm == "pre":
            self.norm_kv = nn.LayerNorm(embed_dim)

        # Feed forward projection
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        """print("-- Inside block forward --")
        print(f"mems[:, :, i]: {value.shape}, h.unsqueeze(1): {query.shape}, mask: {mask.shape}")
        print(f"Applying normalization")"""
        # Apply pre-layer norm across the attention input
        if self.layer_norm == "pre":
            query_ = self.norm1(query)
            value = self.norm_kv(value)
            key = value
        else:
            query_ = query

        # print(f"Before MHA value: {value.shape}, key: {key.shape}, query: {query_.shape}, mask: {mask.shape}")
        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)
        # print(f"After MHA B1 -> attention: {attention.shape}, attention_weights: {attention_weights.shape}")

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            h = self.gate1(query, attention)
        else:
            # Skip connection
            h = attention + query
        
        # print(f"B2 -> h: {h.shape}")
        
        # Apply post-layer norm across the attention output (i.e. projection input)
        if self.layer_norm == "post":
            h = self.norm1(h)

        # Apply pre-layer norm across the projection input (i.e. attention output)
        if self.layer_norm == "pre":
            h_ = self.norm2(h)
        else:
            h_ = h
        
        # print(f"B3 -> h_: {h_.shape}")

        # Forward projection
        forward = self.fc(h_)
        # print(f"B4 -> forward: {forward.shape}")

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            out = self.gate2(h, forward)
        else:
            # Skip connection
            out = forward + h
        # print(f"B5 -> out: {out.shape}")
        
        # Apply post-layer norm across the projection output
        if self.layer_norm == "post":
            out = self.norm2(out)
        # print(f"B6 -> out: {out.shape}")

        return out, attention_weights

class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb

class Transformer(nn.Module):
    """Transformer encoder architecture without dropout. Positional encoding can be either "relative", "learned" or "" (none)."""
    def __init__(self, config, input_dim, max_episode_steps) -> None:
        """Sets up the input embedding, positional encoding and the transformer blocks.
        Arguments:
            config {dict} -- Transformer config
            input_dim {int} -- Dimension of the input
            max_episode_steps {int} -- Maximum number of steps in an episode
        """
        super().__init__()
        self.config = config
        self.num_blocks = config["num_blocks"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.trns_input_dim = config["trns_input_dim"]
        self.max_episode_steps = max_episode_steps
        self.activation = nn.ReLU()

        # Input embedding layer
        self.linear_embedding = nn.Embedding(30528, self.embed_dim) # nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))
        self.conv = nn.Conv2d(100, 10, 4, 1, padding="same")

        # Determine positional encoding
        if config["positional_encoding"] == "relative":
            self.pos_embedding = SinusoidalPosition(dim = self.trns_input_dim // self.embed_dim)
        elif config["positional_encoding"] == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.trns_input_dim // self.embed_dim)) # (batch size, max episoded steps, num layers, layer size)
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.trns_input_dim, self.num_heads, config) 
            for _ in range(self.num_blocks)])

    def forward(self, h, memories, mask, memory_indices):
        """
        Arguments:
            h {torch.tensor} -- Input (query)
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D)
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L)
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L)
        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
            {torch.tensor} -- Out memories (i.e. inputs to the transformer blocks)
        """
        # Feed embedding layer and activate
        # print("Inside transformer forward")
        # print(f"h: {h.shape}, memories: {memories.shape}, mask, {mask.shape}, memory_indices: {memory_indices.shape}")
        h = self.linear_embedding(h)
        # print(f"after embedding h: {h.shape}")
        h = h.permute(0, 2, 1).unsqueeze(2)
        h = self.activation(self.conv(h))
        # print(f"1 h: {h.shape}")
        h = h.reshape(h.shape[0], -1)
        # print(f"1.1: h: {h.shape}")
        # Add positional encoding to every transformer block input
        if self.config["positional_encoding"] == "relative":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
            # memories[:,:,0] = memories[:,:,0] + pos_embedding # add positional encoding only to first layer?
        elif self.config["positional_encoding"] == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)
            # memories[:,:,0] = memories[:,:,0] + self.pos_embedding[memory_indices] # add positional encoding only to first layer?
        # print(f"2 memories: {memories.shape}")
        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            # print(f"before block mems_i: {memories[:,:,i].shape}")
            h, attention_weights = block(memories[:, :, i], memories[:, :, i], h.unsqueeze(1), mask) # args: value, key, query, mask
            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        out_memories = torch.stack(out_memories, dim=1)
        # print("After blocks, h, out_memories: ", h.shape, out_memories.shape)
        return h, out_memories
    
class GRUGate(nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
        """
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)

if __name__ == "__main__":
    model = Alpaca_LSTM(3, 128)