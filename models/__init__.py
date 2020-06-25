from .vae import *
from .pt_vae import *
from .architectures.encoders import fc as fc_encoders
from .architectures.encoders import cnn as cnn_encoders
from .architectures.encoders import rnn as rnn_encoders
from .architectures.decoders import fc as fc_decoders
from .architectures.decoders import cnn as cnn_decoders
from .architectures.decoders import rnn as rnn_decoders

vae_models = {'VAE': BaseVAE,
              'NVDM': PTVAE,
              'NTM': PTVAE,
              'GSM': PTVAE,
              'HFVAE': HFVAE
              }

encoders = {'FCEncoder': fc_encoders.FCEncoder,
            'PTFCEncoder': fc_encoders.PTFCEncoder,
            'HFCEncoder': fc_encoders.HFCEncoder,
            'ConvEncoder': cnn_encoders.ConvEncoder,
            'RNNEncoder': rnn_encoders.RNNEncoder
            }

decoders = {'FCDecoder': fc_decoders.FCDecoder,
            'PTFCDecoder': fc_decoders.PTFCDecoder,
            'HFCDecoder': fc_decoders.HFCDecoder,
            'ConvDecoder': cnn_decoders.ConvDecoder,
            'RNNDecoder': rnn_decoders.RNNDecoder
            }
