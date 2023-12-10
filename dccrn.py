import tensorflow as tf
import tensorflow.keras as keras


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda
from complexmodules import Dccrn


def snr_cost(s_estimate, s_true):
    snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True) + 1e-8)
        # using some more lines, because TF has no log10
    num = tf.math.log(snr + 1e-8) 
    denom = tf.math.log(tf.constant(10, dtype=num.dtype))
    loss = -10*(num / (denom))
    return loss

class DCCRN_model():
    def __init__(self, rnn_layer,num_units,length_in_s,batch_size=16,lr=1e-3):
        self.model=None
        self.numDP = 2
        self.block_len=512
        self.block_shift=256
        self.sample_len=length_in_s*16000
        self.batch_size=batch_size
        self.lr=lr
        self.dccrn=Dccrn(rnn_layer=rnn_layer,num_unit=num_units) 
    
    def stftLayer(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer
        mode: 'mag_pha'   return magnitude and phase spectrogram
              'real_imag' return real and imaginary parts
        '''
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.block_len, self.block_shift)
       
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        output_list = []
    
        real = tf.math.real(stft_dat)
        imag = tf.math.imag(stft_dat)
        output_list = [real, imag]            
        # returning magnitude and phase as list
        return output_list
    
    def ifftLayer(self, x):
        s1_stft = tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)
        return tf.signal.irfft(s1_stft) 
    def overlapAddLayer(self, x):
       
        return tf.signal.overlap_and_add(x, self.block_shift)
    
    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''
        def lossFunction(y_true,y_pred):
            # calculating loss and squeezing single dimensions away
            loss = tf.squeeze(snr_cost(y_pred,y_true)) 
            # calculate mean over batches
            loss = tf.reduce_mean(loss)
            return loss 
        
        return lossFunction
    
    def build_model(self):
        time_dat=Input(batch_shape=(self.batch_size,self.sample_len))
        real,imag = Lambda(self.stftLayer)(time_dat)
        out_real,out_imag=self.dccrn(real,imag)

        out_wav=Lambda(self.ifftLayer)([out_real,out_imag])
        est_wav=Lambda(self.overlapAddLayer)(out_wav)
        self.model=Model(inputs=[time_dat],outputs=[est_wav])
     
        print(self.model.summary())
    def compile_model(self):
        '''
        Method to compile the model for training
        '''
        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(),optimizer=optimizerAdam)
if __name__=='__main__':
    model=DCCRN_model(2,256,10,16)
    model.build_model()
    model.compile_model()
    #model.model.save_weights('./dccrn.h5')