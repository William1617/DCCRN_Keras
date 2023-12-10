import tensorflow.keras as keras
import keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Layer
import tensorflow as tf
from tensorflow.keras.layers import Concatenate,ZeroPadding2D,LSTM,Reshape,Dense,Conv2D
from complexmodules import complexconv2d,complexconvtranspose2d,complex_prelu,complex_cat

class complexconv2d(Layer):
    def __init__(self,channels,kernel_size,stride=2,padding_size=2,**kwargs):
        super(complexconv2d,self).__init__( **kwargs)
        self.real_conv=Conv2D(filters=channels,kernel_size=(2,kernel_size),padding ='valid',strides=(1,stride))
        self.img_conv=Conv2D(filters=channels,kernel_size=(2,kernel_size),padding ='valid',strides=(1,stride))
        self.padding=ZeroPadding2D(padding=((0,0),(padding_size,padding_size)))
    
    def call(self,inputs):
        fre_num=inputs.shape[-1]
        in_real=inputs[:,:,:,:fre_num//2]
        in_imag=inputs[:,:,:,fre_num//2:]
        in_real=self.padding(in_real)
        in_imag=self.padding(in_imag)
        r2r=self.real_conv(in_real)
        r2i=self.img_conv(in_imag)
        i2r=self.real_conv(in_real)
        i2i=self.img_conv(in_imag)
        out_real=r2r-i2i
        out_imag=r2i+i2r
        out=Concatenate(axis=-1)([out_real,out_imag])
        return out

class complexbatchnorm(Layer):
    def __init__(self, num_features,eps=1e-5,momenum=0.9,**kwargs):
        super(complexbatchnorm,self).__init__( **kwargs)
        self.units=num_features
        self.momentum=momenum
        
        self.eps=eps

        self.gammarr=None
        self.gammaii=None
        self.gammari=None

        self.moving_vrr=None
        self.moving_vii=None
        self.moving_vri=None

        self.betai=None
        self.betar=None

        self.moving_meani=None
        self.moving_meanr=None

    def build(self,input_shape):

        param_shape = (input_shape[-1]//2)

        self.gammarr=self.add_weight(shape=param_shape,trainable=True,initializer='ones',name='gammarr')
        self.gammaii=self.add_weight(shape=param_shape,trainable=True,initializer='ones',name='gammaii')
        self.gammari=self.add_weight(shape=param_shape,trainable=True,initializer='ones',name='gammari')

        self.moving_vrr=self.add_weight(shape=param_shape,trainable=False,initializer='zeros',name='moving_vrr')
        self.moving_vii=self.add_weight(shape=param_shape,trainable=False,initializer='zeros',name='moving_vii')
        self.moving_vri=self.add_weight(shape=param_shape,trainable=False,initializer='zeros',name='moving_vri')

        self.betar=self.add_weight(shape=param_shape,trainable=True,initializer='ones',name='betar')
        self.betai=self.add_weight(shape=param_shape,trainable=True,initializer='ones',name='betai')

        self.moving_meanr=self.add_weight(shape=param_shape,trainable=False,initializer='zeros',name='moving_meanr')
        self.moving_meani=self.add_weight(shape=param_shape,trainable=False,initializer='zeros',name='moving_meani')
        self.built=True

    def call(self,inputs):
        input_shape = K.int_shape(inputs)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[-1]
        input_dim=input_shape[-1]//2
        xr=inputs[:,:,:,:input_dim]
        xi=inputs[:,:,:,input_dim:]
        
        Mr=self.moving_meanr
        Mi=self.moving_meani
        xr=xr-Mr
        xi=xi-Mi

        Vrr=self.moving_vrr +self.eps
        Vii=self.moving_vii +self.eps
        Vri=self.moving_vri
        
        tau = Vrr + Vii 
        delta = (Vrr * Vii) - (Vri ** 2)
        s = tf.math.sqrt(delta) # Determinant of square root matrix
        t = tf.math.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st
        Zrr=self.gammarr*Wrr+self.gammari*Wri
        Zri=self.gammarr*Wri+self.gammari*Wii
        Zir=self.gammari*Wrr+self.gammaii*Wri
        Zii=self.gammari*Wri+self.gammaii*Wii
        yr=Zrr*xr+Zri*xi+self.betar
        yi=Zir*xr+Zii*xi+self.betai
          
        outputs=Concatenate(axis=-1)([yr,yi])
        return outputs

class statecomplexrnn(Layer):
    def __init__(self, num_units, **kwargs):
        super(statecomplexrnn,self).__init__(**kwargs)
        self.real_rnn=LSTM(units=num_units,return_sequences=True,return_state=True,unroll=True)
        self.imag_rnn=LSTM(units=num_units,return_sequences=True,return_state=True,unroll=True)
        self.numUnits=num_units
    def call(self,in_real,in_imag,in_states):
        
        states_h = []
        states_c = []
        in_state = [in_states[:,0,:, 0], in_states[:,0,:, 1]]
        r2rout,h_state,c_state=self.real_rnn(in_real,initial_state=in_state)
        states_h.append(h_state)
        states_c.append(c_state)
        in_state = [in_states[:,1,:, 0], in_states[:,1,:, 1]]
        r2iout,h_state,c_state=self.imag_rnn(in_real,initial_state=in_state)
        states_h.append(h_state)
        states_c.append(c_state)
        in_state = [in_states[:,2,:, 0], in_states[:,2,:, 1]]
        i2rout,h_state,c_state=self.real_rnn(in_imag,initial_state=in_state)
        states_h.append(h_state)
        states_c.append(c_state)
        in_state = [in_states[:,3,:, 0], in_states[:,3,:, 1]]
        i2iout,h_state,c_state=self.imag_rnn(in_imag,initial_state=in_state)
        states_h.append(h_state)
        states_c.append(c_state)

        out_states_h = tf.reshape(tf.stack(states_h, axis=0), [1,4,self.numUnits])
        out_states_c = tf.reshape(tf.stack(states_c, axis=0), [1,4,self.numUnits])
        out_states = tf.stack([out_states_h, out_states_c], axis=-1)

        real_out = r2rout - i2iout
        imag_out = i2rout + r2iout 

        return real_out,imag_out,out_states
    
class complexencoder(keras.layers.Layer):
    def __init__(self, kernal_num,kernalsize=5, stride=2,**kwargs):
        super(complexencoder,self).__init__( **kwargs)
        self.conv=complexconv2d(channels=kernal_num,stride=stride,kernel_size=kernalsize)
        self.norm=complexbatchnorm(num_features=kernal_num)
   
        self.act1=complex_prelu()
    def call(self,inputs):
        r=self.conv(inputs)
        r=self.norm(r)
        outs=self.act1(r)
        return outs
    
class complexdecoder(Layer):
    def __init__(self, kernal_num,kernalsize=5, stride=2,**kwargs):
        super(complexdecoder,self).__init__( **kwargs)
        self.convtran=complexconvtranspose2d(channels=kernal_num,stride=stride,kernel_size=kernalsize)
        self.norm=complexbatchnorm(num_features=kernal_num)
        self.act1=complex_prelu()
    def call(self,inputs):
        r=self.convtran(inputs)
        r=self.norm(r)
        outs=self.act1(r)
        return outs
    
class Dccrn_with_state(Layer):
    def __init__(self, rnn_layer,num_unit,fft_len=512,kernalsize=5,kernal_num=[16,32,64,64,128,128], **kwargs):
        super(Dccrn_with_state,self).__init__( **kwargs)
        self.encoders=[]
        for i in range(len(kernal_num)):
            self.encoders.append(complexencoder(kernalsize=kernalsize,kernal_num=kernal_num[i]))
        self.rnns=[]
        self.num_rnnlayer=rnn_layer
        self.num_rnnunits=num_unit
        for i in range(rnn_layer):
            self.rnns.append(statecomplexrnn(num_units=num_unit))
        out_len=(fft_len//2+1)//(2**len(kernal_num))+1
        self.r_transorm=Dense(units=out_len*kernal_num[-1])
        self.i_transforme=Dense(units=out_len*kernal_num[-1])
        self.decoders=[]

        for idx in range(len(kernal_num)):
            if(idx !=len(kernal_num)-1):
                self.decoders.append(complexdecoder(kernal_num=kernal_num[len(kernal_num)-2-idx]))
            else:
                self.decoders.append(complexdecoder(kernal_num=1))
    
    def call(self,mic_real,mic_img,in_states,conv_caches,deconv_caches):
        x_real=K.expand_dims(mic_real,axis=-1)
        x_imag=K.expand_dims(mic_img,axis=-1)
       
        out=Concatenate(axis=-1)([x_real,x_imag])
        encoder_out=[]
        new_convcaches=[]
        new_deconvcaches=[]
        
        for idx,encoder in enumerate(self.encoders):
            new_convcaches.append(out)
            out=Concatenate(axis=1)([conv_caches[idx],out])
            out=encoder(out)
            
            encoder_out.append(out)
        channel_size=out.shape[-1]
        r_rnn_in=out[:,-1,:,:channel_size//2]
        i_rnn_in=out[:,-1,:,channel_size//2:]


        r_rnn_in=Reshape((1,-1))(r_rnn_in)
        i_rnn_in=Reshape((1,-1))(i_rnn_in)
        out_states=[]
        
        for idx,rnn in enumerate(self.rnns):
            in_state=in_states[:,idx,:,:,:]
            r_rnn_in,i_rnn_in,out_state=rnn(r_rnn_in,i_rnn_in,in_state)
            out_states.append(out_state)
        nnout_states=tf.reshape(tf.stack(out_states,axis=0),[1,self.num_rnnlayer,4,self.num_rnnunits,2])

        r_rnn_in=self.r_transorm(r_rnn_in)
        i_rnn_in=self.i_transforme(i_rnn_in)
        
        r_rnn_in=Reshape((1,-1,channel_size//2))(r_rnn_in)
        i_rnn_in=Reshape((1,-1,channel_size//2))(i_rnn_in)
      
        out=Concatenate(axis=-1)([r_rnn_in,i_rnn_in])

        for idx,decoder in enumerate(self.decoders):
            out=complex_cat(out,encoder_out[-1-idx])
            new_deconvcaches.append(out)
            out=Concatenate(axis=1)([deconv_caches[idx],out])
            out=decoder(out)
            
            out=out[:,1:,1:,:]

        mask_real=out[:,:,:,0]
        mask_imag=out[:,:,:,1]
        
   
        return mask_real,mask_imag,nnout_states,new_convcaches,new_deconvcaches

if __name__=='__main__':
    testdccrn=Dccrn_with_state(rnn_layer=2,num_unit=256)
    mic_real=Input(batch_shape=(1,1,257))
    mic_imag=Input(batch_shape=(1,1,257))
    in_states=Input(batch_shape=(1,2,4,256,2))
    
    conv_caches=[]
    deconv_caches=[]

    conv_caches.append(Input(batch_shape=(1,1,257,2)))
    conv_caches.append(Input(batch_shape=(1,1,129,32)))
    conv_caches.append(Input(batch_shape=(1,1,65,64)))
    conv_caches.append(Input(batch_shape=(1,1,33,128)))
    conv_caches.append(Input(batch_shape=(1,1,17,128)))
    conv_caches.append(Input(batch_shape=(1,1,9,256)))

    deconv_caches.append(Input(batch_shape=(1,1,5,512)))
    deconv_caches.append(Input(batch_shape=(1,1,9,512)))
    deconv_caches.append(Input(batch_shape=(1,1,17,256)))
    deconv_caches.append(Input(batch_shape=(1,1,33,256)))
    deconv_caches.append(Input(batch_shape=(1,1,65,128)))
    deconv_caches.append(Input(batch_shape=(1,1,129,64)))
  
    real_mask,imag_mask,out_staets,new_convcaches,new_deconvcaches=testdccrn(mic_real,mic_imag,in_states,conv_caches,deconv_caches)
    testmodel=Model(inputs=[mic_real,mic_imag,in_states,conv_caches,deconv_caches],outputs=[real_mask,imag_mask,out_staets,new_convcaches,new_deconvcaches])
    testmodel.load_weights('./dccrn.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(testmodel)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('dccrn.tflite', 'wb') as f:
        f.write(tflite_model)
    