
import tensorflow.keras as keras
import keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Layer
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, PReLU,ZeroPadding2D,LSTM,Multiply,Reshape
def complex_cat(input1,input2):
    input_shape=input1.shape[-1]

    real1=input1[:,:,:,:input_shape//2]
    img1=input1[:,:,:,input_shape//2:]
    real2=input2[:,:,:,:input_shape//2]
    img2=input2[:,:,:,input_shape//2:]
    cat_real=Concatenate(axis=-1)([real1,real2])
    cat_imag=Concatenate(axis=-1)([img1,img2])
    cat_out=Concatenate(axis=-1)([cat_real,cat_imag])
    return cat_out


class complexrnn(Layer):
    def __init__(self, num_units,stateful=False,**kwargs):
        super(complexrnn,self).__init__(**kwargs)
        self.real_rnn=LSTM(units=num_units,return_sequences=True,stateful=stateful)
        self.imag_rnn=LSTM(units=num_units,return_sequences=True,stateful=stateful)
    def call(self,in_real,in_imag):
        r2rout=self.real_rnn(in_real)
        r2iout=self.imag_rnn(in_real)
        i2rout=self.real_rnn(in_imag)
        i2iout=self.imag_rnn(in_imag)
        real_out = r2rout - i2iout
        imag_out = i2rout + r2iout 
     
        return real_out,imag_out
    
class complex_prelu(Layer):
    def __init__(self,**kwargs):
        super(complex_prelu,self).__init__(**kwargs)
        self.r_prelu=PReLU(shared_axes=[1,2])
        self.i_prelu=PReLU(shared_axes=[1,2])
    def call(self,inputs):
        fre_num=inputs.shape[-1]
        in_real=inputs[:,:,:,:fre_num//2]
        in_imag=inputs[:,:,:,fre_num//2:]
        out_real=self.r_prelu(in_real)
        out_imag=self.i_prelu(in_imag)
        out=Concatenate(axis=-1)([out_real,out_imag])
        return out


    
class complexconv2d(Layer):
    def __init__(self,channels,kernel_size,stride=2,padding_size=2,**kwargs):
        super(complexconv2d,self).__init__( **kwargs)
        self.real_conv=Conv2D(filters=channels,kernel_size=(2,kernel_size),padding ='valid',strides=(1,stride))
        self.img_conv=Conv2D(filters=channels,kernel_size=(2,kernel_size),padding ='valid',strides=(1,stride))
        self.padding=ZeroPadding2D(padding=((1,0),(padding_size,padding_size)))
    
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

class complexconvtranspose2d(Layer):
    def __init__(self,channels,kernel_size,stride=2,**kwargs):
        super(complexconvtranspose2d,self).__init__( **kwargs)
        self.real_convtra=Conv2DTranspose(filters=channels,kernel_size=(2,kernel_size),padding='same',strides=(1,stride))
        self.img_convtra=Conv2DTranspose(filters=channels,kernel_size=(2,kernel_size),padding='same',strides=(1,stride))
      
        
    def call(self,inputs):
        
        fre_num=inputs.shape[-1]
        in_real=inputs[:,:,:,:fre_num//2]
        in_imag=inputs[:,:,:,fre_num//2:]
        
        r2r=self.real_convtra(in_real)
        r2i=self.img_convtra(in_imag)
        i2r=self.real_convtra(in_real)
        i2i=self.img_convtra(in_imag)
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
      
        Mr = K.mean(xr, axis=reduction_axes)
        Mi = K.mean(xi,axis=reduction_axes)
       
        xr=xr-Mr
        xi=xi-Mi

       
        Vrr = K.mean(xr*xr,axis=reduction_axes) + self.eps
        Vii = K.mean(xi*xi,axis=reduction_axes) + self.eps
            # Vri contains the real and imaginary covariance for each feature map.
        Vri = K.mean(xr*xi,axis=reduction_axes)
        
        
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

        
        K.moving_average_update(self.moving_meanr, Mr, self.momentum)
        K.moving_average_update(self.moving_meani, Mi, self.momentum)
        K.moving_average_update(self.moving_vrr,Vrr,self.momentum)
        K.moving_average_update(self.moving_vri,Vri,self.momentum)
        K.moving_average_update(self.moving_vii,Vii,self.momentum)
          
        outputs=Concatenate(axis=-1)([yr,yi])
        return outputs

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

class complexdecoder(keras.layers.Layer):
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




class Dccrn(Layer):
    def __init__(self, rnn_layer,num_unit,fft_len=512,kernalsize=5,kernal_num=[16,32,64,64,128,128], **kwargs):
        super(Dccrn,self).__init__( **kwargs)
        self.encoders=[]
        for i in range(len(kernal_num)):
            self.encoders.append(complexencoder(kernalsize=kernalsize,kernal_num=kernal_num[i]))
        self.rnns=[]
        for i in range(rnn_layer):
            self.rnns.append(complexrnn(num_units=num_unit))
        out_len=(fft_len//2+1)//(2**len(kernal_num))+1

        self.r_transorm=keras.layers.Dense(units=out_len*kernal_num[-1])
        
        self.i_transforme=keras.layers.Dense(units=out_len*kernal_num[-1])
        
        self.decoders=[]

        for idx in range(len(kernal_num)):
            if(idx !=len(kernal_num)-1):
                self.decoders.append(complexdecoder(kernal_num=kernal_num[len(kernal_num)-2-idx]))
            else:
                self.decoders.append(complexdecoder(kernal_num=1))
        
    
    def call(self,mic_real,mic_img):
        x_real=K.expand_dims(mic_real,axis=-1)
        x_imag=K.expand_dims(mic_img,axis=-1)
       
        out=Concatenate(axis=-1)([x_real,x_imag])
        encoder_out=[]
        for layer in self.encoders:
            out=layer(out)
            print(out.shape)
            
            encoder_out.append(out)
        channel_size=out.shape[-1]
        r_rnn_in=out[:,:,:,:channel_size//2]
        i_rnn_in=out[:,:,:,channel_size//2:]

        length=out.shape[1]

        r_rnn_in=Reshape((length,-1))(r_rnn_in)
        i_rnn_in=Reshape((length,-1))(i_rnn_in)
        
        for rnn in self.rnns:
            r_rnn_in,i_rnn_in=rnn(r_rnn_in,i_rnn_in)
        
        r_rnn_in=self.r_transorm(r_rnn_in)
        i_rnn_in=self.i_transforme(i_rnn_in)
       
        r_rnn_in=Reshape((length,-1,channel_size//2))(r_rnn_in)
        i_rnn_in=Reshape((length,-1,channel_size//2))(i_rnn_in)
      
        out=Concatenate(axis=-1)([r_rnn_in,i_rnn_in])
   
    
        for idx,decoder in enumerate(self.decoders):
            out=complex_cat(out,encoder_out[-1-idx])
            print(out.shape)
            out=decoder(out)
            
            out=out[:,:,1:,:]
        
        mask_real=out[:,:,:,0]
        mask_imag=out[:,:,:,1]
        
        out_r2r=Multiply()([mask_real,mic_real])
        out_i2i=Multiply()([mask_imag,mic_img])
        out_r2i=Multiply()([mask_real,mic_img])
        out_i2r=Multiply()([mask_imag,mic_real])
        out_real=out_r2r-out_i2i
        out_imag=out_i2r+out_r2i
        return out_real,out_imag

if __name__=='__main__':
    testdccrn=Dccrn(rnn_layer=2,num_unit=256)
    mic_real=Input(batch_shape=(3,100,257))
    mic_imag=Input(batch_shape=(3,100,257))
  
    real_mask,imag_mask=testdccrn(mic_real,mic_imag)
    testmodel=Model(inputs=[mic_real,mic_imag],outputs=[real_mask,imag_mask])
    print(testdccrn)
    





        
    

        

        




        




        



