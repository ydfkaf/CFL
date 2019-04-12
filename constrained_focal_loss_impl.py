def focal_loss(onehot_labels, cls_preds,                           #FL loss function
                            gamma=None, name=None, scope=None):

    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        onehot_labels = slim.one_hot_encoding(onehot_labels, 2)
        logits = tf.convert_to_tensor(cls_preds,tf.float32)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        predictions = tf.nn.softmax(logits)
        pt = tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,predictions),axis=1),[-1,1])
        loss=-tf.reduce_sum(tf.pow(1-pt,gamma)*(tf.log(pt)),axis=1)
        loss=tf.reduce_mean(loss)
        return loss



def _focal_loss(onehot_labels, cls_preds,                                      #FL* loss function
                            alpha=None,gamma=None, name=None, scope=None):

    with tf.name_scope(scope, 'focal_loss_', [cls_preds, onehot_labels]) as sc:

        onehot_labels = slim.one_hot_encoding(onehot_labels, 2)
        logits = tf.convert_to_tensor(cls_preds,tf.float32)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        predictions = tf.nn.softmax(logits)
        pt = tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,predictions),axis=1),[-1,1])
        alpha=tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,tf.constant([1.0-alpha,alpha])),axis=1),[-1,1])
 
        epsilon = 1e-8
        loss=-tf.reduce_sum(tf.pow(1-pt,gamma)*(tf.multiply(alpha,tf.log(pt+epsilon))),axis=1)
        loss=tf.reduce_mean(loss)
        return loss


def constrained_focal_loss(onehot_labels, cls_preds,                                        #CFL loss function             
                            beita=None, name=None, scope=None):
 
    with tf.name_scope(scope, 'constrained_focal_loss', [cls_preds, onehot_labels]) as sc:
        alpha=tf.reduce_sum(tf.cast(tf.reshape(onehot_labels,[-1]),dtype=tf.float32))        

        alpha=tf.cond(tf.subtract(tf.reduce_mean(alpha),tf.reduce_mean(tf.constant(0.001)))<0,lambda:tf.constant([1.0]),
    lambda:tf.reshape(tf.pow(tf.divide(tf.cast(tf.shape(tf.reshape(onehot_labels,[-1]))[0]-alpha,dtype=tf.float32),alpha),1/beita),[-1]))
  
        onehot_labels = slim.one_hot_encoding(onehot_labels, 2)
        logits = tf.convert_to_tensor(cls_preds,tf.float32)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        predictions = tf.nn.softmax(logits)
        pt = tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,predictions),axis=1),[-1,1])
        alpha=tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,tf.concat([tf.constant([1.0]),alpha],0)),axis=1),[-1,1])
        epsilon = 1e-8
        loss=-tf.reduce_sum(tf.multiply(alpha,tf.log(pt+epsilon)),axis=1)
        loss=tf.reduce_mean(loss)
        return loss

def _constrained_focal_loss(onehot_labels, cls_preds,                          #CFL* loss function
                            beita=None, name=None, scope=None):

    with tf.name_scope(scope, 'CFL_xing_loss', [cls_preds, onehot_labels]) as sc:
        alpha=tf.reduce_sum(tf.cast(tf.reshape(onehot_labels,[-1]),dtype=tf.float32))        
        heigth2_and_width=tf.cond(tf.cast(tf.shape(tf.reshape(onehot_labels,[-1]))[0],dtype=tf.float32)<2073600.0,lambda:tf.constant([513*2,513]),lambda:tf.constant([1080,1920]))
        onehot_labels_copy=tf.tile(onehot_labels,[1])    
        top_copy=tf.tile(onehot_labels,[1])     
        bottom_copy=tf.tile(onehot_labels,[1])
        left_copy=tf.tile(onehot_labels,[1])
        right_copy=tf.tile(onehot_labels,[1])


        
        onehot_labels_copy=tf.reshape(onehot_labels,heigth2_and_width)     
        top_copy=tf.reshape(top_copy,heigth2_and_width)    
        bottom_copy=tf.reshape(bottom_copy,heigth2_and_width)
        left_copy=tf.reshape(left_copy,heigth2_and_width)
        right_copy=tf.reshape(right_copy,heigth2_and_width)         
        
        top_copy_pad_bottom=tf.slice(top_copy,[3,0],tf.subtract(heigth2_and_width,tf.constant([3,0])))     
        bottom_copy_pad_top=tf.slice(top_copy,[0,0],tf.subtract(heigth2_and_width,tf.constant([3,0])))
        left_copy_pad_right=tf.slice(top_copy,[0,3],tf.subtract(heigth2_and_width,tf.constant([0,3])))
        right_copy_pad_left=tf.slice(top_copy,[0,0],tf.subtract(heigth2_and_width,tf.constant([0,3])))
        
        top_copy_pad_bottom=tf.pad(top_copy_pad_bottom,[[0,3],[0,0]])   
        bottom_copy_pad_top=tf.pad(bottom_copy_pad_top,[[3,0],[0,0]])
        left_copy_pad_right=tf.pad(left_copy_pad_right,[[0,0],[0,3]])
        right_copy_pad_left=tf.pad(right_copy_pad_left,[[0,0],[3,0]])

        onehot_labels_copy=tf.add(onehot_labels_copy,top_copy_pad_bottom)   
        onehot_labels_copy=tf.add(onehot_labels_copy,bottom_copy_pad_top)
        onehot_labels_copy=tf.add(onehot_labels_copy,left_copy_pad_right)
        onehot_labels_copy=tf.add(onehot_labels_copy,right_copy_pad_left)

        onehot_labels_copy=tf.ceil(tf.divide(tf.cast(onehot_labels_copy,dtype=tf.float32),tf.constant(10.0)))
        onehot_labels_copy=tf.reshape(onehot_labels_copy,[-1])     

        onehot_labels_copy = slim.one_hot_encoding(tf.cast(onehot_labels_copy,dtype=tf.int32), 2)       #end 


        alpha=tf.cond(tf.subtract(tf.reduce_mean(alpha),tf.reduce_mean(tf.constant(0.001)))<0,lambda:tf.constant([1.0]),
        lambda:tf.reshape(tf.pow(tf.divide(tf.cast(tf.shape(tf.reshape(onehot_labels,[-1]))[0]-alpha,dtype=tf.float32),alpha),1/beita),[-1]))
     
        onehot_labels = slim.one_hot_encoding(onehot_labels, 2)
        logits = tf.convert_to_tensor(cls_preds,tf.float32)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        predictions = tf.nn.softmax(logits)
        pt = tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,predictions),axis=1),[-1,1])
        alpha=tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels_copy,tf.concat([tf.constant([1.0]),alpha],0)),axis=1),[-1,1])
        epsilon = 1e-8
        loss=-tf.reduce_sum(tf.multiply(alpha,tf.log(pt+epsilon)),axis=1)
        loss=tf.reduce_mean(loss)
        return loss

def balanced_cross_entropy_a_loss(onehot_labels, cls_preds, name=None, scope=None):         #BCE_A loss function

    with tf.name_scope(scope, 'balanced_cross_entropy_a_loss', [cls_preds, onehot_labels]) as sc:
        alpha=tf.reduce_sum(tf.cast(tf.reshape(onehot_labels,[-1]),dtype=tf.float32))
        #print(alpha)
        alpha=tf.cond(tf.subtract(tf.reduce_mean(alpha),tf.reduce_mean(tf.constant(0.001)))<0,lambda:tf.constant([1.0]),
    lambda:tf.reshape(tf.sqrt(tf.divide(tf.cast(tf.shape(tf.reshape(onehot_labels,[-1]))[0]-alpha,dtype=tf.float32),alpha)),[-1]))
  
        onehot_labels = slim.one_hot_encoding(onehot_labels, 2)
        logits = tf.convert_to_tensor(cls_preds,tf.float32)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        predictions = tf.nn.softmax(logits)
        pt = tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,predictions),axis=1),[-1,1])
        alpha=tf.reshape(tf.reduce_sum(tf.multiply(onehot_labels,tf.concat([tf.constant([1.0]),alpha],0)),axis=1),[-1,1])
        epsilon = 1e-8
        loss=-tf.reduce_sum(tf.multiply(alpha,tf.log(pt+epsilon)),axis=1)
        loss=tf.reduce_mean(loss)

        return loss
