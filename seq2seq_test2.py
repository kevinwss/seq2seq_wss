import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = 20
batch_size = 1
VECTOR_LEN = 20

stick_data = [[227,123,146,159,270,190,130,250,311,116,212,273,165,368,309,301,83,453,265,397],[227,119,147,161,282,175,138,247,304,97,211,269,146,358,312,299,59,423,298,401],[229,120,151,168,288,170,158,252,290,91,213,267,171,369,314,301,62,395,336,407],[226,124,170,193,242,210,183,281,329,188,214,283,190,379,290,344,75,367,338,447],[226,125,201,213,216,215,219,299,304,228,213,274,212,375,288,350,102,345,329,449],[226,124,200,214,228,221,281,267,262,299,212,275,227,372,274,362,127,335,312,466],[225,124,190,212,230,216,199,300,309,253,214,281,266,369,236,387,154,346,229,491],[226,127,167,200,231,216,146,284,319,194,214,274,214,382,301,332,182,486,187,374],[227,127,149,172,267,200,142,262,330,142,211,275,198,378,304,308,133,467,230,379]]
stick_data2 = [[270,147,198,150,327,206,132,194,368,139,212,252,141,324,268,344,73,266,314,430],[273,148,200,155,326,213,134,190,368,141,208,254,145,326,265,344,70,261,315,432],[275,150,201,162,322,216,135,195,367,148,208,262,145,331,263,346,76,270,312,434],[276,159,206,174,316,230,132,201,357,164,207,262,149,344,262,357,79,278,308,444],[278,162,206,183,299,243,136,212,361,190,206,263,162,349,256,361,83,296,292,452],[276,169,205,193,283,254,142,230,348,223,208,263,173,358,247,360,99,310,270,460],[277,171,210,205,269,253,152,253,338,239,201,258,195,361,229,359,110,332,246,465],[279,172,216,211,246,241,166,279,316,258,201,261,221,353,221,362,128,352,224,464],[277,170,234,228,219,213,186,297,280,268,198,252,246,345,201,366,148,361,198,465],[275,167,209,196,245,237,258,256,213,319,202,254,253,338,185,357,166,359,176,470],[265,158,202,171,255,241,227,257,265,320,202,253,180,348,264,328,154,461,186,384],[264,155,198,163,263,236,196,248,303,306,206,252,168,353,264,331,133,454,204,409],[266,154,201,153,286,239,162,228,349,271,207,253,160,345,271,324,89,417,234,426],[267,148,199,156,299,227,140,213,370,197,208,255,153,339,269,331,64,337,267,435],[273,151,197,155,323,212,133,194,375,152,211,253,143,325,267,341,66,275,304,437]]
#stick_data += stick_data2


def make_lstm_batch(max_sequence_length,batch_size,length, type_ = "e_input",data = stick_data):
    '''
    data = np.array(data)
    #normalize
    data = data/504
    
    data = data.reshape(max_sequence_length,batch_size,length)
    '''
    data = np.ones((10,batch_size,encoder_hidden_units)).astype(np.float32)
    data = data/2
    
    return data


data = make_lstm_batch(max_sequence_length = 9,batch_size = 1,length = 20, type_ = "e_input",data = stick_data)
#print("data",data)



def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
            for _ in range(batch_size)
            ]


batches = random_sequences(length_from=3, length_to=10,
                           vocab_lower=2, vocab_upper=10,
                           batch_size=batch_size)


def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths



    '''
    if type_ == "d_input":
        batch = np.ones(shape=[max_sequence_length,batch_size,length], dtype=np.int32)
        return batch
    else: #"d_target"
        batch = np.ones(shape=[max_sequence_length,batch_size,length], dtype=np.int32)
        return batch
    '''

lr = 0.0002

train_graph = tf.Graph()
with train_graph.as_default():
    dim = 20
    
    encoder_inputs = tf.placeholder(shape=(None, None,dim), dtype=tf.float32, name='encoder_inputs')   #[time,batch,length per time   ]
    decoder_inputs = tf.placeholder(shape=(None, None,dim), dtype=tf.float32, name='decoder_inputs')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    #encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    #decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
    decoder_inputs_embedded = decoder_inputs
    #encoder_inputs_embedded_tf = tf.convert_to_tensor(encoder_inputs_embedded)

    encoder_inputs_embedded = encoder_inputs
    #encoder
    '''
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
    )
    '''
    #encoder_outputs (10,100,20)
    #decoder
    c = tf.convert_to_tensor(np.ones((batch_size,encoder_hidden_units)).astype(np.float32))
    h = tf.convert_to_tensor(np.ones((batch_size,encoder_hidden_units)).astype(np.float32))
    encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c,h)
    
   
    
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    cell_init_state = decoder_cell.zero_state(batch_size,dtype = tf.float32)
    
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, decoder_inputs_embedded,
        initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope="G_plain_decoder",
    )
    #print("decoder_outputs",decoder_outputs.shape)
    
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)
    output_reshape = tf.reshape(decoder_outputs,[-1,VECTOR_LEN])
    w1 = tf.get_variable('G_w1', [output_reshape.get_shape()[1], VECTOR_LEN], initializer=w_init)
    b1 = tf.get_variable('G_b1', [VECTOR_LEN], initializer=b_init)
    pred = tf.matmul(output_reshape,w1)+b1
    pred = tf.reshape(decoder_outputs,[-1,batch_size,VECTOR_LEN]) #(steps,batch_size,cell_size)
    
    G_z = pred
    
    '''
    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    decoder_prediction = tf.argmax(decoder_logits, 2)
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )
    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    '''
    #D
    with tf.variable_scope('D') as scope:
        def discriminator_lstm(x,drop_out = 0):
    
    # initializers
            w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
            b_init = tf.constant_initializer(0.)
    
            encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, x,
                dtype=tf.float32, time_major=True,
            )
            c = encoder_final_state.c
            h = encoder_final_state.h
            #print(h,c)
            decoder_final = tf.concat([c,h],1)
        #decoder_final = h
    
            w1 = tf.get_variable('D_w1', [decoder_final.get_shape()[1], 1], initializer=w_init)
            b1 = tf.get_variable('D_b1', [1], initializer=b_init)
    
            o = tf.sigmoid(tf.matmul(decoder_final, w1) + b1)
            return o
    
        #x = tf.placeholder(tf.float32, shape=(None,None,VECTOR_LEN))  #(time,batch,vector_length)
        D_real = discriminator_lstm(encoder_inputs)
        scope.reuse_variables()
        D_fake = discriminator_lstm(G_z)

    eps = 1e-2
    D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
    G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

    # trainable variables for each network
    t_vars = tf.trainable_variables()
    print(t_vars)
    D_vars = [var for var in t_vars if 'D_' in var.name]
    G_vars = [var for var in t_vars if 'G_' in var.name]

    # optimizer for each network
    D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)


loss_track = []
epochs = 3001


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        G_losses = []
        D_losses = []
        
        batch = next(batches)
        encoder_inputs_= make_lstm_batch(max_sequence_length = 9,batch_size = 1,length = 20, type_ = "e_input",data = stick_data)# distribution to learn
        #print(encoder_inputs_)
        
        #print(encoder_inputs_.shape)
        decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])# not use   
        decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])
        
        decoder_inputs_ = np.ones((10,batch_size,encoder_hidden_units)).astype(np.float32)
        
        
        feed_dict = {encoder_inputs: encoder_inputs_, decoder_inputs: decoder_inputs_,
                     decoder_targets: decoder_targets_,
                     }
         
        # update discriminator
        loss_d_, _ = sess.run([D_loss, D_optim], {encoder_inputs: encoder_inputs_,decoder_inputs: decoder_inputs_})
        D_losses.append(loss_d_)
        
        # update generator
        loss_g_, _ = sess.run([G_loss, G_optim], {encoder_inputs: encoder_inputs_,decoder_inputs: decoder_inputs_})
        G_losses.append(loss_g_)
        '''
        _, l = sess.run([train_op, loss], feed_dict)
        loss_track.append(l)
        '''
        if epoch == 0 or epoch % 1000 == 0:
            print(encoder_inputs_)
            print('loss_d: {}'.format(sess.run(D_loss, feed_dict)))
            print('loss_g: {}'.format(sess.run(G_loss, feed_dict)))
            
            predict_ = sess.run(decoder_outputs, feed_dict)
            #sess.run(decoder_outputs, feed_dict)
            print("decoder_inputs_",decoder_inputs_.shape)
            print("predict_",predict_.shape,predict_)
            
            '''
            for i, (inp, pred) in enumerate(zip(feed_dict[encoder_inputs].T, predict_.T)):
                print('input > {}'.format(inp))
                print('predicted > {}'.format(pred))
                if i >= 20:
                    break
            '''
plt.plot(loss_track)
plt.show()
