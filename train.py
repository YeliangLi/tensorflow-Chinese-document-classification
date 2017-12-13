#-*- coding:utf-8 -*-
import tensorflow as tf
import modules
from sklearn.externals import joblib

tf.app.flags.DEFINE_integer("epoches",80,"epoches")
tf.app.flags.DEFINE_integer("batchSize",20,"batch size")
tf.app.flags.DEFINE_integer("embeddingSize",128,"embedding size")
tf.app.flags.DEFINE_integer("cellSize",128,"cell size")
tf.app.flags.DEFINE_integer("numHiddens",2,"number of cell hidden layer")
tf.app.flags.DEFINE_float("dropoutKeepProb",0.8," the probability that each element is kept")
tf.app.flags.DEFINE_float("lr",0.003,"learning rate")
tf.app.flags.DEFINE_integer("decaySteps",15000,"decay steps")
tf.app.flags.DEFINE_float("decayRate",0.15,"decay rate")
tf.app.flags.DEFINE_float("clipNorm",5.0," clip norm about gradients")
FLAGS = tf.app.flags.FLAGS
max_sent_length = 50
max_sent_num = 50
lang = joblib.load("./data/lang.pkl")
vocab_size = lang.n_words
cls_num = 9 #number of classes

def main(_):
    batch_size = FLAGS.batchSize
    embedding_size = FLAGS.embeddingSize
    cell_size = FLAGS.cellSize
    n_hiddens = FLAGS.numHiddens
    clip_norm = FLAGS.clipNorm
    dropout_keep_prob = FLAGS.dropoutKeepProb
    data_size = joblib.load("./data/trainDataSize.pkl")
    filename_queue = tf.train.string_input_producer(['./data/trainData.tfrecords'],FLAGS.epoches)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    fea_dict = {}
    fea_dict['text'] = tf.FixedLenFeature([],tf.string)
    fea_dict['sentences_length'] = tf.FixedLenFeature([],tf.string)
    fea_dict['label'] = tf.FixedLenFeature([],tf.int64)
    feas = tf.parse_single_example(serialized_example,fea_dict)
    texts_batch,labels_batch,sentences_length_batch = tf.train.shuffle_batch(
        [tf.reshape(tf.decode_raw(feas['text'],tf.int32),[max_sent_num,max_sent_length]),
         feas['label'],
         tf.reshape(tf.decode_raw(feas['sentences_length'],tf.int32),[max_sent_num,])
         ],
        batch_size,
        capacity=1000,
        min_after_dequeue=500,
        num_threads=4,
        allow_smaller_final_batch=True 
    )
    # texts_batch: shape=(batch_size,max_sent_num,max_sent_length) 
    # labels_batch: shape=(batch_size,)
    # sentences_length_batch: shape=(batch_size,max_sent_num)
    with tf.device("/gpu:0"):
        with tf.name_scope("model"):
            with tf.variable_scope("embedding"):
                embedding = tf.get_variable("embedding",
                                            [vocab_size,embedding_size],
                                            tf.float32,
                                            tf.random_uniform_initializer(-1.0,1.0)
                                            )
                inputs = tf.nn.dropout(
                    tf.nn.embedding_lookup(embedding,texts_batch),
                    dropout_keep_prob
                    )
                # (batch_size,max_sent_num,max_sent_length,embedding_size)
            with tf.name_scope("wordLevel"):
                wordEncoder = modules.WordEncoder(cell_size,n_hiddens,dropout_keep_prob)
                outputs,sequence_length = wordEncoder(batch_size,inputs,sentences_length_batch)
                wordAttention = modules.WordAttention(4*cell_size)
                sentVecs,sentVec_sequence_length = wordAttention(outputs,sequence_length,batch_size,dropout_keep_prob)
                #sentVecs: shape=(batch_size,max_sent_num,2*cell_size)
                #sentVec_sequence_length: shape=(batch_size,)
            with tf.name_scope("sentenceLevel"):
                sentEncoder = modules.SentenceEncoder(cell_size,n_hiddens,dropout_keep_prob)
                outputs,output_sequence_length = sentEncoder(batch_size,sentVecs,sentVec_sequence_length)
                sentAttention = modules.SentenceAttention(4*cell_size)
                docVecs = sentAttention(outputs,output_sequence_length,batch_size,dropout_keep_prob)
                #(batch_size,2*cell_size)
        with tf.name_scope("evaluationIndicator"):
            logits = tf.layers.dense(docVecs,cls_num,tf.nn.softmax)
            labels = tf.one_hot(labels_batch,cls_num,dtype=tf.float32)
            loss = -1.0*tf.reduce_mean(tf.reduce_sum(labels * tf.log(logits),axis=1))
            tf.summary.scalar("loss",loss)
            acc = tf.reduce_mean(
                tf.to_float(
                    tf.equal(tf.argmax(logits,axis=1),labels_batch)
                )
            )
            tf.summary.scalar("accuracy",acc)    
        with tf.name_scope("optimization"):
            global_step = tf.Variable(0,False,dtype=tf.int32,name="global_step")
            lr = tf.train.exponential_decay(FLAGS.lr,
                                            global_step,
                                            FLAGS.decaySteps,
                                            FLAGS.decayRate,
                                            staircase=True
                                            )
            optimizer = tf.train.AdamOptimizer(lr)
            tvars = tf.trainable_variables()
            grads,_ = tf.clip_by_global_norm(tf.gradients(loss,tvars),clip_norm)
            train_op = optimizer.apply_gradients(zip(grads,tvars),global_step)
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)  
    config.gpu_options.allow_growth=True      
    with tf.Session(config=config) as sess:
        sess.run([tf.local_variables_initializer(),tf.global_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        fileWriter = tf.summary.FileWriter("./logs",sess.graph)
        merged = tf.summary.merge_all()
        for i in range(FLAGS.epoches * data_size // batch_size):
            if i % 200 == 0:
                _,cost,accuracy,summary = sess.run([train_op,loss,acc,merged])
                fileWriter.add_summary(summary,i+1)
            else:
                _,cost,accuracy = sess.run([train_op,loss,acc])
            print("step:%d loss:%f accuracy:%f" %(i+1,cost,accuracy))
            if i % 500 == 0:
                saver.save(sess,"./model/TextClassifier",i+1)
        saver.save(sess,"./model/TextClassifier")
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
