#-*- coding:utf-8 -*-
import tensorflow as tf
import modules
from sklearn.externals import joblib

tf.app.flags.DEFINE_integer("batchSize",20,"batch size")
tf.app.flags.DEFINE_integer("embeddingSize",128,"embedding size")
tf.app.flags.DEFINE_integer("cellSize",128,"cell size")
tf.app.flags.DEFINE_integer("numHiddens",2,"number of cell hidden layer")
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
    data_size = joblib.load("./data/testDataSize.pkl")
    assert data_size % batch_size == 0,"total data size can't be divisible by batch."
    filename_queue = tf.train.string_input_producer(['./data/testData.tfrecords'],1)
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
    with tf.name_scope("model"):
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding",
                                        [vocab_size,embedding_size],
                                        tf.float32,
                                        tf.random_uniform_initializer(-1.0,1.0)
                                        )
            inputs = tf.nn.embedding_lookup(embedding,texts_batch)
            # (batch_size,max_sent_num,max_sent_length,embedding_size)
        with tf.name_scope("wordLevel"):
            wordEncoder = modules.WordEncoder(cell_size,n_hiddens)
            outputs,sequence_length = wordEncoder(batch_size,inputs,sentences_length_batch)
            wordAttention = modules.WordAttention(4*cell_size)
            sentVecs,sentVec_sequence_length = wordAttention(outputs,sequence_length,batch_size)
            #sentVecs: shape=(batch_size,max_sent_num,2*cell_size)
            #sentVec_sequence_length: shape=(batch_size,)
        with tf.name_scope("sentenceLevel"):
            sentEncoder = modules.SentenceEncoder(cell_size,n_hiddens)
            outputs,output_sequence_length = sentEncoder(batch_size,sentVecs,sentVec_sequence_length)
            sentAttention = modules.SentenceAttention(4*cell_size)
            docVecs = sentAttention(outputs,output_sequence_length,batch_size)
            #(batch_size,2*cell_size)
    with tf.name_scope("evaluationIndicator"):
        logits = tf.layers.dense(docVecs,cls_num,tf.nn.softmax)
        acc = tf.reduce_mean(
            tf.to_float(
                tf.equal(tf.argmax(logits,axis=1),labels_batch)
            )
        )
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
        sess.run([tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        saver.restore(sess,"./model/TextClassifier")
        accuracy = 0.0
        for i in range(data_size // batch_size):
            o = sess.run(acc)
            print("batch:%d accuracy:%f" %(i+1,o))
            accuracy += o
        print("\ntotal data accuracy:%f" %(accuracy / (data_size // batch_size)))
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
