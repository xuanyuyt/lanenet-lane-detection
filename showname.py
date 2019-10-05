import tensorflow as tf
org_weights_path="./model/tusimple_lanenet_mobilenet_v2/tusimple_lanenet_mobilenet_v2_2019-10-02-15-45-47.ckpt-37401"
# org_weights_path="./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg_changename.ckpt"
org_weights_mess = []
load = tf.train.import_meta_graph(org_weights_path + '.meta')
with tf.Session() as sess:
    load.restore(sess, org_weights_path)
    for var in tf.global_variables():
        var_name = var.op.name
        var_name_mess = str(var_name).split('/')
        var_shape = var.shape
        org_weights_mess.append([var_name, var_shape])
        print(var_name, var_shape)