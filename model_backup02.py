import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
# from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.variables import PartitionedVariable

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class Model_SINE():
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, topic_num, category_num, alpha,
                 neg_num, cpt_feat, user_norm, item_norm, cate_norm, n_head, share_emb = True, flag = "DNN"):

        self.model_flag = flag
        self.reg = False
        self.user_eb = None
        self.batch_size = batch_size #批处理大小
        self.n_size = n_mid # item总数
        self.lr = 0.001
        self.hist_max = seq_len # 用户行为序列长度
        self.dim = embedding_dim
        self.share_emb = share_emb
        with tf.name_scope('Inputs'):
            self.i_ids = tf.placeholder(shape=[None], dtype=tf.int32)
            self.item = tf.placeholder(shape=[None, seq_len], dtype=tf.int32)
            self.nbr_mask = tf.placeholder(shape=[None, seq_len], dtype=tf.float32)

        self.num_topic = topic_num
        self.category_num = category_num
        self.hidden_units = hidden_size
        self.alpha_para = alpha
        self.temperature = 0.07
        # self.temperature = 0.1
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.cate_norm = cate_norm
        self.neg_num = neg_num
        self.num_heads = n_head

        # Embedding layer
        '''
        item_input_lookup: [n_mid, embedding_dim]              所有item的embedding
        item_input_lookup_var: [n_mid]                         
        position_embedding: [1, self.hist_max, embedding_dim]  用户行为序列的位置编码
        item: [None, seq_len]                                  用户行为序列
        emb:  [N, embedding_dim]    N = user_count*seq_len     与用户交互的所有item的embedding编码
        item_emb: [batch_size, hist_max, embedding_dim]        每行表示一个用户的序列行为编码
        item_out_emb:归一化后的item-embedding矩阵
        '''

        with tf.name_scope('Embedding_layer'):
            self.item_input_lookup = tf.get_variable("input_embedding_var", [n_mid, embedding_dim],
                                                     trainable=True)  # 所有item的embedding编码
            self.item_input_lookup_var = tf.get_variable("input_bias_lookup_table", [n_mid],
                                                         initializer=tf.zeros_initializer(), trainable=False)
            self.position_embedding = tf.get_variable(
                shape=[1, self.hist_max, embedding_dim],
                name='position_embedding')  # 用户行为序列的位置编码
            if self.share_emb:
                self.item_output_lookup = self.item_input_lookup
                self.item_output_lookup_var = self.item_input_lookup_var
            else:
                self.item_output_lookup = tf.get_variable("output_embedding_var", [n_mid, embedding_dim],
                                                          trainable=True)
                self.item_output_lookup_var = tf.get_variable("output_bias_lookup_table", [n_mid],
                                                              initializer=tf.zeros_initializer(), trainable=False)
        '''
        tf.nn.embedding_lookup(tensor, id) - 选取一个张量里面索引对应的元素
        reshape(t,[-1]) - 按行展平，eg t = [[1,1,1],[2,2,2],[3,3,3]] 得到 [1,1,1,2,2,2,3,3,3]
        reduce_sum(x, index) 求和函数 x = 0，1，-1 按行求和，按列求和，按最后一个维度求和
        '''
        emb = tf.nn.embedding_lookup(self.item_input_lookup,
                                     tf.reshape(self.item, [-1]))
        print('reshape', tf.reshape(self.item, [-1]))
        self.item_emb = tf.reshape(emb, [-1, self.hist_max, self.dim])
        print('item_emb', self.item_emb)
        self.mask_length = tf.cast(tf.reduce_sum(self.nbr_mask, -1), dtype=tf.int32)
        self.item_output_emb = self.output_item2()



        '''
        topic_embed : [num_topic, dim]    所有主题的embedding
        self.item_emb: [-1, self.hist_max, self.dim]      每行表示一个用户的序列行为编码
        self.nbr_mask: [None, seq_len]
        self.seq_multi:　[batch_size*num_head,category_num, dim]
        self.user_eb: [batch_size*num_head, dim]                            
        '''

        if cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False

        '''
        tf.variable_scope() 用于定义创建变量层的操作的上下文管理器
        '''
        with tf.variable_scope('topic_embed', reuse=tf.AUTO_REUSE):
            self.topic_embed = \
                tf.get_variable(
                    shape=[self.num_topic, self.dim],
                    name='topic_embedding')
        self.seq_multi = self.sequence_encode_cpt(self.item_emb, self.nbr_mask)
        self.user_eb = self.labeled_attention(self.seq_multi)
        self._xent_loss_weight(self.user_eb, self.seq_multi)

    '''
            将物品的emb进行归一化
            '''

    def output_item2(self):
        if self.item_norm:
            item_emb = tf.nn.l2_normalize(self.item_output_lookup, dim=-1)
            return item_emb
        else:
            return self.item_output_lookup

    def _xent_loss(self, user):
        emb_dim = self.dim
        loss = tf.nn.sampled_softmax_loss(
            weights=self.output_item2(),
            biases=self.item_output_lookup_var,
            labels=tf.reshape(self.i_ids, [-1, 1]),
            inputs=tf.reshape(user, [-1, emb_dim]),
            num_sampled=self.neg_num * self.batch_size,
            num_classes=self.n_size,
            partition_strategy='mod',
            remove_accidental_hits=True
        )

        self.loss = tf.reduce_mean(loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        return loss

    '''
    user: [batch_size*num_head, dim]
    seq_multi:　[batch_size*num_head,category_num, dim]  
    biaese：item_input_lookup_var: [n_mid]
    labels：[-1, 1]
    inputs: [batch_size*num_head, dim]
    '''

    def _xent_loss_weight(self, user, seq_multi):
        emb_dim = self.dim
        loss = tf.nn.sampled_softmax_loss(
            weights=self.output_item2(),  # item的embedding向量进行归一化
            # weights=self.item_output_lookup,
            biases=self.item_output_lookup_var,
            labels=tf.reshape(self.i_ids, [-1, 1]),
            inputs=tf.reshape(user, [-1, emb_dim]),
            num_sampled=self.neg_num * self.batch_size,
            num_classes=self.n_size,
            partition_strategy='mod',
            remove_accidental_hits=True
        )

        regs = self.calculate_interest_loss(seq_multi)

        self.loss = tf.reduce_mean(loss)
        self.reg_loss = self.alpha_para * tf.reduce_mean(regs)
        loss = self.loss + self.reg_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return loss

    def train(self, sess, hist_item, nbr_mask, i_ids):
        feed_dict = {
            self.i_ids: i_ids,
            self.item: hist_item,
            self.nbr_mask: nbr_mask
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.item_output_emb)
        # item_embs = sess.run(self.item_output_lookup)
        return item_embs

    def output_user(self, sess, hist_item, nbr_mask):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.item: hist_item,
            self.nbr_mask: nbr_mask
        })
        return user_embs

    def save(self, sess, path):
        # if not os.path.exists(path):
        #     os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + '_model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + '_model.ckpt')
        print('model restored from %s' % path)

    def calculate_interest_loss(self, user_interest):
        norm_interests = tf.nn.l2_normalize(user_interest, -1)
        dim0, dim1, dim2 = get_shape(user_interest)

        interests_losses = []
        for i in range(1, (dim1 + 1) // 2):
            roll_interests = array_ops.concat(
                (norm_interests[:, i:, :], norm_interests[:, 0:i, :]), axis=1)
            # compute pair-wise interests similarity.
            interests_radial_diffs = math_ops.multiply(
                array_ops.reshape(norm_interests, [dim0 * dim1, dim2]),
                array_ops.reshape(roll_interests, [dim0 * dim1, dim2]))
            interests_loss = math_ops.reduce_sum(interests_radial_diffs, axis=-1)
            interests_loss = array_ops.reshape(interests_loss, [dim0, dim1])
            interests_loss = math_ops.reduce_sum(interests_loss, axis=-1)
            interests_losses.append(interests_loss)

        if dim1 % 2 == 0:
            half_dim1 = dim1 // 2
            interests_part1 = norm_interests[:, :half_dim1, :]
            interests_part2 = norm_interests[:, half_dim1:, :]
            interests_radial_diffs = math_ops.multiply(
                array_ops.reshape(interests_part1, [dim0 * half_dim1, dim2]),
                array_ops.reshape(interests_part2, [dim0 * half_dim1, dim2]))
            interests_loss = math_ops.reduce_sum(interests_radial_diffs, axis=-1)
            interests_loss = array_ops.reshape(interests_loss, [dim0, half_dim1])
            interests_loss = math_ops.reduce_sum(interests_loss, axis=-1)
            interests_losses.append(interests_loss)

        # NOTE(reed): the original interests_loss lay in [0, 2], so the
        # combination_size didn't divide 2 to normalize interests_loss into
        # [0, 1]
        self._interests_length = None
        if self._interests_length is not None:
            combination_size = math_ops.cast(
                self._interests_length * (self._interests_length - 1),
                dtypes.float32)
        else:
            combination_size = dim1 * (dim1 - 1)
        interests_loss = 0.5 + (
                math_ops.reduce_sum(interests_losses, axis=0) / combination_size)

        return interests_loss
    '''
    item_emb: [batch_size, hist_max, embedding_dim]
    self.nbr_mask: [None, seq_len]
    item_list_emb: [batch_size, hist_max, embedding_dim]
    position_embedding: [1, self.hist_max, embedding_dim]
    tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1]): [batch_size, hist_max, embedding_dim]
    item_list_add_pos: [batch_size, hist_max, embedding_dim]
    item_hidden: [batch_size, hist_max, hidden_units]
    item_att_w: [batch_size, hist_max, num_heads]
    item_att_w: [batch_size, num_heads, hist_max]
    atten_mask: [None, num_heads, hist_max]
    paddings: [None, num_heads, hist_max]
    item_att_w: [batch_size, num_heads, hist_max]
    item_att_w: [batch_size, num_heads, hist_max]
    item_emb: [batch_size, num_heads, embedding_dim]
    seq: [batch_size, num_heads, embedding_dim]
    seq: [batch_size*num_heads, embedding_dim]
    '''
    def sequence_encode_concept(self, item_emb, nbr_mask):
        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, self.num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(seq, axis=1)
                mu = tf.layers.dense(mu, self.dim, name='maha_cpt')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))
                wg = tf.nn.softmax(wg, dim=1)
                seq = tf.reduce_mean(seq * wg, axis=1)
            else:
                seq = tf.reshape(seq, [-1, self.dim])
        return seq


    '''
    seq: [batch_size*num_head,category_num, dim]
    self.cate_dist：[batch_size, category_num, hist_max]
    item_emb: [batch_size, hist_max, category_num]
    self.batch_tpt_emb(Cu): [batch_size*num, category_num, embedding_dim]
    item_emb: [batch_size, hist_max, embedding_dim]
    self.item_emb: [batch_size, hist_max, embedding_dim]
    tf.reshape(self.item_emb, [-1, self.hist_max, self.dim]): [batch_size, hist_max, embedding_dim]
    item_emb: [batch_size, hist_max, embedding_dim]
    target_item: [batch_size*num_heads, embedding_dim]
    mu_seq: [batch_size*num_heads, embedding_dim]
    target_label: [batch_size*num_heads, 2*embedding_dim]
    mu: [batch_size*num_heads, embedding_dim]
    tf.expand_dims(mu, axis=-1): [batch_size*num_heads, embedding_dim, 1]
    wg: [batch_size*num_heads, category_num, 1]
    seq*wg: [batch_size*num_head,category_num, dim]
    use_emb: [batch_size*num_head, dim]
    '''
    def labeled_attention(self, seq):
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        item_emb = tf.transpose(self.cate_dist, [0, 2, 1])
        # tf.matmul: 对于高维矩阵的相乘实质上是对高维矩阵中每个二维矩阵相乘，
        # 所以，我们要保证两个要相乘的高维矩阵的最后两个维度符合二维矩阵相乘的规则，
        # 两个高维矩阵的其他维度需要相同
        item_emb = tf.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            item_emb = item_emb + tf.reshape(self.item_emb, [-1, self.hist_max, self.dim])
        target_item = self.sequence_encode_concept(item_emb, self.nbr_mask)#[N,  D]

        mu_seq = tf.reduce_mean(seq, axis=1)  # [N,H,D] -> [N,D]

        #   t1 = [[1, 2, 3], [4, 5, 6]]
        #   t2 = [[7, 8, 9], [10, 11, 12]]
        #   tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        #   tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        target_label = tf.concat([mu_seq, target_item], axis=1)

        mu = tf.layers.dense(target_label, self.dim, name='maha_cpt2', reuse=tf.AUTO_REUSE)

        wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1)
        wg = tf.nn.softmax(wg, axis=1)

        # tf.reduce_sum(arg1, arg2)函数
        # arg1即为要求和的数据
        # arg2=0表示纵向对矩阵求和
        # arg2=1表示横向对矩阵求和
        # arg2省略时，默认对所有元素进行求和
        user_emb = tf.reduce_sum(seq * wg, axis=1)  # [N,H,D]->[N,D]

        if self.user_norm:
            user_emb = tf.nn.l2_normalize(user_emb, dim=-1)
        return user_emb


    '''
    seq_aggre: 生成Zu
    item_list_emb -- seq:[batch_size, his_max, embedding_dim] # 用户序列行为编码
    nbr_mask: [None, seq_len]
    position_embedding:[1, self.hist_max, embedding_dim]      # 用户行为序列的位置编码
    tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])：[batch_size, his_max, embedding_dim]
    item_list_add_pos:[batch_size, his_max, embedding_dim] # 包含位置信息的用户行为序列编码
    hidden_units: 默认512
    item_hidden:[batch_size, his_max, hidden_untis]
    item_att_w:[batch_size, his_max, num_aggre]
    item_att_w:[batch_size, num_aggre, his_max]  # 通过transpos 函数互换位置，每行表示一个序列行为对应的兴趣           
    tf.expand_dims(nbr_mask, axis=1): [None, 1, seq_len]  
    atten_mask: [None, num_aggre, seq_len]
    padding: [None, num_aggre, seq_len]  # 值全为-2**32 + 1
    item_att_w: [batch_size, num_aggre, his_max]
    item_emb: [batch_size, num_agree, embedding_dim]
    item_emb: [batch_size*num_agree, embedding_dim]          
    '''
    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        # tf.tile(input, multiples, name) -- input:输入，multiples：同一维度上复制的次数
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        with tf.variable_scope("self_atten_aggre", reuse=tf.AUTO_REUSE) as scope:
            # tf.layers.dense(inputs, units, activation) # 全连接层(输入数据，输出维度大小-改变inputs的最后一维，激活函数)
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_aggre, activation=None) # 全连接层
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])
            # tf.expand_dims(input, axis = None, name = None, dim = None) 给定一个input，在axis轴处增加一维（一个为1的维度）
            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, num_aggre, 1])
            # tf.ones_like(tensor, dtype = None, name=None, optimize = Truee) # 创建和tensor维度一样，元素都为1的张量
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)
            # tf.equal(x, y)    逐元素判断x和y中的元素是否相等，如果相等就是True，不相等就是false
            # tf.where(condition, x, y)   condition中取值为True的元素替换为x中的元素，为False的元素替换为y中对应的元素
            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            # item_att_w: a = softmax(tanh(Xu * W1) * W2)
            item_att_w = tf.nn.softmax(item_att_w)

            # item_emb: z_u = (aT* Xu)
            item_emb = tf.matmul(item_att_w, item_list_emb)

            # item_emb 转置
            item_emb = tf.reshape(item_emb, [-1, self.dim])

            # item_emb 对输入序列矩阵进行加权求和得到用户的综合意图
            return item_emb

    '''
    topic_select: 返回Sigmoid(Su(idx:)1T), index
    input_seq -- item_emb_input: [batch_size * his_max, embedding_dim] 用户序列行为编码
    seq：[batch_size, his_max, embedding_dim] # 用户序列行为编码
    seq_emb(Zu): [batch_size*num_agree, embedding_dim] ----对输入序列矩阵进行加权求和得到的用户的综合意图
    topic_embed(C): [num_topic, embedding_dim]    --- 概念池的embedding编码  
    topic_logit(Su):[batch_size*num_agree, num_topic]    概念池embedding矩阵 topic_embed * 综合意图 seq_emb
    top_logits(Sigmoid(Su(idx:)1T)): [batch_size*num, categorty_num]
    top_index(idx): [batch_size*num, category_num] 前category_num个topic_logit对应的索引
    '''
    def topic_select(self, input_seq):
        seq = tf.reshape(input_seq, [-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        print('seq_emb',seq_emb)
        if self.cate_norm:
            seq_emb = tf.nn.l2_normalize(seq_emb, dim=-1)
            topic_emb = tf.nn.l2_normalize(self.topic_embed, dim=-1)
            topic_logit = tf.matmul(seq_emb, topic_emb, transpose_b=True)
        else:
            # Su = <C, Zu>
            topic_logit = tf.matmul(seq_emb, self.topic_embed, transpose_b=True)#[batch_size, topic_num]
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)#two [batch_size, categorty_num] tensors
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index


    ''' 
    计算 P-k_t
    input_seq -- item_emb_input: [batch_size * his_max, embedding_dim] 用户序列行为编码
    top_logit(Sigmoid(Su(idx:)1T)): [batch_size*num, categorty_num]  前category_num个topic_logit
    top_index(idx):  [batch_size*num, categorty_num] 前category_num个topic_logit对应的索引
    self.topic_embed: [self.num_topic, self.dim]
    topic_embed(C(idx,:)):　[batch_size, category_num, embedding_dim]  top_index对应的topic embedding向量
    self.batch_tpt_emb: [batch_size*num, category_num, embedding_dim]  top_index对应的topic embedding向量
    tf.expand_dims(top_logit, axis=2)：[batch_size*num, categorty_num, 1]
    tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim]): [batch_size*num, categorty_num, embedding_dim]
    self.batch_tpt_emb(Cu): [batch_size*num, categorty_num, embedding_dim]
    norm_seq: [batch_size * his_max, embedding_dim, 1]
    cores: [batch_size, category_num, embedding_dim]
    cores_t: [batch_size*his_max, category_num, dim]
    cate_logits: [batch_size*his_max, category_num]
    cate_dist: [batch_size*his_max, category_num]
    Pk|t：
    '''
    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        # tf.nn.embedding_lookup(tensor, id) - 选取一个张量里面索引对应的元素

        # C(idx,:)
        topic_embed = tf.nn.embedding_lookup(self.topic_embed, top_index)
        self.batch_tpt_emb = tf.nn.embedding_lookup(self.topic_embed, top_index)#[-1, cate_num, dim]
        # tf.expand_dims(input, axis = None, name = None, dim = None) 给定一个input，在axis轴处增加一维（一个为1的维度）
        # tf.tile(input, multiples, name) -- input:输入，multiples：同一维度上复制的次数

        # Cu
        # tensor点积，c = a * b, 假设a是二维tensor，b是三维tensor
        # 但是a的维度与b的后两位相同，那么a和b仍然可以做点积，点积结果是一个和b维度一样的三维tensor
        self.batch_tpt_emb = self.batch_tpt_emb * tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])
        print('seq-batch_tpt_emb2:',self.batch_tpt_emb)

        norm_seq = tf.expand_dims(tf.nn.l2_normalize(input_seq, axis=1), axis=-1)#[-1, dim, 1]

        # layernorm(Cu)
        cores = tf.nn.l2_normalize(topic_embed, axis=-1) #[-1, cate_num, dim]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [1, self.hist_max, 1, 1]), [-1, self.category_num, self.dim])

        cate_logits = tf.reshape(tf.matmul(cores_t, norm_seq), [-1, self.category_num]) / self.temperature #[-1, cate_num]
        cate_dist = tf.nn.softmax(cate_logits, axis=-1)
        return cate_dist


    '''
    items: [batch_size, hist_max, embedding_dim]     每行表示一个用户的序列行为编码，用户历史行为emb
    nbr_mask: [None, seq_len]
    item_emb_input(Xu): [batch_size * his_max, embedding_dim]      经过reshpe函数展开，用户的历史行为emb
    self.seq_cate_dist(item_emb_input): [batch_size*his_max, category_num]     用户历史行为类别
    self.cate_dist：[batch_size, category_num, hist_max]     用户历史行为类别
    item_list_emb: [batch_size, his_max, embedding]      同items，用户历史行为emb
    self.position_embedding：[1, hist_max, embedding_dim]    
    tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])：[batch_size, hist_max, embedding_dim]
    item_list_add_pos：[batch_size, hist_max, embedding_dim]  包含位置信息的用户行为序列编码
    item_hiddem: [batch_size, hist_max, hidden_units] 隐藏层输出
    item_att_w: [batch_size, hist_max, nums_heads * category_num]
    item_att_w: [batch_size,  num_heads * category_num, hist_max]
    item_att_w: [batch_size, category_num, num_heads, hist_max]
    category_mask_tile: [batch_size, category_num, num_heads, hist_max]
    seq_att_w: [batch_size, category_num*num_heads, hist_max]
    atten_mask: [None, category_num*num_heads ,seq_len]
    paddings: [None, category_num*num_heads ,seq_len]
    seq_att_w: [None, category_num*num_heads ,seq_len]
    seq_att_w: [batch_size, category_num, num_head, hist_max]
    seq_att_w: [batch_size, category_num, num_head, hist_max] softmax处理后的seq_att_w
    tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1]): [batch_size, category_num, his_max, embedding]
    item_emb: [batch_size, category_num, num_head, dim]
    category_embedding_mat: [batch_size*category, num_head, dim]
    seq:　[batch_size*num_head,category_num, dim]
    '''
    def sequence_encode_cpt(self, items, nbr_mask):
        '''
        items: [batch_size, hist_max, embedding_dim]     每行表示一个用户的序列行为编码，用户历史行为emb
        nbr_mask: [None, seq_len]
        item_emb_input(Xu): [batch_size * his_max, embedding_dim]      经过reshpe函数展开，用户的历史行为emb
        self.seq_cate_dist(item_emb_input): [batch_size*his_max, category_num]     用户历史行为类别
        self.cate_dist：[batch_size, category_num, hist_max]     用户历史行为类别
        item_list_emb: [batch_size, his_max, embedding]      同items，用户历史行为emb
        self.position_embedding：[1, hist_max, embedding_dim]
        tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])：[batch_size, hist_max, embedding_dim]
        item_list_add_pos：[batch_size, hist_max, embedding_dim]  包含位置信息的用户行为序列编码
        item_hiddem: [batch_size, hist_max, hidden_units] 隐藏层输出
        item_att_w: [batch_size, hist_max, nums_heads * category_num]
        item_att_w: [batch_size,  num_heads * category_num, hist_max]
        item_att_w: [batch_size, category_num, num_heads, hist_max]
        category_mask_tile: [batch_size, category_num, num_heads, hist_max]
        seq_att_w: [batch_size, category_num*num_heads, hist_max]
        atten_mask: [None, category_num*num_heads ,seq_len]
        paddings: [None, category_num*num_heads ,seq_len]
        seq_att_w: [None, category_num*num_heads ,seq_len]
        seq_att_w: [batch_size, category_num, num_head, hist_max]
        seq_att_w: [batch_size, category_num, num_head, hist_max] softmax处理后的seq_att_w
        tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1]): [batch_size, category_num, his_max, embedding]
        item_emb: [batch_size, category_num, num_head, dim]
        category_embedding_mat: [batch_size*category, num_head, dim]
        seq:　[batch_size*num_head,category_num, dim]
            '''
        item_emb_input = tf.reshape(items, [-1, self.dim])
        # self.cate_dist = tf.reshape(self.seq_cate_dist(self.item_emb), [-1, self.category_num, self.hist_max])
        self.cate_dist = tf.transpose(tf.reshape(self.seq_cate_dist(item_emb_input), [-1, self.hist_max, self.category_num]), [0, 2, 1])
        item_list_emb = tf.reshape(item_emb_input, [-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(item_hidden, self.num_heads * self.category_num, activation=None, name='fc2')

            item_att_w = tf.transpose(item_att_w, [0, 2, 1]) #[batch_size, category_num*num_head, hist_max]

            item_att_w = tf.reshape(item_att_w, [-1, self.category_num, self.num_heads, self.hist_max]) #[batch_size, category_num, num_head, hist_max]

            category_mask_tile = tf.tile(tf.expand_dims(self.cate_dist, axis=2), [1, 1, self.num_heads, 1]) #[batch_size, category_num, num_head, hist_max]
            # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)

            # multiply(a,b) 两个矩阵中对应元素各自相乘，a和b为两个矩阵，或者两个数，或者数和矩阵
            seq_att_w = tf.reshape(tf.multiply(item_att_w, category_mask_tile), [-1, self.category_num * self.num_heads, self.hist_max])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.category_num * self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)
            # tf.equal(x, y)    逐元素判断x和y中的元素是否相等，如果相等就是True，不相等就是false
            # tf.where(condition, x, y)   condition中取值为True的元素替换为x中的元素，为False的元素替换为y中对应的元素
            seq_att_w = tf.where(tf.equal(atten_mask, 0), paddings, seq_att_w)
            seq_att_w = tf.reshape(seq_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            seq_att_w = tf.nn.softmax(seq_att_w)

            # here use item_list_emb or item_list_add_pos, that is a question
            item_emb = tf.matmul(seq_att_w, tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1])) #[batch_size, category_num, num_head, dim]

            category_embedding_mat = tf.reshape(item_emb, [-1, self.num_heads, self.dim]) #[batch_size, category_num, dim]
            if self.num_heads != 1:
                mu = tf.reduce_mean(category_embedding_mat, axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.dim, name='maha')
                wg = tf.matmul(category_embedding_mat, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1) = [N,H,1]
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]

                # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
                seq = tf.reduce_sum(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            else:
                seq = category_embedding_mat
            self.category_embedding_mat = seq
            seq = tf.reshape(seq, [-1, self.category_num, self.dim])

        return seq
