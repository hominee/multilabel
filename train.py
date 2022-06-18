# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

import re
from converter.translate.langconv import Converter
from util import *

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

EPOCHS = 3
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
TRAIN_DATA_SIZE = 0 # estimate the training data set size
NUM_TRAIN_STEPS = TRAIN_DATA_SIZE * EPOCHS
NUM_WARMUP_STEPS = int(EPOCHS * TRAIN_DATA_SIZE * 0.1 / BATCH_SIZE)
FILE_TRAIN = "path/to/training/file"

OUTPUT_DIR = "output/path"
BERT_VOCAB= 'path/to/vocab.txt'
BERT_INIT_CHKPNT = 'path/to/bert_model.ckpt.index'
BERT_CONFIG = 'path/to/bert_config.json'
LABEL_COLUMNS = ['tag1','tag2','tag3'] # your label here

LEARNING_RATE = 2e-5
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500


# implement your function here
# to yield a example 
def load_example():
    actor = Preprocess()
    for (i, ln) in enumerate(open("path/to/train/file")) :
        (label, text) = ln.split("\t")
        text = actor.process_text(text)
        yield InputExample(guid=i, text_a=text,label=label)

class Preprocess(object):
    def __init__(self, max_seq_length=256,process_text_fn=bert.tokenization.convert_to_unicode,):
        self.max_seq_length = max_seq_length
        self.process_text_fn = process_text_fn

    def rm_text(self, text):
        NON_BMP_RE = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)
        text = NON_BMP_RE.sub(u'', text)
        text=Converter('zh-hans').convert(text)
        text = re.sub(r"<img.*?>", "",text)
        text = re.sub(r"http(s)?://[0-9A-z/%\:\.\-\+]+", "",text)
        text = re.sub(r"={3,}", "",text)
        text = re.sub(r"~{3,}", "",text)
        text = re.sub(r">{3,}", "",text)
        text = re.sub(r"<{3,}", "",text)
        text = re.sub(r"\+{3,}", "",text)
        text = re.sub(r"\*{3,}", "",text)
        text = re.sub(r"\-{3,}", "",text)
        text = re.sub(r"―{3,}", "",text)
        text = re.sub(r"～{3,}", "",text)
        text = re.sub(r"…{3,}", "",text)
        text = text.strip()
        return text

    def break_point(self, text, max_seq_length):
        max_seq_length = max_seq_length - 2
        # higher level
        high = 0
        low = 0
        ind = 0
        while ind < len(text):
            if text[ind] in ['\n', '?', '？', '。', '!', '！']:
                high = ind
            if text[ind] in [';', '；', '，', ',', '：', ' ', '　' ]:
                low = ind
            ind += 1
        if high and high > 0.35*max_seq_length:
                return high + 1
        if low and low > 0.35*max_seq_length:
            return low + 1
        return ind + 1

    def process_text(self, text, max_seq_length=256):
        datas = []
        text = self.rm_text(text)
        if len(text) < 20 :
            return datas
        ## tokenize the long text into many sentences
        if len(text) < max_seq_length :
            # print('got a whole： ', text)
            datas.append(text)
            return datas
        #######
        ## \n ? 。!！has higher priori to ; ；，,
        #######
        index = 0
        while len(text[index:]) > max_seq_length:
            edge = text[index+max_seq_length]
            slc = text[index:index+max_seq_length]
            if edge in ['\n', '?', '？', '。', '!', '！', ';', '；', '，', ',', ')', '）', '`', ']', '」', '}', '》']:
                ## edge is a trailing breaking point
                # print('got： ', slc)
                datas.append(slc)
                index += max_seq_length+1
                continue
            ind = self.break_point(slc, max_seq_length)
            # print('got: ', slc[:ind])
            datas.append(slc[:ind])
            index += ind
        if len(text[index:]) > 0.35 * max_seq_length:
            datas.append(text[index:])
        return datas

    def create_examples(self, _id, text_a, label):
        """Creates examples for the training/dev/test sets."""
        guid = _id
        text_a = self.process_text_fn(text_a)
        text_b = None
        # construct an example
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


def get_dataset(load_example, seq_length, batch_size, is_training):

    tokenizer = tokenization.FullTokenizer(
        vocab_file = BERT_VOCAB,
        do_lower_case = True
    )


    def create_classifier_dataset():
        for example in load_example():
            """Creates input dataset from (tf)records files for train/eval."""
            feature = convert_single_example(example, seq_length, tokenizer)
            record = select_data_from_record(feature) 
            yield record


    dataset = tf.data.Dataset.from_generator( create_classifier_dataset, output_types=({"input_word_ids": tf.int32, "input_mask": tf.int32, "input_type_ids": tf.int32}, tf.int32 ), output_shapes=({'input_word_ids':  [seq_length], 'input_mask': [seq_length], 'input_type_ids': [seq_length]}, [1,len(LABEL_COLUMNS)]) )

    if is_training:
        dataset = dataset.shuffle(3000)
        dataset = dataset.repeat(EPOCHS)

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        # probabilities = tf.nn.softmax(logits, axis=-1) ### multiclass case
        probabilities = tf.nn.sigmoid(logits)#### multi-label case

        labels = tf.cast(labels, tf.float32)
        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        # probabilities = tf.nn.softmax(logits, axis=-1)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        #
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        #tf.logging.info("*** Features ***")
        #for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        print(features.keys())
        input_ids = features["input_word_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["input_type_ids"]
        label_ids = labels
        is_real_example = True
        # if "is_real_example" in features:
             # is_real_example = tf.cast(features.is_real_example, dtype=tf.float32)
        # else:
             # is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                logits_split = tf.split(probabilities, num_labels, axis=-1)
                label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                return eval_dict

                ## original eval metrics
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # accuracy = tf.metrics.accuracy(
                #     labels=label_ids, predictions=predictions, weights=is_real_example)
                # loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                # return {
                #     "eval_accuracy": accuracy,
                #     "eval_loss": loss,
                # }

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            print("mode:", mode,"probabilities:", probabilities)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold=scaffold_fn)
        return output_spec

    return model_fn

def input_fn_db():
    return get_dataset(FILE_TRAIN, MAX_SEQ_LENGTH, batch_size, True) 

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    keep_checkpoint_max=1,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
model_fn = model_fn_builder(
  bert_config=bert_config,
  num_labels= len(LABEL_COLUMNS),
  init_checkpoint=BERT_INIT_CHKPNT,
  learning_rate=LEARNING_RATE,
  num_train_steps=NUM_TRAIN_STEPS,
  num_warmup_steps=NUM_WARMUP_STEPS,
  use_tpu=False,
  use_one_hot_embeddings=False)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})


estimator.train(input_fn=input_fn_db, max_steps=NUM_TRAIN_STEPS)
