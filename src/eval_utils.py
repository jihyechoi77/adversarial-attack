from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#plt.switch_backend('Agg')
import numpy as np
import scipy

# reference about using legend: http://jb-blog.readthedocs.io/en/latest/posts/0012-matplotlib-legend-outdide-plot.html

def prepare_figure(fid):
    plt.figure(fid)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    if fid is None:
        fid = plt.gcf().number
        return fid


def plot_roc_general(truth, pred, savefig):
    # truth and pred: N by 1 array
    fpr, tpr, _ = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)

    prepare_figure(fid=None)
    plt.plot(fpr, tpr, label="ROC face verifier (AUC = %0.3f" % roc_auc)
    plt.title('ROC curve of face verifier trained on the easy set')
    plt.legend(loc='best')
    plt.savefig(savefig)


fid_VGG16s = prepare_figure(fid=None)
art_all = []
fontP = FontProperties()
fontP.set_size('small')
def plot_roc(modeltype, attribute, truth, pred):
    # truth and pred: N by 1 array
    fpr, tpr, _ = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)

    """
    prepare_figure(fid=None)
    plt.plot(fpr, tpr, label="ROC %s (AUC = %0.3f)" % (attribute, roc_auc))
    plt.title("ROC curve of %s attribute classifier - %s" % (modeltype, attribute))
    plt.legend(loc='best')
    # plt.savefig("/data/jihyec/vgg_keras_tf/trained_models/attr_classifiers_tuned/figures/%s_%s.png" % (modeltype, attribute))  # for grey4
    plt.savefig("/data/jihyec/vgg_keras_tf/trained_models/figures/%s_%s.png" % (modeltype, attribute))  # for grey6

    prepare_figure(fid=None)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr, tpr, label="ROC %s (AUC = %0.3f)" % (attribute, roc_auc))
    plt.title("ROC curve (zoomed in at top left) of %s attribute classifier - %s" % (modeltype, attribute))
    plt.legend(loc='best')
    # plt.show()
    # plt.savefig("/data/jihyec/vgg_keras_tf/trained_models/attr_classifiers_tuned/figures/%s_%s_zoomed.png" % (modeltype, attribute)) # for grey4
    plt.savefig("/data/jihyec/vgg_keras_tf/trained_models/figures/%s_%s_zoomed.png" % (modeltype, attribute))  # for grey4
    """

    plt.figure(fid_VGG16s)
    fontP = FontProperties()
    fontP.set_size('small')
    plt.plot(fpr, tpr, label="ROC %s (AUC = %0.3f)" % (attribute, roc_auc))
    plt.title("ROC curve of 40 %s attribute classifiers" % modeltype)
    lgd_all = plt.legend(prop=fontP, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    art_all.append(lgd_all)
    # plt.savefig("/data/jihyec/vgg_keras_tf/trained_models/attr_classifiers_tuned/figures/%s_all.png" % modeltype, bbox_inches="tight") # for grey4
    # plt.savefig("/data/jihyec/vgg_keras_tf/trained_models/figures/%s_all.png" % modeltype,
    #             additional_artists=art_all, bbox_inches="tight")  # for grey6
    plt.savefig("/home/jihyec/adversarial-ml/vgg_keras_tf/attribute_classifiers_inputdim160/%s/%s_roc_all.png" % (modeltype, modeltype), bbox_inches="tight") # for devbox

    return roc_auc


def plot_roc_multiple(attributes, truth, pred, savefig):
    # truth and pred: N by 40 array
    num_attr = len(attributes)
    roc_auc = []

    fid1 = prepare_figure(fid=None)
    fid2 = prepare_figure(fid=None)

    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)

    art3 = []
    art4 = []
    for i in range(0, num_attr):
        plt.figure(fid1)
        fpr, tpr, _ = roc_curve(truth[:, i], pred[:, i])
        roc_auc = np.append(roc_auc, auc(fpr, tpr))
        plt.plot(fpr, tpr, label="%s (AUC = %0.3f)" % (attributes[i], roc_auc[-1]))
        plt.title('ROC curve of multilabel attribute classifier')
        lgd3 = plt.legend(prop=fontP, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        art3.append(lgd3)
        plt.savefig(savefig+'_roc.png', additional_artists=art3, bbox_inches="tight")

        plt.figure(fid2)
        plt.plot(fpr, tpr, label="ROC %s (AUC = %0.3f)" % (attributes[i], roc_auc[-1])) # , roc_auc[-1]
        plt.title('ROC curve (zoomed in at top left) of multilabel attribute classifier')
        # plt.legend(loc='best')
        lgd4 = plt.legend(prop=fontP, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        art4.append(lgd4)
        plt.savefig(savefig+'_roc_zoomed.png', additional_artists=art4, bbox_inches="tight")

    return roc_auc


def compute_accuracy(truth, pred):
    accuracy = np.sum(np.equal(truth, pred)) / len(truth)
    return accuracy


# def evaluate_multilab(modelname, preds, truth):
def evaluate_multilab(preds, truth, save_name):
    # reference: https://github.com/keras-team/keras/issues/5335

    attributes = np.array(
        ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',  # 0~5
         'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',  # 6~12
         'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',  # 13~19
         'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',  # 20~25
         'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',  # 26~31
         'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',  # 32~36
         'Wearing_Necklace', 'Wearing_Necktie', 'Young'])  # 37~39

    roc_auc = plot_roc_multiple(attributes=attributes, pred=preds, truth=truth,
                                # savefig="/data/jihyec/vgg_keras_tf/trained_models/figures/%s_notop_on_celeba" % modelname)
                                savefig=save_name)

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    accur = []  # per-class accuracy
    for i in range(0, 40):
        accur = np.append(accur, compute_accuracy(truth=truth[:, i], pred=preds[:, i]))

    # result_save_file = "/data/jihyec/vgg_keras_tf/trained_models/%s_notop_on_celeba_accuracy.mat" % modelname
    result_save_file = save_name + '_accuracy.mat'
    scipy.io.savemat(result_save_file, mdict={'accuracy': accur, 'auc': roc_auc, 'pred': preds, 'truth': truth})

    top_unbal_attr_idx = (np.array(roc_auc)).argsort()[:8]
    print('\n top 8 unbalanced attribute indices...')
    print(top_unbal_attr_idx)
    print('\n and their AUC values...')
    print(roc_auc[top_unbal_attr_idx])

    return accur, roc_auc, top_unbal_attr_idx







