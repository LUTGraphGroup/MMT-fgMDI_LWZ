from __future__ import division
from __future__ import print_function
import time
import argparse
from model import *
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# training parameters
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--attention_dropout', type=float, default=0.1,
                    help='Dropout in the attention layer')

# model parameters
parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')
parser.add_argument('--hops', type=int, default=2,
                        help='Hop of neighbors to be calculated')
parser.add_argument('--graphformer_layers', type=int, default=2,
                    help='number of Graphormer layers')
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of attention heads in graphormer.')
parser.add_argument('--node_input', type=int, default=64,
                    help='input dimensions of node features/PCA.')
parser.add_argument('--node_hidden', type=int, default=32,
                    help='hidden dimensions of node features.')
parser.add_argument('--node_output', type=int, default=64,
                    help='output dimensions of node features.')
parser.add_argument('--ffn_dim', type=int, default=256,
                        help='FFN layer size')
parser.add_argument('--num_relations', type=int, default=6,
                    help='number of relations.')

args = parser.parse_args()
print('args', args)

Dr_M, Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_hops, Dr_G_hops, \
M_Di_hops, M_G_hops, label_matrix, label1_index, label2_index, k_folds = load_data(args.seed, args.node_input, args.pe_dim, args.hops)

acc_result = []

cumulative_cm = np.zeros((3, 3), dtype=int)
micro_fpr_result = []
micro_tpr_result = []
macro_fpr_result = []
macro_tpr_result = []

pre_result = {
                'pre_micro': [],
                'pre_macro': [],
                }
reca_result = {
                'recall_micro': [],
                'recall_macro': [],
                }
f1_result = {
                'f1_micro': [],
                'f1_macro': [],
                }

ma_auc_result = []
mi_auc_result = []
ma_aupr_result = []
mi_aupr_result = []
micro_recall_result= []
micro_pre_result = []
macro_recall_result = []
macro_pre_result = []

tsne_scores_result = []
tsne_labels_result = []


print("seed=%d, evaluating metabolite-disease...." % args.seed)
for k in range(k_folds):
    print("------this is %dth cross validation------" % (k + 1))
    Dr_M_matrix = np.matrix(Dr_M, copy=True)
    val_pos_edge_index = np.array(label1_index[k]+label2_index[k]).T
    val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long).to(device)

    val_neg_edge_index = np.mat(np.where(label_matrix < 1)).T.tolist()
    random.seed(args.seed)
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = val_neg_edge_index[:len(label2_index[k])]
    val_neg_edge_index = np.array(val_neg_edge_index).T
    val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).to(device)

    Dr_M_matrix[tuple(np.array(label1_index[k]+label2_index[k]).T)] = 0
    train_pos_edge_index = np.mat(np.where(Dr_M_matrix > 0))
    train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long).to(device)

    model = TransformerModel(hops=args.hops,
                             output_dim=args.node_output,
                             input_dim=Dr_Di_hops.shape[2],
                             num_drug=Dr_M.shape[0],
                             num_meta=Dr_M.shape[1],
                             graphformer_layers=args.graphformer_layers,
                             num_heads=args.n_heads,
                             hidden_dim=args.node_hidden,
                             ffn_dim=args.ffn_dim,
                             dropout_rate=args.dropout,
                             attention_dropout_rate=args.attention_dropout,
                             ).to(device)
    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = F.cross_entropy

    best_acc = 0
    best_cm = 0
    best_epoch = 0
    best_ma_auc = 0
    best_mi_auc = 0
    best_ma_aupr = 0
    best_mi_aupr = 0
    best_tpr = 0
    best_fpr = 0
    best_recall = 0
    best_precision = 0
    best_micro_fpr = 0
    best_micro_tpr = 0
    best_macro_fpr = 0
    best_macro_tpr = 0
    best_micro_recall = 0
    best_micro_pre = 0
    best_macro_recall = 0
    best_macro_pre = 0
    best_val_label = 0
    best_val_score = 0
    best_val_feature = 0

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad()

        train_neg_edge_index = np.mat(np.where(Dr_M_matrix < 1)).T.tolist()
        random.shuffle(train_neg_edge_index)
        train_neg_edge_index = train_neg_edge_index[:np.count_nonzero(label_matrix==2)-len(label2_index[k])]
        train_neg_edge_index = np.array(train_neg_edge_index).T
        train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)

        train_edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], 1)
        val_edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)

        output = model(Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_hops, Dr_G_hops, M_Di_hops, M_G_hops, train_edge_index, val_edge_index, args.hops)

        train_scores = output[: train_edge_index.shape[1]]
        train_scores = F.softmax(train_scores, dim=1)
        label_matrix_tensor = torch.tensor(label_matrix).to(device)
        train_labels = label_matrix_tensor[train_edge_index[0], train_edge_index[1]].to(device)

        loss_train = criterion(train_scores, train_labels.long()).to(device)
        loss_train.backward(retain_graph=True)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            score_train_cpu = train_scores.detach().cpu().numpy()
            label_train_cpu = train_labels.detach().cpu().numpy()
            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu, multi_class='ovr')

            val_scores = output[train_edge_index.shape[1]:]
            val_scores = F.softmax(val_scores, dim=1)
            val_scores_cpu = val_scores.detach().cpu().numpy()

            val_labels = label_matrix_tensor[val_edge_index[0], val_edge_index[1]].to(device)
            val_labels_cpu = val_labels.detach().cpu().numpy()
            ma_val_auc = round(metrics.roc_auc_score(val_labels_cpu, val_scores_cpu, multi_class='ovr', average="macro"), 4)
            mi_val_auc = round(metrics.roc_auc_score(val_labels_cpu, val_scores_cpu, multi_class='ovr', average="micro"), 4)

            val_scores_1 = val_scores.max(1)[1].detach().cpu().numpy()
            val_acc = round(accuracy_score(val_labels_cpu, val_scores_1), 4)

            cm = confusion_matrix(val_labels_cpu, val_scores_1)

            val_bin = label_binarize(val_labels_cpu, classes=[0, 1, 2])
            mi_val_aupr = round(metrics.average_precision_score(val_bin, val_scores_cpu, average="micro"), 4)
            ma_val_aupr = round(metrics.average_precision_score(val_bin, val_scores_cpu, average="macro"), 4)

            val_result = classification_report(val_labels_cpu, val_scores_1)
            precision_score_average_micro = precision_score(val_labels_cpu, val_scores_1, average='micro')
            precision_score_average_macro = precision_score(val_labels_cpu, val_scores_1, average='macro')
            precision_scores = {

                'pre_micro': round(precision_score_average_micro, 4),
                'pre_macro': round(precision_score_average_macro, 4)

            }

            recall_score_average_micro = recall_score(val_labels_cpu, val_scores_1, average='micro')
            recall_score_average_macro = recall_score(val_labels_cpu, val_scores_1, average='macro')

            recall_scores = {

                'recall_micro': round(recall_score_average_micro, 4),
                'recall_macro': round(recall_score_average_macro, 4)
            }

            f1_score_average_micro = f1_score(val_labels_cpu, val_scores_1, average='micro')
            f1_score_average_macro = f1_score(val_labels_cpu, val_scores_1, average='macro')
            f1_scores = {

                'f1_micro': round(f1_score_average_micro, 4),
                'f1_macro': round(f1_score_average_macro, 4)
            }

            end = time.time()
            # if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(), 'Train AUC: %.4f' % train_auc,
                  'val acc: %.4f' % val_acc,
                  'ma AUC: %.4f' % ma_val_auc,
                  'mi AUC: %.4f' % mi_val_auc,
                  'ma AUPR: %.4f' % ma_val_aupr,
                  'mi AUPR: %.4f' % mi_val_aupr,
                  'ma pre: %.4f' % precision_score_average_macro,
                  'ma recall: %.4f' % recall_score_average_macro,
                  'ma f1: %.4f' % f1_score_average_macro,
                  'Time: %.2f' % (end - start))

            fpr = dict()
            tpr = dict()
            val_auc = dict()
            precision = dict()
            recall = dict()
            val_aupr = dict()

            n_classes = val_bin.shape[1]
            for i in range(n_classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(val_bin[:, i], val_scores_cpu[:, i])
                val_auc[i] = metrics.auc(fpr[i], tpr[i])
                precision[i], recall[i], _ = metrics.precision_recall_curve(val_bin[:, i], val_scores_cpu[:, i])
                val_aupr[i] = metrics.auc(recall[i], precision[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = metrics.roc_curve(val_bin.ravel(), val_scores_cpu.ravel())
            val_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

            # Compute micro-average PR curve and PR area
            precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(val_bin.ravel(), val_scores_cpu.ravel())

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            val_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

            # Compute macro-average PR curve and PR area
            # First aggregate all recall
            all_recall = np.unique(
                np.concatenate([recall[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_pre = np.zeros_like(all_recall)
            for i in range(n_classes):
                mean_pre += np.interp(1-all_recall, 1-recall[i], precision[i])
            # Finally average it and compute AUC
            mean_pre /= n_classes
            recall["macro"] = all_recall
            precision["macro"] = mean_pre

            if val_acc > best_acc and ma_val_auc > best_ma_auc and ma_val_aupr > best_ma_aupr:
                best_acc = val_acc
                best_pre = precision_scores
                best_rec = recall_scores
                best_f1 = f1_scores
                best_ma_auc = ma_val_auc
                best_mi_auc = mi_val_auc
                best_ma_aupr = ma_val_aupr
                best_mi_aupr = mi_val_aupr
                best_epoch = epoch + 1
                best_cm = cm
                best_micro_fpr = fpr["micro"]
                best_micro_tpr = tpr["micro"]
                best_macro_fpr = fpr["macro"]
                best_macro_tpr = tpr["macro"]

                best_micro_recall = recall["micro"]
                best_micro_pre = precision["micro"]
                best_macro_recall = recall["macro"]
                best_macro_pre = precision["macro"]

                best_val_score = val_scores_cpu
                best_val_label = val_labels_cpu


    print('Fold:', k + 1, 'Best Epoch:', best_epoch,
              'best_acc: %.4f' % best_acc,
              'best_pre: %.4f' % best_pre['pre_macro'],
              'best_rec: %.4f' % best_rec['recall_macro'],
              'best_f1: %.4f' % best_f1['f1_macro'],
              'best_ma_auc: %.4f' % best_ma_auc,
              'best_mi_auc: %.4f' % best_mi_auc,
              'best_ma_aupr: %.4f' % best_ma_aupr,
              'best_mi_aupr: %.4f' % best_mi_aupr
              )

    for key, value in best_pre.items():
        if key in pre_result:
            pre_result[key].append(value)
        else:
            pre_result[key] = value

    for key, value in best_rec.items():
        if key in reca_result:
            reca_result[key].append(value)
        else:
            reca_result[key] = value

    for key, value in best_f1.items():
        if key in f1_result:
            f1_result[key].append(value)
        else:
            f1_result[key] = value

    acc_result.append(best_acc)
    cumulative_cm += best_cm
    average_cm = cumulative_cm / k_folds
    ma_auc_result.append(best_ma_auc)
    mi_auc_result.append(best_mi_auc)
    ma_aupr_result.append(best_ma_aupr)
    mi_aupr_result.append(best_mi_aupr)

    micro_fpr_result.append(best_micro_fpr)
    micro_tpr_result.append(best_micro_tpr)
    macro_fpr_result.append(best_macro_fpr)
    macro_tpr_result.append(best_macro_tpr)

    micro_recall_result.append(best_micro_recall)
    micro_pre_result.append(best_micro_pre)
    macro_recall_result.append(best_macro_recall)
    macro_pre_result.append(best_macro_pre)

    tsne_scores_result.append(best_val_score)
    tsne_labels_result.append(best_val_label)

tsne_scores_result = np.vstack(tsne_scores_result)
tsne_labels_result = np.concatenate(tsne_labels_result)

pd.DataFrame(tsne_scores_result).to_csv('../result/tsne_scores.csv', index=False)
pd.DataFrame(tsne_labels_result).to_csv('../result/tsne_labels.csv', index=False)

pd.DataFrame(micro_fpr_result).to_csv('../result/micro_fpr.csv', index=False)
pd.DataFrame(micro_tpr_result).to_csv('../result/micro_tpr.csv', index=False)
pd.DataFrame(macro_fpr_result).to_csv('../result/macro_fpr.csv', index=False)
pd.DataFrame(macro_tpr_result).to_csv('../result/macro_tpr.csv', index=False)

pd.DataFrame(micro_recall_result).to_csv('../result/micro_recall.csv', index=False)
pd.DataFrame(micro_pre_result).to_csv('../result/micro_pre.csv', index=False)
pd.DataFrame(macro_recall_result).to_csv('../result/macro_recall.csv', index=False)
pd.DataFrame(macro_pre_result).to_csv('../result/macro_pre.csv', index=False)

averages_pre = {key: round(np.mean(values), 4) for key, values in pre_result.items()}
pre_std_devs = {key: round(np.std(values), 4) for key, values in pre_result.items()}
# print('pre_result', pre_result)
# print('averages_pre', averages_pre)
# print('pre_std_devs', pre_std_devs)

averages_recall = {key: round(np.mean(values), 4) for key, values in reca_result.items()}
recall_std_devs = {key: round(np.std(values), 4) for key, values in reca_result.items()}
# print('reca_result', reca_result)
# print('averages_recall', averages_recall)
# print('recall_std_devs', recall_std_devs)

averages_f1 = {key: round(np.mean(values), 4) for key, values in f1_result.items()}
f1_std_devs = {key: round(np.std(values), 4) for key, values in f1_result.items()}
# print('f1_result', f1_result)
# print('averages_f1', averages_f1)
# print('f1_std_devs', f1_std_devs)


print('## Training Finished !')
print('-----------------------------------------------------------------------------------------------')
print('acc: %s, mean: %.4f, variance: %.4f \n' % (acc_result, np.mean(acc_result), np.std(acc_result)))

print("Average Confusion Matrix:\n", average_cm)

print('macro_auc: %s, mean: %.4f, variance: %.4f \n' % (ma_auc_result, np.mean(ma_auc_result), np.std(ma_auc_result)))
print('micro_auc: %s, mean: %.4f, variance: %.4f \n' % (mi_auc_result, np.mean(mi_auc_result), np.std(mi_auc_result)))

print('macro_aupr: %s, mean: %.4f, variance: %.4f \n' % (ma_aupr_result, np.mean(ma_aupr_result), np.std(ma_aupr_result)))
print('micro_aupr: %s, mean: %.4f, variance: %.4f \n' % (mi_aupr_result, np.mean(mi_aupr_result), np.std(mi_aupr_result)))

print('micro_pre: %s, mean: %.4f, variance: %.4f \n' % (pre_result['pre_micro'], averages_pre['pre_micro'], pre_std_devs['pre_micro']))
print('macro_pre: %s, mean: %.4f, variance: %.4f \n' % (pre_result['pre_macro'], averages_pre['pre_macro'], pre_std_devs['pre_macro']))


print('micro_recall: %s, mean: %.4f, variance: %.4f \n' % (reca_result['recall_micro'], averages_recall['recall_micro'], recall_std_devs['recall_micro']))
print('macro_recall: %s, mean: %.4f, variance: %.4f \n' % (reca_result['recall_macro'], averages_recall['recall_macro'], recall_std_devs['recall_macro']))


print('micro_f1: %s, mean: %.4f, variance: %.4f \n' % (f1_result['f1_micro'], averages_f1['f1_micro'], f1_std_devs['f1_micro']))
print('macro_f1: %s, mean: %.4f, variance: %.4f \n' % (f1_result['f1_macro'], averages_f1['f1_macro'], f1_std_devs['f1_macro']))

"""SNE Visualization"""
plot_T_SNE(tsne_scores_result, tsne_labels_result, directory='../result', name='T_SNE')

# Confusion Matrix Visualization
plot_cm(average_cm, directory='../result', name='average_cm')

