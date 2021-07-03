# 模型测试

for idx, pred_line in enumerate(top_pred_slots):
    length = lens[idx]
    pred_seq = [idx_to_tag[tag] for tag in pred_line][:length]
    lab_seq = [idx_to_tag[tag] if type(tag) == int else tag for tag in raw_tags[idx]]
    pred_chunks = acc.get_chunks(['O']+pred_seq+['O'])
    label_chunks = acc.get_chunks(['O']+lab_seq+['O'])
    for pred_chunk in pred_chunks:
        if pred_chunk in label_chunks:
            TP += 1
        else:
            FP += 1
    for label_chunk in label_chunks:
        if label_chunk not in pred_chunks:
            FN += 1

#########################################################################################################
# 下图a 和 f 分别表示槽填充任务的准确率和F1 值，我们往往只取F1 值表示该模型的性能。
if TP == 0:
        p, r, a, f = 0, 0, 0, 0
    else:
        p, r, a, f = 100*TP/(TP+FP), 100*TP/(TP+FN), 100*(TP+TN)/(TP+FP+FN+TN),100*2*TP/(2*TP+FN+FP)
        
#########################################################################################################
# 意图识别准确率预测
pred_classes = [idx_to_class[i] for i,p in enumerate(snt_probs[idx]) if p > 0.5]
gold_classes = [idx_to_class[i] for i in raw_classes[idx]]
for pred_class in pred_classes:
    if pred_class in gold_classes:
        TP2 += 1
    else:
        FP2 += 1
for gold_class in gold_classes:
    if gold_class not in pred_classes:
        FN2 += 1
