# 数据处理过程中，则通过以下代码读取数据：
with open(data_path, 'r') as f:
    for ind, line in enumerate(f):
        line_num += 1
        slot_tag_line, class_name = line.strip('\n\r').split(' <=> ')
        if slot_tag_line == "":
            continue
        in_seq, tag_seq = [], []
        for item in slot_tag_line.split(' '):
            tmp = item.split(separator)
            assert len(tmp) >= 2
            word, tag = separator.join(tmp[:-1]), tmp[-1]
            if lowercase:
                word = word.lower()
            in_seq.append(word2idx[word] if word in word2idx else word2idx['<unk>'])
            tag_seq.append(tag2idx[tag] if tag in tag2idx else (tag2idx['<unk>'], tag))
        if keep_order:
            in_seq.append(line_num)
        input_seqs.append(in_seq)
        tag_seqs.append(tag_seq)
        if multiClass:
            if class_name == '':
                class_labels.append([])
            else:
                class_labels.append([class2idx[val] for val in class_name.split(';')])
        else:
            if ';' not in class_name:
                class_labels.append(class2idx[class_name])
            else:
                class_labels.append((class2idx[class_name.split(';')[0]], class_name.split(';')))

#########################################################################################################
# 本模型是分别用LSTM-atten-crf 模型和LSTM 对槽填充和意图理解任务进行建模，并联合训练。
optimizer.zero_grad()
print("slot filling: CRF layer")
max_len = max(lens)
masks = [([1] * l) + ([0] * (max_len - l)) for l in lens]
masks = torch.tensor(masks, dtype=torch.uint8, device=opt.device)
lstm_out, packed_h_t_c_t, lengths= model_tag._get_lstm_features(inputs, lens)
crf_feats, encoder_info, slot_att = model_tag.attention_net(lstm_out, packed_h_t_c_t, lengths,with_snt_classifier=True)
tag_loss = model_tag.neg_log_likelihood(crf_feats, masks, tags)
class_scores = model_class(encoder_info_filter(encoder_info),slot_att)
class_loss = class_loss_function(class_scores, classes)
losses.append([tag_loss.item()/sum(lens), class_loss.item()/len(lens)])
total_loss = opt.st_weight * tag_loss + (1 - opt.st_weight) * class_loss
total_loss.backward()

# Clips gradient norm of an iterable of parameters.
 if opt.max_norm > 0:
    torch.nn.utils.clip_grad_norm_(params, opt.max_norm)
optimizer.step()

#########################################################################################################
# 模型预测
# 槽填充F1 值预测
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
