# 模型训练

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