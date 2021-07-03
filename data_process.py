# 数据处理
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
