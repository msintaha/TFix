from collections import defaultdict

from sklearn.model_selection import train_test_split
import torch

def extract_warning_types(data):
    all_warnings = []
    for sample in data:
        if sample.linter_report.rule_id not in all_warnings:
            all_warnings.append(sample.linter_report.rule_id)
    return all_warnings

def filter_rule(data, rule_type):
    filtered_data = []
    for point in data:
        if point.linter_report.rule_id == rule_type:
            filtered_data.append(point)
    return filtered_data

def split_filtered(filtered_data, include_warning, model_name, seed=13):
    filtered_data_temp = filtered_data

    inputs = [data_point.GetT5Representation(include_warning)[0] for data_point in filtered_data]
    outputs = [data_point.GetT5Representation(include_warning)[1] for data_point in filtered_data_temp]

    test_size = 0.1 if len(inputs) >= 10 else 1 / len(inputs)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, outputs, shuffle=True, random_state=seed, test_size=test_size)

    train_info, test_info = train_test_split(filtered_data, shuffle=True, random_state=seed, test_size=test_size) 

    val_size = 0.1 if len(train_inputs) >= 10 else 1 / len(train_inputs)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, shuffle=True, random_state=seed, test_size=val_size)
        
    train_info, val_info = train_test_split(train_info, shuffle=True, random_state=seed, test_size=test_size) 

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info

def create_data(data, linter_warnings: list, include_warning, model_name):
    train, train_labels, val, val_labels = [], [], [], []
    test, test_labels = defaultdict(list), defaultdict(list)
    n_test_samples = 0

    train_info, val_info = [], []
    test_info = defaultdict(list)

    for warning in linter_warnings:
        filtered_data = filter_rule(data, warning)        
        train_w, train_w_labels, val_w, val_w_labels, test_w, test_w_labels, train_w_info, val_w_info, test_w_info = split_filtered(filtered_data, include_warning, model_name)

        train += train_w
        train_labels += train_w_labels

        val += val_w
        val_labels += val_w_labels

        train_info += train_w_info
        val_info += val_w_info

        test[warning] = test_w
        test_labels[warning] = test_w_labels

        test_info[warning] = test_w_info

        n_test_samples += len(test_w)
    print('train size: {}\nval size: {}\ntest size: {}'.format(len(train), len(val), n_test_samples))
    return train, train_labels, val, val_labels, test, test_labels, train_info, val_info, test_info

class BugFixDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.target_encodings = targets

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.target_encodings['input_ids'][index], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def create_dataset(inputs, labels, tokenizer, pad_truncate, max_length=None):
    if max_length is not None:
        input_encodings = tokenizer(inputs, truncation=pad_truncate, padding=pad_truncate, max_length=max_length)
        label_encodings = tokenizer(labels, truncation=pad_truncate, padding=pad_truncate, max_length=max_length)
    else:
        input_encodings = tokenizer(inputs, truncation=pad_truncate, padding=pad_truncate, max_length=256)
        label_encodings = tokenizer(labels, truncation=pad_truncate, padding=pad_truncate, max_length=256)

    dataset = BugFixDataset(input_encodings, label_encodings)
    return dataset
