from cgi import test
from datetime import datetime
import argparse
import json
import sys
from typing import final

sys.path.append("..")


from transformers import set_seed
import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time

# transformers.logging.set_verbosity_info()
set_seed(42)
print("start time: ", get_current_time())

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("-bs", "--batch-size", type=int, default=1)
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
parser.add_argument("-gcv", "--gradient-clip-val", type=float, default=0.0)
parser.add_argument("-wd", "--weight-decay", type=float, default=0)
parser.add_argument(
    "-mn",
    "--model-name",
    type=str,
    choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
    required=True,
)
parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
parser.add_argument("-md", "--model-dir", type=str, default="")
parser.add_argument("-et", "--error-type", type=str, default="")
parser.add_argument("-stl", "--save-total-limit", type=int, default=-1)
parser.add_argument("-pt", "--pre-trained", type=boolean_string, default=True)
args = parser.parse_args()

# Create job directory
# model_name = args.model_name
# if args.model_dir != "":
#     model_directory = args.model_dir
# else:
#     now = datetime.now()
#     dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
#     model_directory = "t5global" + "_" + dt_string
#     model_directory = model_name + "_global_" + dt_string

# os.makedirs(model_directory)
# with open(os.path.join(model_directory, "commandline_args.txt"), "w") as f:
#     f.write("\n".join(sys.argv[1:]))

# Read and prepare data
data = GetDataAsPython("/Users/miftasintaha/Downloads/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython("/Users/miftasintaha/Downloads/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint
all_warning_types = extract_warning_types(data)
if args.error_type != "":
    all_warning_types = [args.error_type]
# print(all_warning_types)
(
    train_inputs,
    train_labels,
    val_inputs,
    val_labels,
    test_inputs,
    test_labels,
    train_info,
    val_info,
    test_info,
) = create_data(data, all_warning_types, include_warning=True, model_name="t5-base")


print(type(train_inputs))
rule_ids = test_inputs.keys()
test_data = []
for k in rule_ids:
    for data in zip(test_inputs[k], test_labels[k]):
        input = data[0]
        output = data[1]
        [rule_id, message, warning_line, source_code] = input.split("%$%")
        test_data.append({
            'rule_id': rule_id,
            'evidence': message,
            'message': message,
            'source_code': source_code.strip(),
            'target_code': output.strip()
        })

train_data = []
for data in zip(train_inputs, train_labels):
    input = data[0]
    output = data[1]
    [rule_id, message, warning_line, source_code] = input.split("%$%")
    train_data.append({
        'rule_id': rule_id,
        'evidence': message,
        'message': message,
        'source_code': source_code.strip(),
        'target_code': output.strip()
    })

print('TRAIN', len(train_data), 'TEST', len(test_data))

with open('train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4)
with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)
