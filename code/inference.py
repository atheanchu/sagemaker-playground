import os
import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):
    tokenizer_init = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    compiled_model = os.path.exists(f'{model_dir}/model.pt')
    if compiled_model:
        os.environ["NEURON_RT_NUM_CORES"] = "4"
        model = torch.jit.load(f'{model_dir}/model.pt')
        # model = NeuronModelForSentenceTransformers.from_pretrained(model_dir).to(device)
    else: 
        model = AutoModel.from_pretrained(model_dir).to(device)
    
    return (model, tokenizer_init)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    

def predict_fn(input_data, models):

    model_bert, tokenizer = models
    sequence_0, sequence_1 = input_data
    max_length=1024

    tokenized_sequence_pair = tokenizer.encode_plus(sequence_0,
                                                    sequence_1,
                                                    max_length=max_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt').to(device)
    
    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs = tokenized_sequence_pair['input_ids'], tokenized_sequence_pair['attention_mask']
    
    with torch.no_grad():
        outputs = model_bert(*example_inputs)

    
    return str(outputs['pooler_output'])


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    
