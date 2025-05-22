
import torch
from minicheck.minicheck import MiniCheck
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


class PassageUtility:
    def __init__(self, args, type='nli'):
        self.args = args
        self.type = type
        if type == 'minicheck':
            model_name = 'Bespoke-MiniCheck-7B'
            self.model = MiniCheck(model_name=model_name, enable_prefix_caching=False)
    
        elif type == 'nli':
            # - Labels: {0: Contradiction, 1: Neutral, 2: Entailment}
            model_name = "microsoft/deberta-large-mnli"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(args.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()
    
    def __call__(self, context, question, generated_text):
        
        generated_text_ = f"{question} {generated_text}"
        context_text = ''
        if len(context) > 0:
            for idx, doc_item in enumerate(context):
                content = doc_item['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                context_text += f"Doc {idx+1} (Title: {title}) {text}\n"
        else:
            return (0, 0.0)
        
        
        if self.type == 'minicheck':
            pred_label, raw_prob, _, _ = self.model.score(docs=[context_text], claims=[generated_text_])
            return (pred_label[0], raw_prob[0])
        
        elif self.type == 'nli':
            input = context_text + ' [SEP] ' + generated_text_
            reverse_input = generated_text_ + ' [SEP] ' + context_text
            
            with torch.no_grad():
                encoded_input = self.tokenizer.encode(input, padding=True)
                encoded_reverse_input = self.tokenizer.encode(reverse_input, padding=True)
                
                prediction = self.model(torch.tensor(torch.tensor([encoded_input]), device=self.args.device))['logits']
                predicted_label = torch.argmax(prediction, dim=1)
                
                reverse_prediction = self.model(torch.tensor(torch.tensor([encoded_reverse_input]), device=self.args.device))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
            
            # === Get label
            nli_label = 0 if (0 in predicted_label or 0 in reverse_predicted_label) else 2
            prediction_dist = torch.softmax(prediction, dim=1).tolist()[0]
            reverse_prediction_dist = torch.softmax(reverse_prediction, dim=1).tolist()[0]
            entail_score = max(prediction_dist[2], reverse_prediction_dist[2])
            contradict_score = max(prediction_dist[0], reverse_prediction_dist[0])
            
            return (nli_label, entail_score)
            