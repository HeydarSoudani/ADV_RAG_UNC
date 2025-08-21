
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DebertaForSequenceClassification, DebertaTokenizer

class SelfDetection:
    def __init__(
        self,
        output_type: str = "entropy",
        method_for_similarity: str = "semantic",
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        prompt_for_generating_question=SELF_DETECTION_QUESTION_PROMPT,
        system_prompt=SELF_DETECTION_SYSTEM_PROMPT,
        prompt_for_entailment: str = ENTAILMENT_PROMPT,
        system_prompt_for_entailment: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt=SELF_DETECTION_USER_PROMPT,
        entailment_model_device="cuda",
        question_max_new_tokens=128,
        question_temperature=1.0,
        number_of_questions=5,
    ):
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.model_for_entailment = model_for_entailment

        if (model_for_entailment is None or tokenizer_for_entailment is None) and method_for_similarity == "semantic":
            self.model_for_entailment = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(entailment_model_device)
            self.tokenizer_for_entailment = DebertaTokenizer.from_pretrained("microsoft/deberta-large-mnli")

        self.number_of_questions = number_of_questions
        self.prompt_for_generating_question = prompt_for_generating_question
        self.system_prompt = system_prompt
        self.prompt_for_entailment = prompt_for_entailment
        self.system_prompt_for_entailment = system_prompt_for_entailment
        self.question_max_new_tokens = question_max_new_tokens
        self.question_temperature = question_temperature
        self.user_prompt = user_prompt
        
        if output_type not in ["entropy", "consistency"]:
            raise ValueError("output_type should be either 'entropy' or 'consistency'")
        self.output_type = output_type

        if method_for_similarity not in ["generation", "semantic", "jaccard"]:
            raise ValueError("method_for_similarity should be either 'generation' or 'semantic' or 'jaccard'")
        self.method_for_similarity = method_for_similarity


    def __call__(self, sampled_gen_dict, prediction, context):
        question = sampled_gen_dict['question']
        generated_texts = sampled_gen_dict["generated_texts"]
        generated_tokens_texts = sampled_gen_dict["tokens_text"]
        logprobs = sampled_gen_dict["logprobs"]
        score = 0
        return {
            "confidence": score,
            "uncertainty": -score
        } 