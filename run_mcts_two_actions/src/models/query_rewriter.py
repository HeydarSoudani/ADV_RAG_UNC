import re
import torch


def get_rewritten_queries(text):
    pattern = re.compile(r"<rewritten_query>(.*?)</rewritten_query>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches
    else:
        return None


# Based on https://github.com/kyriemao/LLM4CS
class QueryRewriter:
    def __init__(self, args, generator, tokenizer) -> None:
        self.args = args
        self.generator = generator
        self.tokenizer = tokenizer
        
        self.examples = [
            {
                "history": [
                    ("think", "I need to find the place of birth of Jacques Cassini's father. I'll search for it."),
                    ("query", "Jacques Cassini's father"),
                    ("think", "I need to find the place of birth of Jacques Cassini's father, Giovanni Domenico Cassini. I'll search for it.")
                ],
                "original": "place of birth of Giovanni Domenico Cassini",
                "rewritten": [
                    "Where was Giovanni Domenico Cassini born?",
                    "Birthplace of Giovanni Domenico Cassini",
                    "Giovanni Domenico Cassini place of birth",
                    "Which city was Giovanni Domenico Cassini born in?",
                    "Location of Giovanni Domenico Cassini’s birth",
                    "Giovanni Domenico Cassini’s hometown",
                    "City of birth for Giovanni Domenico Cassini",
                    "Giovanni Domenico Cassini birth location",
                    "In which country was Giovanni Domenico Cassini born?",
                    "Origin of Giovanni Domenico Cassini’s birth",
                ] 
            },
            {
                "history": [
                    ("think", "I need to find the year when Princess Basma bint Talal's nephew became king. I'll search for it."),
                    ("query", "Princess Basma bint Talal's nephew became king"),
                    ("think", "I found out that Princess Basma bint Talal's nephew is King Abdullah II of Jordan. Now I need to find the year when he became king."), 
                ],
                "original": "King Abdullah II of Jordan became king",
                "rewritten": [
                    "When did King Abdullah II ascend to the throne of Jordan",
                    "Year King Abdullah II took over as king of Jordan",
                    "Date King Abdullah II became Jordan’s monarch",
                    "King Abdullah II coronation year",
                    "In what year did Abdullah II become king of Jordan",
                    "Time King Abdullah II was officially made king",
                    "What year was King Abdullah II crowned in Jordan",
                    "King Abdullah II's succession to the Jordanian throne",
                    "When was Abdullah II declared king of Jordan",
                    "Year King Abdullah II assumed kingship of Jordan",
                ] 
            }
        ]
    
    def get_instruction(self, history, original_search_query, n=5):
        # --
        input_text = ''
        input_text += "You are an expert query rewriter model."
        input_text += f"Given an original search query, and the pervious search queries and their corresponding thought leading to the search query in the session, generate {n} semantically diverse and effective paraphrased search queries that capture the same intent but use different wording or structure."
        input_text += "These rewritten queries should be suitable for improving search engine results by covering various phrasings a user might employ."
        input_text += "Do not add extra information in the new queries."
        
        # --
        input_text += "Here are some examples:\n"
        for example in self.examples:
            input_text += "History:\n"
            for (key, value) in example["history"]:
                input_text += f"<{key}> {value} </{key}>\n"
            input_text += '\n'
            input_text += f"<original_query> {example['original']} </original_query>\n"
            
            for pr_query in example['rewritten'][:n]:
                input_text += f"<rewritten_query> {pr_query} </rewritten_query>\n"
            input_text += '\n\n'

        # --
        input_text += "History:\n"
        for (key, value) in history:
            input_text += f"<{key}> {value} </{key}>\n"
        input_text += '\n'
        input_text += f"<original_query> {original_search_query.strip()} </original_query>\n"

        return input_text

    def generate(self,
        input_text,
        max_new_tokens = 1024,
        num_return:int = 1,
        temperature:float = 1.0,
        do_sample:bool = True
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.generator.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
        return generated_texts
    
    def inference(self, qid, original_sq, history, repeat=5):
        sq_prompt = self.get_instruction(history, original_sq, n=repeat)
        sq_output = self.generate(sq_prompt, temperature=0.7)[0]
        rewritten_queries = get_rewritten_queries(sq_output)
        
        # check if paraphrased_queries are None
        if rewritten_queries == None:
            print(f"Rewritten queries are not provided for query {qid} ...")
            for i in range(self.args.retry):
                print(f"Rewritten queries, try {i+1} ...")
                sq_output = self.generate(sq_prompt, temperature=1.0)[0]
                rewritten_queries = get_rewritten_queries(sq_output)
                if rewritten_queries != None:
                    break
            else:
                print(f"Failed to generate 'Rewritten queries' after all retries for query {qid}!!!")
                rewritten_queries = []
        
    
        # Check if the number of rewritten_queries is equal to "repeat"
        if rewritten_queries is None:
            rewritten_queries = []

        max_iterations = 10
        iteration = 0
        while len(rewritten_queries) != repeat and iteration < max_iterations:
            remaining = repeat - len(rewritten_queries)        
            extra_prompt = self.get_instruction(original_sq, n=remaining)
            extra_output = self.generate(extra_prompt, temperature=1.0)[0]
            extra_queries = get_rewritten_queries(extra_output)

            if extra_queries:
                rewritten_queries.extend(extra_queries)
                rewritten_queries = rewritten_queries[:repeat]  # trim if over
            else:
                print(f"Failed to generate extra queries on iteration {iteration + 1}")
            iteration += 1
        if len(rewritten_queries) != repeat:
            print(f"Warning: Only generated {len(rewritten_queries)} queries out of {repeat} after {iteration} iterations.")
    
        return rewritten_queries
