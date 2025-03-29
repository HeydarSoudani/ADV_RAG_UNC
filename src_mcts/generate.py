

class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator) -> None:
        self.evaluator = evaluator
        
        self.num_subquestions = args.num_subquestions
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score
        self.mcts_num_last_votes = args.mcts_num_last_votes
        
        # Actions' prompts
        self.a1_direct_answer_prompt = read_text()
        self.a2_