from copy import deepcopy
from typing import List, Dict, Tuple

from src_mcts.MCTS_backbone import MCTS_Node
from src_mcts.generate_node import Generator
from utils.mcts_utils import (
    Node_Type,
    reach_terminal_subquestion
)



class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        
        # --- For instantiating root node ---
        question_id: str = None,
        user_question: str = None,
        gt_answer: List[str] = None,
        gt_reasoning_steps: List[str] = None,
        generator: Generator = None,
        node_value: float = None,
        max_depth_allowed: int = None,
        
        # --- My Actions ---------------------
        think: str = None,
        search_query: str = None,
        retrieved_documents: List[str] = None,
        answer: str = None,
        
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
        
        
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()
        
        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value > 0, breakpoint()
        
            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        think,
                        search_query,
                        retrieved_documents,
                        answer,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        max_depth_allowed,
                    ]
                )
        
            elif node_type is Node_Type.THINK_SERACH:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        node_value,
                        answer,   
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        think,
                        search_query,
                        retrieved_documents,
                    ]
                )
            
            elif node_type is Node_Type.THINK_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        gt_answer,
                        gt_reasoning_steps,
                        max_depth_allowed,
                        search_query,
                        retrieved_documents,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        node_value,
                        think,
                        answer,
                    ]
                )
        
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()
        
        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.think = think
        self.search_query = search_query
        self.retrieved_documents = retrieved_documents
        self.answer = answer
        
        if parent is None:  # root
            self.verbose = verbose
            self.question_id = question_id
            self.user_question = user_question
            self.gt_answer = gt_answer
            self.gt_reasoning_steps = gt_reasoning_steps
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.question_id = parent.question_id
            self.user_question = parent.user_question
            self.gt_answer = parent.gt_answer
            self.gt_reasoning_steps = parent.gt_reasoning_steps
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.enable_potential_score = parent.enable_potential_score
        
        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {"user_question": user_question, "qid": question_id, "ground_truth": gt_answer, "reasoning_steps": gt_reasoning_steps}
            }
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)
            
            if node_type is Node_Type.THINK_SERACH:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "think_search": {"think": think, "search_query": search_query, "retrieved_documents": retrieved_documents}
                }
            
            elif node_type is Node_Type.THINK_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1] = {
                    "think_answer": {"think": think, "answer": answer, "value": node_value}
                }
        
        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUESTION
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUESTION
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers
    
    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "UQ",
            Node_Type.THINK_SERACH: "TS",
            Node_Type.THINK_ANSWER: "TA",
        }
        return f"{type2str[self.node_type]}-{self.id}"
    
    def _create_children(self):
        #! Action Functions
        def do_action_think_search():
            print(f"---- Generating think search for node {self.id}...")
            think, search_query, retrieved_docs = self.generator.generate_think_search(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.THINK_SERACH,
                    think=think,
                    search_query=search_query,
                    retrieved_documents=retrieved_docs
                )
            )
        
        def do_action_think_answer():
            print(f"---- Generating think answer for node {self.id}...")
            think, answer, value = self.generator.generate_think_answer(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.THINK_ANSWER,
                    think=think,
                    answer=answer,
                    node_value=value
                )
            )
    
        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            do_action_think_search()
            do_action_think_answer()
        elif self.node_type is Node_Type.THINK_SERACH:
            do_action_think_search()
            do_action_think_answer()
        elif self.node_type is Node_Type.THINK_ANSWER:
            raise ValueError("THINK_ANSWER node cannot create children!!")    
        
        assert self.children
        return self.children
        
    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.THINK_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.THINK_ANSWER

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION

