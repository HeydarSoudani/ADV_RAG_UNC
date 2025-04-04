
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
        expected_answer: str = None,
        generator: Generator = None,
        node_value: float = None,
        max_depth_allowed: int = None,
        
        # --- My Actions ---------------------
        direct_answer: str = None,
        retrieved_document: str = None,
        subquestion: str = None,
        
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
                        direct_answer,
                        retrieved_document,
                        node_value,
                        subquestion,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        expected_answer,
                        max_depth_allowed,
                        
                    ]
                )
                
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        expected_answer,
                        max_depth_allowed,
                        retrieved_document,
                        subquestion,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        node_value,
                        direct_answer
                    ]
                )
                
            elif node_type is Node_Type.RETRIEVED_DOC:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        expected_answer,
                        max_depth_allowed,
                        node_value,
                        direct_answer,
                        subquestion,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        retrieved_document,
                    ]
                )

            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        question_id,
                        user_question,
                        expected_answer,
                        max_depth_allowed,
                        node_value,
                        direct_answer,
                        retrieved_document
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        parent,
                        subquestion,
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
        self.direct_answer = direct_answer
        self.retrieved_document = retrieved_document
        self.subquestion = subquestion
        
        if parent is None:  # root
            self.verbose = verbose
            self.question_id = question_id
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.expected_answer = expected_answer
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.question_id = parent.question_id
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.enable_potential_score = parent.enable_potential_score


        #! keep track of paraphrasing
        #! record number of subquestions till now

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "qid": question_id}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)
            
            if node_type is Node_Type.RETRIEVED_DOC:
                self.solution_trace[max(self.solution_trace.keys())+1] = {"document": retrieved_document}
            elif node_type is Node_Type.SUBQUESTION:
                self.solution_trace[max(self.solution_trace.keys())+1] = {"subquestion": subquestion}
            elif node_type is Node_Type.DIRECT_ANSWER:
                self.solution_trace[max(self.solution_trace.keys())+1] = {"answer": direct_answer, "value": node_value}
            
            
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
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.RETRIEVED_DOC: "RD",
            Node_Type.SUBQUESTION: "SQ"
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        
        #! Action Functions
        def do_action_generate_direct_answers():
            print(f"---- Generating direct answers for node {self.id}...")
            direct_answer, value = self.generator.generate_direct_answers(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.DIRECT_ANSWER,
                    direct_answer=direct_answer,
                    node_value=value,
                )
            )
        
        def do_action_document_retrieval():
            print(f"---- Retrieving documents for node {self.id}...")
            doc = self.generator.generate_retrieve_docs(solution_trace=self.solution_trace)
            self.children.append(
                Reasoning_MCTS_Node(
                    parent=self,
                    depth=self.depth + 1,
                    node_type=Node_Type.RETRIEVED_DOC,
                    retrieved_document=doc,
                )
            )

        def do_action_generate_subquestions():
            print(f"---- Generating subquestions for node {self.id}...")
            if self.node_type is Node_Type.USER_QUESTION:
                query = self.user_question
            elif self.node_type is Node_Type.SUBQUESTION:
                query = self.subquestion
            
            subquery_list = self.generator.generate_query_decomposition(query=query)
            for sq in subquery_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SUBQUESTION,
                        subquestion=sq,
                    )
                )
        
        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            do_action_generate_direct_answers() # A1
            do_action_document_retrieval()  # A2
            do_action_generate_subquestions() # A3

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")

        elif self.node_type is Node_Type.RETRIEVED_DOC:
            do_action_generate_direct_answers() # A1

        elif self.node_type is Node_Type.SUBQUESTION:
            do_action_generate_direct_answers() # A1
            do_action_document_retrieval()  # A2
            
        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (
            self.node_type is Node_Type.SUBQUESTION and reach_terminal_subquestion(self.subquestion, self.user_question)
        ) or self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

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

