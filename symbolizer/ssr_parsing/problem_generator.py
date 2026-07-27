# symbolizer/symbolizer/ssr_parsing/problem_generator.py
"""
PDDL Problem Generator - Interface and utilities for generating PDDL problems.

This module provides the infrastructure for converting SSR outputs to valid
PDDL problem files that can be solved by planners like Fast Downward.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import re


@dataclass
class PDDLObject:
    """Represents a PDDL object with name and type."""
    name: str
    type: str
    
    def __hash__(self):
        return hash((self.name, self.type))
    
    def __eq__(self, other):
        if not isinstance(other, PDDLObject):
            return False
        return self.name == other.name and self.type == other.type
    
    def to_pddl(self) -> str:
        """Return PDDL representation."""
        return f"{self.name} - {self.type}"


@dataclass
class PDDLPredicate:
    """Represents a grounded PDDL predicate."""
    name: str
    args: List[str]
    negated: bool = False
    
    def __hash__(self):
        return hash((self.name, tuple(self.args), self.negated))
    
    def __eq__(self, other):
        if not isinstance(other, PDDLPredicate):
            return False
        return (self.name == other.name and 
                self.args == other.args and 
                self.negated == other.negated)
    
    def to_pddl(self) -> str:
        """Generate PDDL string for this predicate."""
        pred_str = f"({self.name} {' '.join(self.args)})"
        return f"(not {pred_str})" if self.negated else pred_str


@dataclass
class PDDLProblem:
    """Represents a complete PDDL problem specification."""
    domain_name: str
    problem_name: str
    objects: List[PDDLObject]
    init: List[PDDLPredicate]
    goal: List[PDDLPredicate]
    metric: Optional[str] = None  # Optional optimization metric
    
    def to_pddl(self) -> str:
        """Generate PDDL problem file content."""
        lines = [
            f"(define (problem {self.problem_name})",
            f"  (:domain {self.domain_name})",
            "  (:objects",
        ]
        
        # Group objects by type
        by_type: Dict[str, List[str]] = {}
        for obj in self.objects:
            by_type.setdefault(obj.type, []).append(obj.name)
        
        for obj_type in sorted(by_type.keys()):
            names = sorted(by_type[obj_type])
            lines.append(f"    {' '.join(names)} - {obj_type}")
        lines.append("  )")
        
        # Init state
        lines.append("  (:init")
        for pred in sorted(self.init, key=lambda p: (p.name, tuple(p.args))):
            lines.append(f"    {pred.to_pddl()}")
        lines.append("  )")
        
        # Goal specification
        if len(self.goal) == 1 and not self.goal[0].negated:
            # Single positive goal - no 'and' needed
            lines.append(f"  (:goal {self.goal[0].to_pddl()})")
        else:
            lines.append("  (:goal (and")
            for pred in sorted(self.goal, key=lambda p: (p.name, tuple(p.args))):
                lines.append(f"    {pred.to_pddl()}")
            lines.append("  ))")
        
        # Optional metric
        if self.metric:
            lines.append(f"  (:metric {self.metric})")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def save(self, filepath: str) -> None:
        """Save problem to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_pddl())
    
    @classmethod
    def from_pddl_file(cls, filepath: str) -> 'PDDLProblem':
        """Parse a PDDL problem file."""
        with open(filepath, 'r') as f:
            content = f.read()
        return cls.from_pddl_string(content)
    
    @classmethod
    def from_pddl_string(cls, content: str) -> 'PDDLProblem':
        """Parse PDDL problem from string content."""
        # Extract domain name
        domain_match = re.search(r':domain\s+(\S+)\)', content)
        domain_name = domain_match.group(1) if domain_match else "unknown"
        
        # Extract problem name
        problem_match = re.search(r'problem\s+(\S+)\)', content)
        problem_name = problem_match.group(1) if problem_match else "problem"
        
        # Extract objects
        objects = []
        objects_match = re.search(r':objects\s*(.*?)\)', content, re.DOTALL)
        if objects_match:
            objects_text = objects_match.group(1)
            # Parse "obj1 obj2 - type" format
            current_names = []
            tokens = objects_text.split()
            i = 0
            while i < len(tokens):
                token = tokens[i].strip()
                if token == '-':
                    # Next token is the type
                    if i + 1 < len(tokens):
                        obj_type = tokens[i + 1].strip()
                        for name in current_names:
                            objects.append(PDDLObject(name, obj_type))
                        current_names = []
                        i += 2
                    else:
                        i += 1
                elif token and not token.startswith('(') and not token.startswith(')'):
                    current_names.append(token)
                    i += 1
                else:
                    i += 1
        
        # Extract init predicates
        init = []
        init_match = re.search(r':init\s*(.*?)\)\s*\(:goal', content, re.DOTALL)
        if init_match:
            init_text = init_match.group(1)
            init = cls._parse_predicates(init_text)
        
        # Extract goal predicates
        goal = []
        # Try to find goal section - look for balanced parentheses
        goal_start = content.find(':goal')
        if goal_start != -1:
            # Find the goal content by counting parentheses
            goal_section = content[goal_start:]
            # Skip ":goal" and find opening paren
            paren_start = goal_section.find('(')
            if paren_start != -1:
                depth = 0
                end_idx = paren_start
                for i, c in enumerate(goal_section[paren_start:], paren_start):
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                        if depth == 0:
                            end_idx = i
                            break
                goal_text = goal_section[paren_start:end_idx+1]
                goal = cls._parse_predicates(goal_text)
        
        return cls(
            domain_name=domain_name,
            problem_name=problem_name,
            objects=objects,
            init=init,
            goal=goal
        )
    
    @staticmethod
    def _parse_predicates(text: str) -> List[PDDLPredicate]:
        """Parse predicates from PDDL text using balanced parentheses matching."""
        predicates = []
        i = 0
        
        while i < len(text):
            # Find opening parenthesis
            if text[i] == '(':
                # Find the matching closing parenthesis
                depth = 1
                start = i
                j = i + 1
                while j < len(text) and depth > 0:
                    if text[j] == '(':
                        depth += 1
                    elif text[j] == ')':
                        depth -= 1
                    j += 1
                
                # Extract the content between parentheses
                content = text[start+1:j-1].strip()
                
                # Check if this is a simple predicate (no nested parens)
                if '(' not in content:
                    # Parse predicate name and args
                    parts = content.split()
                    if parts:
                        name = parts[0]
                        args = parts[1:]
                        
                        # Skip keywords
                        if name.lower() not in ('and', 'or', 'not', 'forall', 'exists', 'when', 'imply'):
                            # Check for negation by looking at preceding text
                            negated = False
                            pre_text = text[:start].strip()
                            if pre_text.endswith('not'):
                                negated = True
                            
                            predicates.append(PDDLPredicate(name, args, negated))
                else:
                    # Has nested parens - check if it's (not (...))
                    if content.startswith('not '):
                        # Recursively parse the inner predicate and mark as negated
                        inner_predicates = PDDLProblem._parse_predicates(content[4:])
                        for p in inner_predicates:
                            p.negated = True
                            predicates.append(p)
                    else:
                        # Recursively parse nested content (e.g., (and ...) )
                        inner_predicates = PDDLProblem._parse_predicates(content)
                        predicates.extend(inner_predicates)
                
                i = j
            else:
                i += 1
        
        return predicates


class ProblemGenerator(ABC):
    """Abstract base class for PDDL problem generators.
    
    Different simulators and environments may require different approaches
    to extract state information and generate PDDL problems. Subclasses
    implement the specifics for each environment.
    """
    
    @abstractmethod
    def generate_from_gt_state(self, state: Any, goal_spec: Any) -> PDDLProblem:
        """Generate problem from simulator ground truth state.
        
        Args:
            state: Environment-specific state representation
            goal_spec: Environment-specific goal specification
            
        Returns:
            PDDLProblem ready to be serialized
        """
        pass
    
    @abstractmethod
    def generate_from_image(
        self, 
        image: np.ndarray, 
        goal_text: str,
        domain_file: str
    ) -> PDDLProblem:
        """Generate problem from image using SSR (no ground truth).
        
        This is the key capability - using visual state recognition
        to generate PDDL problems from raw images.
        
        Args:
            image: RGB image of the environment state
            goal_text: Natural language or PDDL goal specification
            domain_file: Path to the PDDL domain file
            
        Returns:
            PDDLProblem generated from visual recognition
        """
        pass
    
    @abstractmethod
    def get_domain_name(self) -> str:
        """Return the PDDL domain name for this generator."""
        pass


def compare_problems(
    generated: PDDLProblem, 
    ground_truth: PDDLProblem
) -> Dict[str, float]:
    """Compare a generated PDDL problem against ground truth.
    
    Args:
        generated: Problem generated by SSR pipeline
        ground_truth: Ground truth problem from simulator
        
    Returns:
        Dictionary with comparison metrics
    """
    # Object comparison
    gen_objects = {(o.name, o.type) for o in generated.objects}
    gt_objects = {(o.name, o.type) for o in ground_truth.objects}
    
    obj_intersection = len(gen_objects & gt_objects)
    obj_precision = obj_intersection / len(gen_objects) if gen_objects else 0
    obj_recall = obj_intersection / len(gt_objects) if gt_objects else 0
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0
    
    # Init comparison (predicates)
    gen_init = {(p.name, tuple(p.args)) for p in generated.init}
    gt_init = {(p.name, tuple(p.args)) for p in ground_truth.init}
    
    init_intersection = len(gen_init & gt_init)
    init_precision = init_intersection / len(gen_init) if gen_init else 0
    init_recall = init_intersection / len(gt_init) if gt_init else 0
    init_f1 = 2 * init_precision * init_recall / (init_precision + init_recall) if (init_precision + init_recall) > 0 else 0
    
    # Goal comparison
    gen_goal = {(p.name, tuple(p.args), p.negated) for p in generated.goal}
    gt_goal = {(p.name, tuple(p.args), p.negated) for p in ground_truth.goal}
    
    goal_intersection = len(gen_goal & gt_goal)
    goal_precision = goal_intersection / len(gen_goal) if gen_goal else 0
    goal_recall = goal_intersection / len(gt_goal) if gt_goal else 0
    goal_f1 = 2 * goal_precision * goal_recall / (goal_precision + goal_recall) if (goal_precision + goal_recall) > 0 else 0
    
    # Jaccard similarity
    obj_jaccard = len(gen_objects & gt_objects) / len(gen_objects | gt_objects) if (gen_objects | gt_objects) else 0
    init_jaccard = len(gen_init & gt_init) / len(gen_init | gt_init) if (gen_init | gt_init) else 0
    goal_jaccard = len(gen_goal & gt_goal) / len(gen_goal | gt_goal) if (gen_goal | gt_goal) else 0
    
    return {
        'object_precision': obj_precision,
        'object_recall': obj_recall,
        'object_f1': obj_f1,
        'object_jaccard': obj_jaccard,
        'init_precision': init_precision,
        'init_recall': init_recall,
        'init_f1': init_f1,
        'init_jaccard': init_jaccard,
        'goal_precision': goal_precision,
        'goal_recall': goal_recall,
        'goal_f1': goal_f1,
        'goal_jaccard': goal_jaccard,
    }


def validate_pddl_syntax(problem: PDDLProblem, val_binary: str = "external/VAL/build/bin/Parser") -> Tuple[bool, str]:
    """Validate PDDL problem syntax using VAL parser.
    
    Args:
        problem: PDDLProblem to validate
        val_binary: Path to VAL Parser binary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import subprocess
    import tempfile
    
    # Write problem to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as f:
        f.write(problem.to_pddl())
        problem_file = f.name
    
    try:
        result = subprocess.run(
            [val_binary, problem_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr or result.stdout
    except FileNotFoundError:
        return False, f"VAL binary not found at {val_binary}"
    except subprocess.TimeoutExpired:
        return False, "Validation timed out"
    finally:
        Path(problem_file).unlink(missing_ok=True)


# Convenience exports
__all__ = [
    'PDDLObject',
    'PDDLPredicate', 
    'PDDLProblem',
    'ProblemGenerator',
    'compare_problems',
    'validate_pddl_syntax',
]
