# chatgpt_utils.py

from pddl import parse_domain, parse_problem
from typing import List, Dict, Any, Set, Collection, Union, Tuple, Literal
from pydantic import create_model, BaseModel, Field
import base64
import requests


class PDDLDomainWrapper:
    """
    A wrapper class for parsing and handling PDDL domain files.
    """

    def __init__(self, domain_file: str):
        """
        Initialize the domain wrapper by parsing the domain file.

        Args:
            domain_file (str): Path to the PDDL domain file.
        """
        self.domain = parse_domain(domain_file)
        self.actions = self.domain.actions
        self.types = self.domain.types

        # Filter out predicates that are actually action names
        action_names = {action.name for action in self.actions}
        self.predicates = frozenset(
            pred for pred in self.domain.predicates if pred.name not in action_names
        )

    def get_predicates_string(self) -> str:
        """
        Get a string representation of the domain's predicates with their argument types.

        Returns:
            str: String representation of predicates with types.
        """
        return _print_predicates_with_types(self.predicates)


class PDDLProblemWrapper:
    """
    A wrapper class for parsing and handling PDDL problem files.
    """

    def __init__(self, problem_file: str, domain: PDDLDomainWrapper):
        """
        Initialize the problem wrapper by parsing the problem file.

        Args:
            problem_file (str): Path to the PDDL problem file.
            domain (PDDLDomainWrapper): The associated domain wrapper.
        """
        self.problem = parse_problem(problem_file)
        self.domain = domain
        self.objects = self.problem.objects
        self.init_state = self.problem.init
        self.goal = self.problem.goal


def _print_predicates_with_types(predicates: Collection) -> str:
    """
    Generate a string representation of predicates with their argument types.

    Args:
        predicates (Collection): A collection of predicates.

    Returns:
        str: String representation of predicates with types.
    """
    result = ""
    for predicate in sorted(predicates, key=lambda p: p.name):
        if predicate.arity == 0:
            result += f"({predicate.name}) "
        else:
            result += f"({predicate.name}"
            for term in predicate.terms:
                if len(term.type_tags) > 1:
                    types_str = ' '.join(sorted(term.type_tags))
                    result += f" ?{term.name} - (either {types_str})"
                elif term.type_tags:
                    result += f" ?{term.name} - {sorted(term.type_tags)[0]}"
                else:
                    result += f" ?{term.name}"
            result += ") "
    return result.strip()


def get_object_schema_from_pddl(file_path: str) -> Tuple[BaseModel, Tuple[str, ...]]:
    """
    Generate a Pydantic schema for objects based on the PDDL domain file.

    Args:
        file_path (str): Path to the PDDL domain file.

    Returns:
        Tuple[BaseModel, Tuple[str, ...]]: A tuple containing the ObjectList model and available types.
    """
    # Create domain wrapper
    domain_wrapper = PDDLDomainWrapper(file_path)

    # Get list of types from the domain
    list_types = tuple(domain_wrapper.types)

    # Create a Literal type for object types
    ObjectType = Literal[list_types]

    class Object(BaseModel):
        name: str
        type: ObjectType = Field(..., description="Must be one of the valid types")

    class ObjectList(BaseModel):
        objects: List[Object]

    return ObjectList, list_types


def create_classes_from_dict(class_dict: Dict[str, List[str]]) -> Dict[str, BaseModel]:
    """
    Dynamically create Pydantic models based on a dictionary mapping class names to available names.

    Args:
        class_dict (Dict[str, List[str]]): Mapping from class names to lists of valid names.

    Returns:
        Dict[str, BaseModel]: Dictionary of dynamically created Pydantic models.
    """
    classes = {}
    for class_name, available_names in class_dict.items():
        attributes = {
            "name": (Literal[tuple(available_names)], ...)
        }
        model_name = class_name.capitalize()
        classes[class_name] = create_model(model_name, **attributes)
    return classes


def create_dynamic_models(
    predicates_structured: Dict[str, Any],
    types_available_classes_mapping: Dict[str, List[str]]
) -> Tuple[Dict[str, BaseModel], BaseModel, Dict[str, BaseModel]]:
    """
    Create dynamic Pydantic models for predicates, atoms, and types based on provided structures.

    Args:
        predicates_structured (Dict[str, Any]): Structure of predicates.
        types_available_classes_mapping (Dict[str, List[str]]): Mapping from types to class names.

    Returns:
        Tuple[Dict[str, BaseModel], BaseModel, Dict[str, BaseModel]]: Predicate classes, Atoms model, generated classes.
    """
    # Generate classes for types
    generated_classes = create_classes_from_dict(types_available_classes_mapping)

    predicate_classes = {}
    for predicate_name, fields_info in predicates_structured.items():
        model_name = predicate_name.capitalize()
        fields = {'predicate_type': (Literal[model_name], Field(default=model_name))}

        for field_name, field_type in fields_info.items():
            if isinstance(field_type, str) and field_type.lower() in generated_classes:
                fields[field_name] = (generated_classes[field_type.lower()], ...)
            else:
                fields[field_name] = (Any, ...)

        predicate_classes[model_name] = create_model(model_name, **fields)

    PredicateType = Union[tuple(predicate_classes.values())]

    class Atoms(BaseModel):
        grounded_predicates: List[PredicateType] = Field(default_factory=list)

        class Config:
            arbitrary_types_allowed = True

    return predicate_classes, Atoms, generated_classes


def encode_image_base64(image_source: str) -> str:
    """
    Encode an image from a local file or URL to base64 format.

    Args:
        image_source (str): Path to the local image file or URL.

    Returns:
        str: Base64 encoded string of the image.

    Raises:
        ValueError: If the image cannot be retrieved from the provided source.
    """
    if image_source.startswith(('http://', 'https://')):
        # Encode image from URL
        response = requests.get(image_source)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    else:
        # Encode image from local file
        try:
            with open(image_source, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise ValueError(f"Image file not found: {image_source}")
