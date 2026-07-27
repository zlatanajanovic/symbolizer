from pddlgym.utils import run_demo
import pddlgym
import matplotlib
import imageio
from pddl import parse_domain as _pddl_parse_domain, parse_problem
from pddl.logic import Predicate, constants, variables
from pddl.core import Domain, Problem
from pddl.action import Action
from pddl.formatter import domain_to_string, problem_to_string
from pddl.requirements import Requirements
import re
import tempfile
from pathlib import Path


_MISSING_REQ_RE = re.compile(r"Missing PDDL requirement,\s*(:[a-zA-Z0-9\-]+)\s*not found")


def _inject_requirements(domain_text: str, requirements: set[str]) -> str:
    reqs_sorted = " ".join(sorted(requirements))
    req_block_re = re.compile(r"\(\s*:requirements\b([^)]*)\)", re.IGNORECASE)
    m = req_block_re.search(domain_text)
    if m:
        existing = set(re.findall(r":[a-zA-Z0-9\-]+", m.group(1)))
        merged = existing | requirements
        merged_text = " ".join(sorted(merged))
        return req_block_re.sub(f"(:requirements {merged_text})", domain_text, count=1)

    # No explicit :requirements block; inject one after domain declaration.
    domain_decl_re = re.compile(
        r"(\(\s*define\s*\(\s*domain\s+[^\)]+\)\s*)",
        re.IGNORECASE | re.MULTILINE,
    )
    m = domain_decl_re.search(domain_text)
    if not m:
        return domain_text
    insert = f"{m.group(1)}\n  (:requirements {reqs_sorted})\n"
    return domain_text[: m.start(1)] + insert + domain_text[m.end(1) :]


def parse_domain(domain_file: str):
    """Wrapper around pddl.parse_domain that handles the 'object' root type.

    The pddl library (v0.4.x) does not recognise 'object' as a valid type
    when used in predicate/action parameters, even though it is the implicit
    PDDL root type.  We work around this by monkey-patching *both* validation
    sites: _check_types_in_has_terms_objects (line-81 style) and
    TypeChecker._check_types_are_available (line-218 style).
    """
    def _parse_with_object_patch(path: str):
        try:
            return _pddl_parse_domain(path)
        except Exception as e:
            if "object" not in str(e):
                raise

        import pddl._validation as _val

        # Patch 1: _check_types_in_has_terms_objects (validates atomic expressions)
        _orig_check = _val._check_types_in_has_terms_objects

        def _patched_check(has_terms_objects, available_types):
            return _orig_check(has_terms_objects, available_types | {"object"})

        # Patch 2: TypeChecker._check_types_are_available (validates terms in
        # predicates, actions, quantified conditions, etc.)
        _orig_tc = _val.TypeChecker._check_types_are_available

        def _patched_tc(self, type_tags, what):
            # Inject 'object' as always-valid (implicit PDDL root type)
            patched_tags = type_tags
            if not self._types.all_types.issuperset(type_tags):
                self._types._all_types = self._types._all_types | {"object"}
            return _orig_tc(self, patched_tags, what)

        _val._check_types_in_has_terms_objects = _patched_check
        _val.TypeChecker._check_types_are_available = _patched_tc
        try:
            return _pddl_parse_domain(path)
        finally:
            _val._check_types_in_has_terms_objects = _orig_check
            _val.TypeChecker._check_types_are_available = _orig_tc

    try:
        return _parse_with_object_patch(domain_file)
    except Exception as first_exc:
        # SSR examples sometimes include domains that use quantifiers but miss
        # the explicit requirement declaration. Patch requirements on the fly.
        domain_text = Path(domain_file).read_text(encoding="utf-8")
        required = set(re.findall(r":[a-zA-Z0-9\-]+", str(first_exc)))
        missing_match = _MISSING_REQ_RE.search(str(first_exc))
        if missing_match:
            required.add(missing_match.group(1).lower())
        if not required:
            raise

        patched_text = domain_text
        tmpdir = tempfile.mkdtemp(prefix="ssr_parse_domain_")
        tmp_path = Path(tmpdir) / "domain_requirements_patch.pddl"
        for _ in range(6):
            patched_text = _inject_requirements(patched_text, required)
            tmp_path.write_text(patched_text, encoding="utf-8")
            try:
                return _parse_with_object_patch(str(tmp_path))
            except Exception as exc:
                missing = _MISSING_REQ_RE.search(str(exc))
                if not missing:
                    raise
                required.add(missing.group(1).lower())
        raise
from typing import List, Dict, Any, Set, Collection
from dataclasses import dataclass
from typing import List, Literal
from pydantic import create_model, BaseModel, Field, ValidationError
from enum import Enum
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import requests
from typing import List, Union, Literal, Dict, Any


#classes to load pddl domain and problem files
class PDDLDomainWrapper:
    def __init__(self, domain_file: str):
        self.domain = parse_domain(domain_file)
        self.objects = self._extract_objects()
        self.predicates = self.domain.predicates
        self.actions = self.domain.actions
        self.types = self.domain.types
        
        action_names = set(action.name for action in self.actions)
        filtered_predicates = frozenset(pred for pred in self.domain.predicates if pred.name not in action_names)
        self.predicates = filtered_predicates

        
    def _extract_objects(self):
        return [obj for obj in self.domain.constants]

    def get_predicates_string(self):
        return _print_predicates_with_types(self.predicates)

class PDDLProblemWrapper:
    def __init__(self, problem_file: str, domain: PDDLDomainWrapper):
        self.problem = parse_problem(problem_file)
        self.domain = domain
        self.objects = self.problem.objects
        self.init_state = self.problem.init
        self.goal = self.problem.goal
        
        
#Functions: Get Objects schema and types from pddl 
#Functions
def _print_predicates_with_types(predicates: Collection):
    result = ""
    for p in sorted(predicates):
        if p.arity == 0:
            result += f"({p.name})"
        else:
            result += f"({p.name}"
            for t in p.terms:
                if len(t.type_tags) > 1:
                    result += f" ?{t.name} - (either {' '.join(sorted(t.type_tags))})"
                else:
                    result += (
                        f" ?{t.name} - {sorted(t.type_tags)[0]}"
                        if t.type_tags
                        else f" ?{t.name}"
                    )
            result += ") "
        result += " "
    return result.strip()

def get_object_schema_from_pddl(file_path):
    # Create domain and problem wrappers
    domain_wrapper = PDDLDomainWrapper(file_path)

    class Type(BaseModel):
        name: str


    list_of_types: List[Type] = []
    for _type in domain_wrapper.types:
        list_of_types.append(Type(name=_type))

    # Always include implicit PDDL root type.
    # Many domains omit it in :types but predicates still use '?x - object'.
    if "object" not in [t.name for t in list_of_types]:
        list_of_types.append(Type(name="object"))

    # Handle untyped PDDL domains (no :types declaration)
    if not list_of_types:
        list_of_types = [Type(name="object")]

    list_types  = tuple([type.name for type in list_of_types])
    # Create an Enum from the list_types
    ObjectType = Literal[tuple(list_types)]
    class Object(BaseModel):
        name: str
        type: Literal[list_types] = Field(..., description="Must be one of the valid types")
    class ObjectList(BaseModel):
        objects: List[Object]
    
    return ObjectList, list_types    


def get_object_schema_from_problem_pddl(domain_file: str, problem_file: str):
    """Build an ObjectList schema where object names are constrained to those
    defined in the PDDL problem file.  This lets the VLM pick from the exact
    vocabulary instead of free-form naming."""
    domain_wrapper = PDDLDomainWrapper(domain_file)

    list_of_types = set(t for t in domain_wrapper.types) or {"object"}
    list_of_types.add("object")

    problem = parse_problem(problem_file)
    # Collect object names and build (name -> type) mapping
    obj_names = []
    obj_name_to_type = {}
    for obj in problem.objects:
        obj_name = obj.name
        # pddl library stores type tags as a frozenset
        obj_type = list(obj.type_tags)[0] if obj.type_tags else "object"
        obj_names.append(obj_name)
        obj_name_to_type[obj_name] = obj_type
        list_of_types.add(obj_type)  # ensure all object types are in schema

    list_types = tuple(sorted(list_of_types))

    if not obj_names:
        # Fallback to free-form schema when problem has no objects
        return get_object_schema_from_pddl(domain_file)

    valid_names = tuple(sorted(set(obj_names)))

    Object = create_model(
        "Object",
        name=(Literal[valid_names], Field(..., description="Must be one of the valid object names")),
        type=(Literal[list_types], Field(..., description="Must be one of the valid types")),
    )
    class ObjectList(BaseModel):
        objects: List[Object]

    return ObjectList, list_types, valid_names, obj_name_to_type

#Functions: Get grounded Atoms
def create_classes_from_dict(class_dict: Dict[str, List[str]]) -> Dict[str, BaseModel]:
    classes = {}
    for class_name, available_names in class_dict.items():
        if not available_names:
            continue
        attributes = {
            "name": (Literal[tuple(available_names)], ...)
        }
        classes[class_name] = create_model(class_name.capitalize(), **attributes)
    return classes

def create_dynamic_models(predicates_structured: Dict[str, Any], types_available_classes_mapping: Dict[str, List[str]], include_negation: bool = False):
    generated_classes = create_classes_from_dict(types_available_classes_mapping)

    predicate_classes = {}
    for key, value in predicates_structured.items():
        class_name = key.capitalize()
        # Keep predicate_type explicit in the JSON schema so constrained decoding
        # cannot emit empty predicate objects for zero-arg predicates.
        fields = {'predicate_type': (Literal[class_name], ...)}

        if include_negation:
            fields['is_negated'] = (bool, Field(default=False, description="Set to true if this goal atom is negated, i.e., (not (predicate args...))"))

        for subkey, subvalue in value.items():
            if isinstance(subvalue, str) and subvalue.lower() in generated_classes:
                fields[subkey] = (generated_classes[subvalue.lower()], ...)
            else:
                fields[subkey] = (Any, ...)
        
        predicate_classes[class_name] = create_model(class_name, **fields)
    
    if not predicate_classes:
        # No valid predicates — return an Atoms model that accepts an empty list
        class Atoms(BaseModel):
            grounded_predicates: List[Any] = Field(default_factory=list)
            class Config:
                arbitrary_types_allowed = True
        return predicate_classes, Atoms, generated_classes

    PredicateType = Union[tuple(predicate_classes.values())]
    class Atoms(BaseModel):
        grounded_predicates: List[PredicateType] = Field(default_factory=list)

        class Config:
            arbitrary_types_allowed = True
    return predicate_classes, Atoms, generated_classes

import base64
import requests
from io import BytesIO
from PIL import Image

def encode_image_base64_from_url(image_url: str, scale: bool = True, max_size: int = 512) -> str:
    """Encode an image retrieved from a remote url to base64 format, optionally scaling it."""
    with requests.get(image_url) as response:
        response.raise_for_status()
        img_data = response.content

    if scale:
        img_data = _scale_image_bytes(img_data, max_size)

    return base64.b64encode(img_data).decode('utf-8')

def encode_image_base64_from_local(image_path: str, scale: bool = True, max_size: int = 512) -> str:
    """Encode an image retrieved from a local file to base64 format, optionally scaling it."""
    print("Image Path from local")
    print(image_path)
    with open(image_path, 'rb') as image_file:
        img_data = image_file.read()

    if scale:
        img_data = _scale_image_bytes(img_data, max_size)

    return base64.b64encode(img_data).decode('utf-8')

def encode_image_base64(image_path: str, scale: bool = True, max_size: int = 512) -> str:
    """Encode an image from a local file to base64 format, optionally scaling it."""
    print("Encode image base64")
    print(image_path)
    print(scale)
    print(max_size)
    return encode_image_base64_from_local(image_path, scale=scale, max_size=max_size)

def _scale_image_bytes(img_bytes: bytes, max_size: int = 512) -> bytes:
    """Scale image bytes to max_size x max_size if necessary, maintaining aspect ratio."""
    # Open image from bytes
    image = Image.open(BytesIO(img_bytes))
    # Check if scaling is needed
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    # Save back to bytes
    output = BytesIO()
    image.save(output, format=image.format or 'PNG')
    output.seek(0)
    return output.read()
