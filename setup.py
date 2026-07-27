from setuptools import setup, find_packages

setup(
    name="symbolizer",
    version="0.1.0",
    description="SYMBOLIZER: VLM-based symbolic grounding and planning via structured JSON output",
    author="Anonymous",
    url="https://github.com/anonymous/symbolizer",
    packages=find_packages(include=["symbolizer*"]),
    install_requires=[
        "python-dotenv",
        "Pillow",
        "tiktoken",
        "pydantic>=2.9",
        "openai>=1.54",
        "requests",
        "inflect",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "gym",
        "PyYAML",
        "pddl",
        "mistralai>=1.0,<2",
        "google-genai",
        "google-generativeai",
        "unified-planning",
        "pyperplan",
        "httpx>=0.28",
        "imageio",
        # NOTE: pddlgym is intentionally NOT listed. This repo ships a *modified*
        # fork under ./pddlgym (adds hanoi_color rendering + blocks/hanoi env
        # registration) that must be installed editable FIRST:
        #     pip install -e ./pddlgym
        # Listing PyPI "pddlgym" would shadow the fork with the stock package.
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
