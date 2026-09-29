# SPEC.md Composition based llm pipeline


## Overview
- to create an app to study that is functional. Clarity and simplicity desired over complex optimization.


### Goals
- using composition instead of inheritance.
- use Pytorch
- use pipeline
- be able to create a simple LLM from scratch
- to train a pre-made LLM model


### Non-Goals
- create an tokenizer
    - use already well-establish tokenizers instead.
    - HuggingFace's tokenizers for the BPE tokenizer
- work only on computer without gpu
    - training suffers.
- we don't implement forward logic as a method that "does everything." 
    - Instead, forward calls pipeline steps. The class is just a parameter holder.

### Acceptance Criteria
- runs on computer with gpu

### Training sets
- Salesforce/wikitext, config wikitext-103-raw-v1
- roneneldan/TinyStories on HuggingFace.

### Future additions
- interaction with application using python rich. https://github.com/Textualize/rich